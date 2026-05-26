from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from typing import Any

from langgraph.graph import END, StateGraph

from agent_pipeline.agents import (
    decision_agent,
    quality_agent_refine,
    route_agent_refine,
    semantic_dedup_agent,
)
from agent_pipeline.llm_agents import AgentLLMConfig, GovernanceLLMBridge
from agent_pipeline.rules_bridge import detect_route_by_rules, run_rule_dedup, run_rule_quality
from agent_pipeline.semantic_dedup_store import SemanticDedupStore
from agent_pipeline.state import GovernanceState
from agent_pipeline.storage_bridge import (
    ensure_governance_dirs,
    load_manifest,
    persist_governance_failure,
    persist_governance_result,
    save_manifest,
)
from agent_pipeline.types import GovernanceResult
from quality.data_quality import DataQualityEngine


def _append_audit(state: GovernanceState, step: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    return state.get("audit_logs", []) + [
        {
            "step": step,
            "at": datetime.now().isoformat(),
            **payload,
        }
    ]


def _build_input_summary(clean_record: dict[str, Any], markdown_text: str) -> dict[str, Any]:
    series = clean_record.get("series", {}) or {}
    stats = clean_record.get("stats", {}) or {}
    return {
        "series_id": str(series.get("series_id") or ""),
        "series_name": str(series.get("series_name") or ""),
        "brand_name": str(series.get("brand_name") or ""),
        "car_type": str(series.get("car_type") or ""),
        "markdown_chars": len(markdown_text or ""),
        "markdown_lines": len((markdown_text or "").splitlines()),
        "model_count": int(stats.get("model_count", 0) or 0),
        "news_count": int(stats.get("news_count", 0) or 0),
        "image_group_count": int(stats.get("image_group_count", 0) or 0),
    }


class GovernanceOrchestrator:
    def __init__(
        self,
        base_dir: str = "data",
        *,
        enable_llm_agents: bool = False,
        agent_llm_provider: str = "ollama",
        agent_llm_model: str | None = None,
        agent_llm_api_base: str | None = None,
        agent_llm_api_key: str | None = None,
        llm_fail_open: bool = True,
        llm_response_provider=None,
        semantic_store: SemanticDedupStore | None = None,
    ):
        self.base_dir = base_dir
        self.dirs = ensure_governance_dirs(base_dir=base_dir)
        self.quality_engine = DataQualityEngine()
        self.semantic_store = semantic_store or SemanticDedupStore(base_dir=base_dir)
        self.llm_bridge = GovernanceLLMBridge(
            config=AgentLLMConfig(
                enabled=enable_llm_agents,
                provider=agent_llm_provider,
                model=agent_llm_model,
                api_base=agent_llm_api_base,
                api_key=agent_llm_api_key,
                fail_open=llm_fail_open,
            ),
            response_provider=llm_response_provider,
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GovernanceState)
        graph.add_node("route", self._route_node)
        graph.add_node("rule_dedup", self._rule_dedup_node)
        graph.add_node("semantic_dedup", self._semantic_dedup_node)
        graph.add_node("quality", self._quality_node)
        graph.add_node("decision", self._decision_node)
        graph.set_entry_point("route")
        graph.add_edge("route", "rule_dedup")
        graph.add_edge("rule_dedup", "semantic_dedup")
        graph.add_edge("semantic_dedup", "quality")
        graph.add_edge("quality", "decision")
        graph.add_edge("decision", END)
        return graph.compile()

    def _route_node(self, state: GovernanceState) -> GovernanceState:
        started = time.perf_counter()
        clean_record = state["clean_record"]
        rule_route = detect_route_by_rules(clean_record)
        heuristic_route = route_agent_refine(clean_record, rule_route)
        route_result, llm_meta = self.llm_bridge.route_refine(clean_record, heuristic_route)
        metadata = dict(state.get("metadata", {}))
        metadata.update(
            {
                "agent_llm_enabled": self.llm_bridge.enabled,
                "route_llm_used": llm_meta.get("llm_used", False),
                "route_llm_provider": llm_meta.get("provider"),
                "route_llm_model": llm_meta.get("model"),
            }
        )
        return {
            **state,
            "metadata": metadata,
            "route_result": route_result,
            "audit_logs": _append_audit(
                state,
                "route",
                {
                    "status": "ok",
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "input_summary": state.get("input_summary", {}),
                    "output_summary": {
                        "channel": route_result.channel,
                        "confidence": route_result.confidence,
                        "route_decision": route_result.route_decision,
                    },
                    "rule_route": rule_route.model_dump(),
                    "heuristic_route": heuristic_route.model_dump(),
                    "route_result": route_result.model_dump(),
                    "llm_agent": llm_meta,
                },
            ),
        }

    def _rule_dedup_node(self, state: GovernanceState) -> GovernanceState:
        started = time.perf_counter()
        manifest_path = os.path.join(self.dirs["manifests"], "dedup_manifest.json")
        result, hashes = run_rule_dedup(
            clean_record=state["clean_record"],
            markdown_text=state["markdown_text"],
            dedup_manifest_path=manifest_path,
        )
        metadata = dict(state.get("metadata", {}))
        metadata.update(hashes)
        return {
            **state,
            "metadata": metadata,
            "rule_dedup_result": result,
            "audit_logs": _append_audit(
                state,
                "rule_dedup",
                {
                    "status": "ok",
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "input_summary": {
                        "content_hash": hashes.get("content_hash"),
                        "normalized_hash": hashes.get("normalized_hash"),
                    },
                    "output_summary": {
                        "is_duplicate": result.is_duplicate,
                        "duplicate_type": result.duplicate_type,
                        "confidence": result.confidence,
                    },
                    "rule_dedup_result": result.model_dump(),
                    "hashes": hashes,
                },
            ),
        }

    def _semantic_dedup_node(self, state: GovernanceState) -> GovernanceState:
        started = time.perf_counter()
        semantic_search_result = self.semantic_store.find_candidates(
            clean_record=state["clean_record"],
            markdown_text=state["markdown_text"],
        )
        result = semantic_dedup_agent(
            clean_record=state["clean_record"],
            markdown_text=state["markdown_text"],
            base_result=state["rule_dedup_result"],
            semantic_search_result=semantic_search_result,
            similarity_threshold=self.semantic_store.similarity_threshold,
            same_series_threshold=self.semantic_store.same_series_threshold,
        )
        metadata = dict(state.get("metadata", {}))
        metadata["semantic_query_dim"] = semantic_search_result.get("query_dim", 0)
        metadata["semantic_candidate_count"] = len(semantic_search_result.get("hits", []) or [])
        metadata["semantic_top_hit_score"] = (
            semantic_search_result.get("hits", [{}])[0].get("score")
            if semantic_search_result.get("hits")
            else None
        )
        return {
            **state,
            "metadata": metadata,
            "semantic_dedup_result": result,
            "audit_logs": _append_audit(
                state,
                "semantic_dedup",
                {
                    "status": "ok",
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "input_summary": {
                        "candidate_count": len(semantic_search_result.get("hits", []) or []),
                        "query_dim": semantic_search_result.get("query_dim", 0),
                    },
                    "output_summary": {
                        "is_duplicate": result.is_duplicate,
                        "duplicate_type": result.duplicate_type,
                        "confidence": result.confidence,
                    },
                    "semantic_dedup_result": result.model_dump(),
                    "semantic_search_result": semantic_search_result,
                },
            ),
        }

    def _quality_node(self, state: GovernanceState) -> GovernanceState:
        started = time.perf_counter()
        base_quality = run_rule_quality(
            clean_record=state["clean_record"],
            markdown_text=state["markdown_text"],
            quality_engine=self.quality_engine,
        )
        heuristic_quality = quality_agent_refine(
            clean_record=state["clean_record"],
            markdown_text=state["markdown_text"],
            base_quality=base_quality,
        )
        result, llm_meta = self.llm_bridge.quality_refine(
            clean_record=state["clean_record"],
            markdown_text=state["markdown_text"],
            base_quality=heuristic_quality,
        )
        metadata = dict(state.get("metadata", {}))
        metadata.update(
            {
                "quality_llm_used": llm_meta.get("llm_used", False),
                "quality_llm_provider": llm_meta.get("provider"),
                "quality_llm_model": llm_meta.get("model"),
            }
        )
        return {
            **state,
            "metadata": metadata,
            "quality_result": result,
            "audit_logs": _append_audit(
                state,
                "quality",
                {
                    "status": "ok",
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "input_summary": state.get("input_summary", {}),
                    "output_summary": {
                        "quality_score": result.quality_score,
                        "quality_tier": result.quality_tier,
                        "rag_readiness": result.rag_readiness,
                        "training_readiness": result.training_readiness,
                        "issue_count": len(result.issues),
                    },
                    "base_quality": base_quality.model_dump(),
                    "heuristic_quality": heuristic_quality.model_dump(),
                    "quality_result": result.model_dump(),
                    "llm_agent": llm_meta,
                },
            ),
        }

    def _decision_node(self, state: GovernanceState) -> GovernanceState:
        started = time.perf_counter()
        result = decision_agent(
            route_result=state["route_result"],
            dedup_result=state["semantic_dedup_result"],
            quality_result=state["quality_result"],
        )
        return {
            **state,
            "decision_result": result,
            "audit_logs": _append_audit(
                state,
                "decision",
                {
                    "status": "ok",
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "input_summary": {
                        "quality_score": state["quality_result"].quality_score,
                        "rag_readiness": state["quality_result"].rag_readiness,
                        "training_readiness": state["quality_result"].training_readiness,
                        "is_duplicate": state["semantic_dedup_result"].is_duplicate,
                    },
                    "output_summary": {
                        "decision": result.decision,
                        "reason": result.reason,
                        "confidence": result.confidence,
                    },
                    "decision_result": result.model_dump(),
                },
            ),
        }

    def _update_manifests(
        self,
        result: GovernanceResult,
        metadata: dict[str, Any],
        clean_record: dict[str, Any],
        markdown_text: str,
    ) -> None:
        if result.decision_result.decision not in {"accept", "repair"}:
            return

        dedup_manifest_path = os.path.join(self.dirs["manifests"], "dedup_manifest.json")
        dedup_manifest = load_manifest(dedup_manifest_path)
        by_content = dedup_manifest.get("by_content_hash", {}) or {}
        by_normalized = dedup_manifest.get("by_normalized_hash", {}) or {}
        by_content[metadata["content_hash"]] = result.series_id
        by_normalized[metadata["normalized_hash"]] = result.series_id
        dedup_manifest["by_content_hash"] = by_content
        dedup_manifest["by_normalized_hash"] = by_normalized
        save_manifest(dedup_manifest_path, dedup_manifest)

        self.semantic_store.add_record(
            clean_record=clean_record,
            markdown_text=markdown_text,
            metadata=metadata,
        )

    def govern(
        self,
        clean_record: dict[str, Any],
        markdown_text: str,
        batch_id: str,
    ) -> GovernanceResult:
        series = clean_record.get("series", {}) or {}
        trace_id = str(uuid.uuid4())
        input_summary = _build_input_summary(clean_record, markdown_text)
        initial_state: GovernanceState = {
            "trace_id": trace_id,
            "batch_id": batch_id,
            "clean_record": clean_record,
            "markdown_text": markdown_text,
            "metadata": {
                "agent_llm_enabled": self.llm_bridge.enabled,
                "agent_llm_provider": self.llm_bridge.config.provider if self.llm_bridge.enabled else None,
                "agent_llm_model": self.llm_bridge.config.model if self.llm_bridge.enabled else None,
            },
            "input_summary": input_summary,
            "audit_logs": [],
        }
        governance_started = time.perf_counter()
        try:
            final_state = self.graph.invoke(initial_state)
        except Exception as exc:
            persist_governance_failure(
                trace_id=trace_id,
                batch_id=batch_id,
                clean_record=clean_record,
                markdown_text=markdown_text,
                error=exc,
                dirs=self.dirs,
            )
            raise
        governance_result = GovernanceResult(
            trace_id=trace_id,
            batch_id=batch_id,
            series_id=str(series.get("series_id") or ""),
            route_result=final_state["route_result"],
            rule_dedup_result=final_state["rule_dedup_result"],
            semantic_dedup_result=final_state["semantic_dedup_result"],
            quality_result=final_state["quality_result"],
            decision_result=final_state["decision_result"],
            metadata={
                **final_state.get("metadata", {}),
                "input_summary": input_summary,
                "governance_duration_ms": round((time.perf_counter() - governance_started) * 1000, 3),
            },
            audit_logs=final_state.get("audit_logs", []),
        )
        persist_governance_result(
            result=governance_result,
            clean_record=clean_record,
            markdown_text=markdown_text,
            dirs=self.dirs,
        )
        self._update_manifests(
            governance_result,
            final_state.get("metadata", {}),
            clean_record,
            markdown_text,
        )
        return governance_result
