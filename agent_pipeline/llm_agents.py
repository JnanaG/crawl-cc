from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from agent_pipeline.types import QualityResult, RouteResult
from utils.llm_client import LLMClient


LLMResponseProvider = Callable[[str, str, str], str]


@dataclass
class AgentLLMConfig:
    enabled: bool = False
    provider: str = "ollama"
    model: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    timeout_sec: int = 90
    fail_open: bool = True
    route_skip_threshold: float = 0.8
    max_markdown_chars: int = 2400


def _clamp_score(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(score, 1.0))


def _normalize_issue_groups(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for key, raw_items in value.items():
        if not isinstance(key, str):
            continue
        if isinstance(raw_items, list):
            items = [str(item).strip() for item in raw_items if str(item).strip()]
        elif raw_items:
            items = [str(raw_items).strip()]
        else:
            items = []
        if items:
            normalized[key] = items
    return normalized


def _merge_issue_groups(base: dict[str, list[str]], extra: dict[str, list[str]]) -> dict[str, list[str]]:
    merged = {group: list(values) for group, values in (base or {}).items()}
    for group, values in (extra or {}).items():
        merged.setdefault(group, [])
        for value in values:
            if value not in merged[group]:
                merged[group].append(value)
    return {group: values for group, values in merged.items() if values}


def _quality_tier_from_score(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.65:
        return "medium"
    if score >= 0.45:
        return "low"
    return "critical"


def _extract_json_block(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("LLM 返回为空")

    fenced = re.findall(r"```json\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates = fenced + [text]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(candidate[index:])
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                continue
    raise ValueError(f"无法从 LLM 输出中解析 JSON: {text[:200]}")


class GovernanceLLMBridge:
    def __init__(
        self,
        config: AgentLLMConfig | None = None,
        response_provider: LLMResponseProvider | None = None,
    ):
        self.config = config or AgentLLMConfig()
        self.response_provider = response_provider
        self._client: LLMClient | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def _get_client(self) -> LLMClient:
        if self._client is None:
            self._client = LLMClient(
                provider=self.config.provider,
                model=self.config.model,
                api_base=self.config.api_base,
                api_key=self.config.api_key,
                timeout_sec=self.config.timeout_sec,
            )
        return self._client

    def _chat(self, agent_name: str, system_prompt: str, user_prompt: str) -> str:
        if self.response_provider is not None:
            return self.response_provider(agent_name, system_prompt, user_prompt)
        return self._get_client().chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.1)

    def route_refine(self, clean_record: dict[str, Any], base_route: RouteResult) -> tuple[RouteResult, dict[str, Any]]:
        metadata = {
            "llm_enabled": self.enabled,
            "llm_used": False,
            "agent": "route",
            "provider": self.config.provider,
            "model": self.config.model,
        }
        if not self.enabled:
            metadata["skip_reason"] = "llm_disabled"
            return base_route, metadata
        if base_route.confidence >= self.config.route_skip_threshold:
            metadata["skip_reason"] = "base_route_confident"
            return base_route, metadata

        series = clean_record.get("series", {}) or {}
        stats = clean_record.get("stats", {}) or {}
        system_prompt = (
            "你是数据治理中的渠道路由 Agent。"
            "你必须输出严格 JSON，不要输出解释。"
            "字段: channel, route_decision, template_version, confidence, reason。"
            "confidence 范围 0 到 1。"
        )
        user_prompt = json.dumps(
            {
                "task": "根据样本字段判断应走哪个渠道和模板路由。",
                "base_route": base_route.model_dump(),
                "sample": {
                    "source": clean_record.get("source"),
                    "entity_type": clean_record.get("entity_type"),
                    "series_id": series.get("series_id"),
                    "series_name": series.get("series_name"),
                    "brand_name": series.get("brand_name"),
                    "car_type": series.get("car_type"),
                    "model_count": stats.get("model_count"),
                    "news_count": stats.get("news_count"),
                },
                "guardrails": {
                    "prefer_base_when_uncertain": True,
                    "do_not_return_markdown": True,
                },
            },
            ensure_ascii=False,
            indent=2,
        )

        try:
            raw_text = self._chat("route", system_prompt, user_prompt)
            payload = _extract_json_block(raw_text)
            metadata["llm_used"] = True
            metadata["raw_response"] = raw_text
            metadata["parsed_payload"] = payload

            candidate = RouteResult(
                channel=str(payload.get("channel") or base_route.channel),
                route_decision=str(payload.get("route_decision") or base_route.route_decision),
                template_version=str(payload.get("template_version") or base_route.template_version),
                confidence=_clamp_score(payload.get("confidence"), default=max(base_route.confidence, 0.65)),
                reason=str(payload.get("reason") or "llm_route_refine"),
            )
            changed_core_fields = (
                candidate.channel != base_route.channel
                or candidate.route_decision != base_route.route_decision
                or candidate.template_version != base_route.template_version
            )
            if changed_core_fields and candidate.confidence < 0.7:
                metadata["fallback_reason"] = "llm_override_confidence_too_low"
                return base_route, metadata
            candidate.reason = f"llm_route_refine:{candidate.reason}"
            return candidate, metadata
        except Exception as exc:
            metadata["error"] = str(exc)
            metadata["fallback_reason"] = "llm_route_failed"
            if self.config.fail_open:
                return base_route, metadata
            raise

    def quality_refine(
        self,
        clean_record: dict[str, Any],
        markdown_text: str,
        base_quality: QualityResult,
    ) -> tuple[QualityResult, dict[str, Any]]:
        metadata = {
            "llm_enabled": self.enabled,
            "llm_used": False,
            "agent": "quality",
            "provider": self.config.provider,
            "model": self.config.model,
        }
        if not self.enabled:
            metadata["skip_reason"] = "llm_disabled"
            return base_quality, metadata

        series = clean_record.get("series", {}) or {}
        stats = clean_record.get("stats", {}) or {}
        excerpt = (markdown_text or "")[: self.config.max_markdown_chars]
        system_prompt = (
            "你是数据治理中的质量诊断 Agent。"
            "你必须输出严格 JSON，不要输出解释。"
            "字段: quality_score, rag_readiness, training_readiness, extra_issues, issue_groups, repair_suggestion, reason。"
            "其中 3 个分数范围都是 0 到 1。"
        )
        user_prompt = json.dumps(
            {
                "task": "在规则质检基础上补充质量诊断，但不要脱离现有规则结果胡乱改分。",
                "base_quality": base_quality.model_dump(),
                "sample": {
                    "series_id": series.get("series_id"),
                    "series_name": series.get("series_name"),
                    "brand_name": series.get("brand_name"),
                    "car_type": series.get("car_type"),
                    "model_count": stats.get("model_count"),
                    "news_count": stats.get("news_count"),
                    "markdown_excerpt": excerpt,
                },
                "guardrails": {
                    "keep_scores_close_to_base": True,
                    "focus_on_content_quality": True,
                    "do_not_return_markdown": True,
                },
            },
            ensure_ascii=False,
            indent=2,
        )

        try:
            raw_text = self._chat("quality", system_prompt, user_prompt)
            payload = _extract_json_block(raw_text)
            metadata["llm_used"] = True
            metadata["raw_response"] = raw_text
            metadata["parsed_payload"] = payload

            llm_quality_score = _clamp_score(payload.get("quality_score"), default=base_quality.quality_score)
            llm_rag = _clamp_score(payload.get("rag_readiness"), default=base_quality.rag_readiness)
            llm_training = _clamp_score(payload.get("training_readiness"), default=base_quality.training_readiness)
            final_quality_score = round((0.7 * base_quality.quality_score) + (0.3 * llm_quality_score), 4)
            final_rag = round((0.7 * base_quality.rag_readiness) + (0.3 * llm_rag), 4)
            final_training = round((0.7 * base_quality.training_readiness) + (0.3 * llm_training), 4)

            extra_issues_raw = payload.get("extra_issues") or []
            if isinstance(extra_issues_raw, list):
                extra_issues = [str(item).strip() for item in extra_issues_raw if str(item).strip()]
            elif extra_issues_raw:
                extra_issues = [str(extra_issues_raw).strip()]
            else:
                extra_issues = []

            merged_issues = list(base_quality.issues)
            for issue in extra_issues:
                if issue not in merged_issues:
                    merged_issues.append(issue)

            llm_issue_groups = _normalize_issue_groups(payload.get("issue_groups"))
            merged_issue_groups = _merge_issue_groups(base_quality.issue_groups, llm_issue_groups)
            repair_suggestion = str(payload.get("repair_suggestion") or "").strip() or base_quality.repair_suggestion
            result = QualityResult(
                quality_score=final_quality_score,
                issues=merged_issues,
                repair_suggestion=repair_suggestion,
                dimensions=dict(base_quality.dimensions),
                issue_groups=merged_issue_groups,
                quality_tier=_quality_tier_from_score(final_quality_score),
                rag_readiness=final_rag,
                training_readiness=final_training,
            )
            metadata["score_delta"] = round(final_quality_score - base_quality.quality_score, 4)
            metadata["reason"] = str(payload.get("reason") or "llm_quality_refine")
            return result, metadata
        except Exception as exc:
            metadata["error"] = str(exc)
            metadata["fallback_reason"] = "llm_quality_failed"
            if self.config.fail_open:
                return base_quality, metadata
            raise
