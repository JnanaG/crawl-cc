from __future__ import annotations

from typing import Any, TypedDict

from agent_pipeline.types import DecisionResult, DedupResult, QualityResult, RouteResult


class GovernanceState(TypedDict, total=False):
    trace_id: str
    batch_id: str
    clean_record: dict[str, Any]
    markdown_text: str
    metadata: dict[str, Any]
    input_summary: dict[str, Any]
    route_result: RouteResult
    rule_dedup_result: DedupResult
    semantic_dedup_result: DedupResult
    quality_result: QualityResult
    decision_result: DecisionResult
    audit_logs: list[dict[str, Any]]
    error: str
