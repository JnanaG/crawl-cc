from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


DecisionType = Literal["accept", "repair", "review", "drop"]


class RouteResult(BaseModel):
    channel: str
    route_decision: str
    template_version: str = "v1"
    confidence: float = 0.0
    reason: str = ""


class DedupResult(BaseModel):
    is_duplicate: bool
    duplicate_type: str = "none"
    matched_record_id: str | None = None
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)


class QualityResult(BaseModel):
    quality_score: float
    issues: list[str] = Field(default_factory=list)
    repair_suggestion: str = ""
    dimensions: dict[str, float] = Field(default_factory=dict)
    issue_groups: dict[str, list[str]] = Field(default_factory=dict)
    quality_tier: str = "unknown"
    rag_readiness: float = 0.0
    training_readiness: float = 0.0


class DecisionResult(BaseModel):
    decision: DecisionType
    confidence: float
    reason: str
    next_action: str = ""


class GovernanceResult(BaseModel):
    trace_id: str
    batch_id: str
    series_id: str
    route_result: RouteResult
    rule_dedup_result: DedupResult
    semantic_dedup_result: DedupResult
    quality_result: QualityResult
    decision_result: DecisionResult
    metadata: dict[str, Any] = Field(default_factory=dict)
    audit_logs: list[dict[str, Any]] = Field(default_factory=list)
