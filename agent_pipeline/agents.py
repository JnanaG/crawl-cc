from __future__ import annotations

from typing import Any

from agent_pipeline.types import DecisionResult, DedupResult, QualityResult, RouteResult


def route_agent_refine(clean_record: dict[str, Any], base_route: RouteResult) -> RouteResult:
    stats = clean_record.get("stats", {}) or {}
    if base_route.confidence >= 0.8:
        return base_route
    return RouteResult(
        channel=base_route.channel,
        route_decision=base_route.route_decision,
        template_version=base_route.template_version,
        confidence=max(base_route.confidence, 0.75),
        reason=(
            f"route_agent_refine: model_count={stats.get('model_count', 0)}, "
            f"news_count={stats.get('news_count', 0)}"
        ),
    )


def semantic_dedup_agent(
    clean_record: dict[str, Any],
    markdown_text: str,
    base_result: DedupResult,
    semantic_search_result: dict[str, Any] | None = None,
    similarity_threshold: float = 0.94,
    same_series_threshold: float = 0.88,
) -> DedupResult:
    if base_result.is_duplicate:
        return base_result

    text = markdown_text or ""
    if text.count("暂无") >= 8:
        return DedupResult(
            is_duplicate=False,
            duplicate_type="none",
            confidence=0.15,
            evidence=["semantic dedup skipped: low-information record"],
        )

    semantic_search_result = semantic_search_result or {}
    hits = semantic_search_result.get("hits", []) or []
    if not hits:
        return DedupResult(
            is_duplicate=False,
            duplicate_type="none",
            confidence=0.25,
            evidence=["semantic search returned no candidates"],
        )

    series = clean_record.get("series", {}) or {}
    series_name = str(series.get("series_name") or "").strip().lower()
    best_hit = hits[0]
    best_name = str(best_hit.get("series_name") or "").strip().lower()
    best_score = float(best_hit.get("score") or 0.0)
    threshold = same_series_threshold if best_name and best_name == series_name else similarity_threshold

    if best_score >= threshold:
        duplicate_type = "semantic_same_series" if threshold == same_series_threshold else "semantic_similarity"
        return DedupResult(
            is_duplicate=True,
            duplicate_type=duplicate_type,
            matched_record_id=str(best_hit.get("series_id") or best_hit.get("series_name") or ""),
            confidence=best_score,
            evidence=[
                f"semantic similarity={best_score:.4f} >= threshold={threshold:.4f}",
                f"candidate={best_hit.get('series_name') or best_hit.get('series_id')}",
            ],
        )

    return DedupResult(
        is_duplicate=False,
        duplicate_type="none",
        confidence=max(best_score, 0.25),
        evidence=[
            f"best semantic similarity={best_score:.4f} < threshold={threshold:.4f}",
            f"candidate={best_hit.get('series_name') or best_hit.get('series_id')}",
        ],
    )


def quality_agent_refine(
    clean_record: dict[str, Any],
    markdown_text: str,
    base_quality: QualityResult,
) -> QualityResult:
    issues = list(base_quality.issues)
    issue_groups = {group: list(values) for group, values in base_quality.issue_groups.items()}
    score = base_quality.quality_score
    text = markdown_text or ""
    stats = clean_record.get("stats", {}) or {}
    models = clean_record.get("models", []) or []
    news = clean_record.get("news", []) or []
    dimensions = dict(base_quality.dimensions)

    def add_issue(group: str, issue: str, penalty: float) -> None:
        nonlocal score
        if issue not in issues:
            issues.append(issue)
        issue_groups.setdefault(group, [])
        if issue not in issue_groups[group]:
            issue_groups[group].append(issue)
        score -= penalty

    if text.count("暂无") >= 10:
        add_issue("content", "low_information_density", 0.15)
    if len(text.splitlines()) < 20:
        add_issue("structure", "section_count_too_low", 0.08)
    if stats.get("model_count", 0) >= 1 and len(models) == 0:
        add_issue("structure", "stats_models_inconsistent", 0.12)
    if stats.get("news_count", 0) >= 1 and len(news) == 0:
        add_issue("freshness", "stats_news_inconsistent", 0.1)
    if text.count("###") < 3:
        add_issue("retrieval", "section_heading_sparse", 0.06)

    final_score = max(round(score, 4), 0.0)
    dimensions["content_score"] = round(max(0.0, dimensions.get("content_score", 0.0) - (0.08 if "low_information_density" in issues else 0.0)), 4)
    dimensions["structure_score"] = round(max(0.0, dimensions.get("structure_score", 0.0) - (0.08 if "section_count_too_low" in issues else 0.0)), 4)
    rag_readiness = round(
        max(
            0.0,
            (0.4 * dimensions.get("coverage_score", 0.0))
            + (0.25 * dimensions.get("content_score", 0.0))
            + (0.2 * dimensions.get("structure_score", 0.0))
            + (0.15 * dimensions.get("freshness_score", 0.0)),
        ),
        4,
    )
    training_readiness = round(
        max(
            0.0,
            (0.35 * dimensions.get("completeness_score", 0.0))
            + (0.25 * dimensions.get("structure_score", 0.0))
            + (0.2 * dimensions.get("content_score", 0.0))
            + (0.2 * dimensions.get("freshness_score", 0.0)),
        ),
        4,
    )

    if final_score >= 0.85:
        quality_tier = "high"
    elif final_score >= 0.65:
        quality_tier = "medium"
    elif final_score >= 0.45:
        quality_tier = "low"
    else:
        quality_tier = "critical"

    repair_parts = []
    if issue_groups.get("structure"):
        repair_parts.append("修正统计字段与正文结构不一致的问题")
    if issue_groups.get("content"):
        repair_parts.append("扩充正文信息密度，减少'暂无'占比")
    if issue_groups.get("retrieval"):
        repair_parts.append("增加更清晰的章节标题和车型事实块，改善检索可切分性")
    if issue_groups.get("freshness"):
        repair_parts.append("补采新闻动态并校正 stats 计数")
    repair_suggestion = "；".join(repair_parts) or base_quality.repair_suggestion

    return QualityResult(
        quality_score=final_score,
        issues=issues,
        repair_suggestion=repair_suggestion if final_score < 0.85 else "",
        dimensions=dimensions,
        issue_groups={k: v for k, v in issue_groups.items() if v},
        quality_tier=quality_tier,
        rag_readiness=rag_readiness,
        training_readiness=training_readiness,
    )


def decision_agent(
    route_result: RouteResult,
    dedup_result: DedupResult,
    quality_result: QualityResult,
) -> DecisionResult:
    if dedup_result.is_duplicate:
        return DecisionResult(
            decision="drop",
            confidence=max(dedup_result.confidence, 0.85),
            reason=f"duplicate:{dedup_result.duplicate_type}",
            next_action="skip_processed_output",
        )

    if route_result.confidence < 0.6:
        return DecisionResult(
            decision="review",
            confidence=0.75,
            reason="route_confidence_low",
            next_action="send_to_review_queue",
        )

    if quality_result.quality_score < 0.55 or quality_result.rag_readiness < 0.45:
        return DecisionResult(
            decision="review",
            confidence=0.85,
            reason="quality_score_too_low",
            next_action="send_to_review_queue",
        )

    if quality_result.quality_score < 0.75 or quality_result.training_readiness < 0.65:
        return DecisionResult(
            decision="repair",
            confidence=0.7,
            reason="quality_score_need_repair",
            next_action="write_repair_candidate_and_keep_processed",
        )

    return DecisionResult(
        decision="accept",
        confidence=0.92,
        reason="passed_route_dedup_quality",
        next_action="write_processed_output",
    )
