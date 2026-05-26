from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from agent_pipeline.types import DedupResult, QualityResult, RouteResult
from quality.data_quality import DataQualityEngine


def detect_route_by_rules(clean_record: dict[str, Any]) -> RouteResult:
    source = str(clean_record.get("source", "") or "unknown")
    entity_type = str(clean_record.get("entity_type", "") or "unknown")
    series = clean_record.get("series", {}) or {}
    stats = clean_record.get("stats", {}) or {}

    confidence = 0.9 if source == "dongchedi" and entity_type == "car_series" else 0.4
    if stats.get("model_count", 0) <= 0:
        confidence -= 0.2

    return RouteResult(
        channel=source,
        route_decision="default_clean_pipeline",
        template_version="dongchedi_series_v1",
        confidence=max(confidence, 0.0),
        reason=(
            f"source={source}, entity_type={entity_type}, "
            f"series_id={series.get('series_id') or 'unknown'}"
        ),
    )


def normalize_text_for_dedup(markdown_text: str) -> str:
    return " ".join((markdown_text or "").strip().lower().split())


def compute_record_hashes(clean_record: dict[str, Any], markdown_text: str) -> dict[str, str]:
    series = clean_record.get("series", {}) or {}
    canonical_payload = {
        "source": clean_record.get("source"),
        "entity_type": clean_record.get("entity_type"),
        "series_id": series.get("series_id"),
        "series_name": series.get("series_name"),
        "markdown_text": markdown_text or "",
    }
    text_payload = markdown_text or ""
    normalized_text = normalize_text_for_dedup(text_payload)
    return {
        "content_hash": hashlib.md5(text_payload.encode("utf-8")).hexdigest(),
        "normalized_hash": hashlib.md5(normalized_text.encode("utf-8")).hexdigest(),
        "record_hash": hashlib.md5(
            json.dumps(canonical_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def run_rule_dedup(
    clean_record: dict[str, Any],
    markdown_text: str,
    dedup_manifest_path: str,
) -> tuple[DedupResult, dict[str, str]]:
    hashes = compute_record_hashes(clean_record, markdown_text)
    existing = {}
    if os.path.exists(dedup_manifest_path):
        try:
            with open(dedup_manifest_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    by_content = existing.get("by_content_hash", {}) or {}
    by_normalized = existing.get("by_normalized_hash", {}) or {}

    if hashes["content_hash"] in by_content:
        return (
            DedupResult(
                is_duplicate=True,
                duplicate_type="exact_content",
                matched_record_id=by_content[hashes["content_hash"]],
                confidence=1.0,
                evidence=["content_hash matched existing record"],
            ),
            hashes,
        )
    if hashes["normalized_hash"] in by_normalized:
        return (
            DedupResult(
                is_duplicate=True,
                duplicate_type="normalized_text",
                matched_record_id=by_normalized[hashes["normalized_hash"]],
                confidence=0.95,
                evidence=["normalized_hash matched existing record"],
            ),
            hashes,
        )

    return (
        DedupResult(
            is_duplicate=False,
            duplicate_type="none",
            confidence=0.1,
            evidence=["no hash-level duplicate detected"],
        ),
        hashes,
    )


def run_rule_quality(
    clean_record: dict[str, Any],
    markdown_text: str,
    quality_engine: DataQualityEngine,
) -> QualityResult:
    clean_qc = quality_engine.validate_clean_record(clean_record)
    token_like_length = max(len(markdown_text or ""), 1)
    score = 1.0
    issues: list[str] = []
    issue_groups: dict[str, list[str]] = {
        "completeness": [],
        "structure": [],
        "content": [],
        "freshness": [],
        "retrieval": [],
    }

    def add_issue(group: str, issue: str, penalty: float) -> None:
        nonlocal score
        issues.append(issue)
        issue_groups.setdefault(group, []).append(issue)
        score -= penalty

    for issue in clean_qc.get("issues", []):
        wrapped = f"clean_record:{issue}"
        if "缺失" in issue or "均缺失" in issue:
            add_issue("completeness", wrapped, 0.15)
        elif "乱码" in issue:
            add_issue("structure", wrapped, 0.18)
        else:
            add_issue("structure", wrapped, 0.15)

    for warning in clean_qc.get("warnings", []):
        wrapped = f"warning:{warning}"
        if "重复" in warning:
            add_issue("retrieval", wrapped, 0.04)
        else:
            add_issue("structure", wrapped, 0.03)

    if token_like_length < 400:
        add_issue("content", "markdown_too_short", 0.2)

    stats = clean_record.get("stats", {}) or {}
    series = clean_record.get("series", {}) or {}
    pricing = clean_record.get("pricing", {}) or {}
    models = clean_record.get("models", []) or []
    news = clean_record.get("news", []) or []
    images = clean_record.get("images", []) or []

    if stats.get("news_count", 0) <= 0:
        add_issue("freshness", "no_news_items", 0.05)
    if stats.get("model_count", 0) <= 0:
        add_issue("retrieval", "no_model_specs", 0.08)
    if not pricing.get("dealer_price_range") and not pricing.get("official_price_range"):
        add_issue("completeness", "missing_price_ranges", 0.08)
    if not series.get("brand_name"):
        add_issue("completeness", "missing_brand_name", 0.08)
    if len(images) <= 0:
        add_issue("content", "no_image_groups", 0.03)
    if len(news) <= 0:
        add_issue("freshness", "no_recent_news", 0.04)
    if len(models) <= 1:
        add_issue("retrieval", "model_coverage_low", 0.05)

    quality_score = max(round(score, 4), 0.0)
    structure_score = max(0.0, 1.0 - len(issue_groups.get("structure", [])) * 0.14)
    completeness_score = max(0.0, 1.0 - len(issue_groups.get("completeness", [])) * 0.16)
    freshness_score = max(0.0, min((stats.get("news_count", 0) or 0) / 5.0, 1.0))
    coverage_score = min(
        (
            (0.45 if stats.get("model_count", 0) > 0 else 0.0)
            + (0.25 if pricing.get("dealer_price_range") or pricing.get("official_price_range") else 0.0)
            + (0.15 if images else 0.0)
            + (0.15 if news else 0.0)
        ),
        1.0,
    )
    content_score = min(token_like_length / 2000.0, 1.0)
    rag_readiness = round((0.35 * coverage_score) + (0.25 * content_score) + (0.2 * structure_score) + (0.2 * completeness_score), 4)
    training_readiness = round((0.3 * completeness_score) + (0.25 * structure_score) + (0.2 * content_score) + (0.15 * freshness_score) + (0.1 * coverage_score), 4)

    if quality_score >= 0.85:
        quality_tier = "high"
    elif quality_score >= 0.65:
        quality_tier = "medium"
    elif quality_score >= 0.45:
        quality_tier = "low"
    else:
        quality_tier = "critical"

    repair_parts: list[str] = []
    if issue_groups["completeness"]:
        repair_parts.append("补齐品牌、价格区间等关键结构化字段")
    if issue_groups["retrieval"]:
        repair_parts.append("补充车型配置与可检索事实，提升 RAG 召回覆盖")
    if issue_groups["freshness"]:
        repair_parts.append("补采新闻/动态信息，增强时效性")
    if issue_groups["content"]:
        repair_parts.append("扩充正文与图片描述，减少低信息密度样本")
    repair_suggestion = "；".join(repair_parts)

    return QualityResult(
        quality_score=quality_score,
        issues=issues,
        repair_suggestion=repair_suggestion if quality_score < 0.85 else "",
        dimensions={
            "structure_score": round(structure_score, 4),
            "completeness_score": round(completeness_score, 4),
            "coverage_score": round(coverage_score, 4),
            "content_score": round(content_score, 4),
            "freshness_score": round(freshness_score, 4),
        },
        issue_groups={k: v for k, v in issue_groups.items() if v},
        quality_tier=quality_tier,
        rag_readiness=rag_readiness,
        training_readiness=training_readiness,
    )
