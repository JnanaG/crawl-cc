from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from statistics import mean
from typing import Any


def load_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _series_key(item: dict[str, Any]) -> str:
    metadata = item.get("metadata", {}) or {}
    series_id = str(metadata.get("series_id") or "").strip()
    title = str(metadata.get("title") or "").strip()
    return series_id or title or "unknown_series"


def _series_name(item: dict[str, Any]) -> str:
    metadata = item.get("metadata", {}) or {}
    return str(metadata.get("title") or metadata.get("series_id") or "unknown_series").strip()


def _infer_readiness_scores(item: dict[str, Any]) -> tuple[float, float]:
    metadata = item.get("metadata", {}) or {}
    text = str(item.get("text") or "")
    quality_score = _safe_float(metadata.get("quality_score"), 0.0)
    route_confidence = _safe_float(metadata.get("route_confidence"), 0.0)
    model_count = _safe_int(metadata.get("model_count"))
    news_count = _safe_int(metadata.get("news_count"))
    has_brand = 1.0 if str(metadata.get("brand_name") or "").strip() else 0.0
    has_price = 1.0 if re.search(r"(指导价|成交价|\d+(?:\.\d+)?-\d+(?:\.\d+)?万)", text) else 0.0
    text_score = min(len(text) / 500.0, 1.0)
    coverage_score = min(
        (0.35 if model_count > 0 else 0.0)
        + (0.2 if news_count > 0 else 0.0)
        + (0.2 if has_brand else 0.0)
        + (0.25 if has_price else 0.0),
        1.0,
    )
    rag_readiness = round(
        min(1.0, (0.45 * quality_score) + (0.25 * coverage_score) + (0.15 * text_score) + (0.15 * route_confidence)),
        4,
    )
    training_readiness = round(
        min(1.0, (0.5 * quality_score) + (0.2 * coverage_score) + (0.15 * text_score) + (0.15 * route_confidence)),
        4,
    )
    return rag_readiness, training_readiness


def _get_rag_readiness(item: dict[str, Any]) -> float:
    metadata = item.get("metadata", {}) or {}
    if metadata.get("rag_readiness") not in (None, ""):
        return _safe_float(metadata.get("rag_readiness"))
    inferred_rag, _ = _infer_readiness_scores(item)
    return inferred_rag


def _get_training_readiness(item: dict[str, Any]) -> float:
    metadata = item.get("metadata", {}) or {}
    if metadata.get("training_readiness") not in (None, ""):
        return _safe_float(metadata.get("training_readiness"))
    _, inferred_training = _infer_readiness_scores(item)
    return inferred_training


def _is_asset_eligible(item: dict[str, Any], readiness_key: str, threshold: float) -> bool:
    metadata = item.get("metadata", {}) or {}
    if not str(item.get("text") or "").strip():
        return False
    if str(metadata.get("governance_decision") or "") not in {"accept", "repair"}:
        return False
    if str(metadata.get("dedup_duplicate_type") or "none") != "none":
        return False
    if _safe_float(metadata.get("route_confidence")) < 0.6:
        return False
    if readiness_key == "rag_readiness":
        readiness_value = _get_rag_readiness(item)
    else:
        readiness_value = _get_training_readiness(item)
    return readiness_value >= threshold


def _clone_with_asset_tags(item: dict[str, Any], asset_name: str, asset_score: float) -> dict[str, Any]:
    metadata = dict(item.get("metadata", {}) or {})
    if metadata.get("rag_readiness") in (None, ""):
        metadata["rag_readiness"] = _get_rag_readiness(item)
        metadata["rag_readiness_inferred"] = True
    if metadata.get("training_readiness") in (None, ""):
        metadata["training_readiness"] = _get_training_readiness(item)
        metadata["training_readiness_inferred"] = True
    metadata["asset_split"] = asset_name
    metadata["asset_score"] = round(asset_score, 4)
    return {
        **item,
        "metadata": metadata,
    }


def build_rag_corpus(items: list[dict[str, Any]], min_rag_readiness: float = 0.45) -> list[dict[str, Any]]:
    rag_rows: list[dict[str, Any]] = []
    for item in items:
        if not _is_asset_eligible(item, "rag_readiness", min_rag_readiness):
            continue
        metadata = item.get("metadata", {}) or {}
        asset_score = (
            0.6 * _get_rag_readiness(item)
            + 0.25 * _safe_float(metadata.get("quality_score"))
            + 0.15 * _safe_float(metadata.get("route_confidence"))
        )
        rag_rows.append(_clone_with_asset_tags(item, "rag_corpus", asset_score))
    return rag_rows


def build_training_corpus(
    items: list[dict[str, Any]],
    min_training_readiness: float = 0.6,
    min_quality_score: float = 0.65,
    max_chunks_per_series: int = 8,
) -> list[dict[str, Any]]:
    by_series: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for item in items:
        if not _is_asset_eligible(item, "training_readiness", min_training_readiness):
            continue
        metadata = item.get("metadata", {}) or {}
        quality_score = _safe_float(metadata.get("quality_score"))
        if quality_score < min_quality_score:
            continue
        training_score = (
            0.55 * _get_training_readiness(item)
            + 0.3 * quality_score
            + 0.15 * _get_rag_readiness(item)
        )
        by_series[_series_key(item)].append((training_score, item))

    training_rows: list[dict[str, Any]] = []
    for series_items in by_series.values():
        ranked = sorted(
            series_items,
            key=lambda pair: (
                -pair[0],
                _safe_int((pair[1].get("metadata", {}) or {}).get("chunk_index")),
            ),
        )
        for score, item in ranked[: max(1, max_chunks_per_series)]:
            training_rows.append(_clone_with_asset_tags(item, "training_corpus", score))
    return training_rows


def _detect_fact_types(series_name: str, merged_text: str, sample_metadata: dict[str, Any]) -> list[str]:
    fact_types = ["overview"]
    if re.search(r"(指导价|成交价|\d+(?:\.\d+)?-\d+(?:\.\d+)?万)", merged_text):
        fact_types.append("price")
    if _safe_int(sample_metadata.get("model_count")) > 0 or "配置" in merged_text or "车型" in merged_text:
        fact_types.append("model_specs")
    if any(keyword in merged_text for keyword in ["续航", "电耗", "油耗", "动力", "加速", "发动机"]):
        fact_types.append("performance")
    if _safe_int(sample_metadata.get("news_count")) > 0 or any(keyword in merged_text for keyword in ["资讯", "新闻", "上市", "改款"]):
        fact_types.append("freshness")
    if not series_name:
        return fact_types
    return fact_types


def _build_suggested_questions(series_name: str, fact_types: list[str]) -> list[dict[str, Any]]:
    if not series_name:
        return []

    suggestions: list[dict[str, Any]] = [
        {
            "question": f"{series_name}有哪些值得关注的核心亮点",
            "query_type": "概览类",
            "difficulty": "easy",
            "ground_truth_contexts": [series_name],
        }
    ]
    if "price" in fact_types:
        suggestions.append(
            {
                "question": f"{series_name}的官方指导价和成交价区间是多少",
                "query_type": "事实查询",
                "difficulty": "easy",
                "ground_truth_contexts": [series_name],
            }
        )
    if "model_specs" in fact_types:
        suggestions.append(
            {
                "question": f"{series_name}有哪些配置和车型版本可以选",
                "query_type": "配置查询",
                "difficulty": "medium",
                "ground_truth_contexts": [series_name],
            }
        )
    if "performance" in fact_types:
        suggestions.append(
            {
                "question": f"{series_name}的动力和能耗表现怎么样",
                "query_type": "性能查询",
                "difficulty": "medium",
                "ground_truth_contexts": [series_name],
            }
        )
    if "freshness" in fact_types:
        suggestions.append(
            {
                "question": f"{series_name}最近有什么新车型或市场动态",
                "query_type": "时效查询",
                "difficulty": "medium",
                "ground_truth_contexts": [series_name],
            }
        )
    return suggestions[:4]


def build_eval_candidates(
    items: list[dict[str, Any]],
    min_eval_readiness: float = 0.6,
    min_chunks_per_series: int = 2,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        metadata = item.get("metadata", {}) or {}
        if str(metadata.get("governance_decision") or "") not in {"accept", "repair"}:
            continue
        grouped[_series_key(item)].append(item)

    candidates: list[dict[str, Any]] = []
    for series_key, series_items in grouped.items():
        rag_scores = [_get_rag_readiness(item) for item in series_items]
        route_scores = [_safe_float((item.get("metadata", {}) or {}).get("route_confidence")) for item in series_items]
        quality_scores = [_safe_float((item.get("metadata", {}) or {}).get("quality_score")) for item in series_items]
        training_scores = [_get_training_readiness(item) for item in series_items]
        avg_rag = round(mean(rag_scores), 4) if rag_scores else 0.0
        avg_route = round(mean(route_scores), 4) if route_scores else 0.0
        if avg_rag < min_eval_readiness or avg_route < 0.6 or len(series_items) < min_chunks_per_series:
            continue

        ranked_items = sorted(
            series_items,
            key=lambda item: (
                -_get_rag_readiness(item),
                -_safe_float((item.get("metadata", {}) or {}).get("quality_score")),
            ),
        )
        sample_item = ranked_items[0]
        sample_metadata = sample_item.get("metadata", {}) or {}
        series_name = _series_name(sample_item)
        merged_text = "\n".join(str(item.get("text") or "") for item in ranked_items[:3])
        fact_types = _detect_fact_types(series_name, merged_text, sample_metadata)
        candidates.append(
            {
                "series_id": str(sample_metadata.get("series_id") or series_key),
                "series_name": series_name,
                "brand_name": str(sample_metadata.get("brand_name") or ""),
                "car_type": str(sample_metadata.get("car_type") or ""),
                "source": str(sample_metadata.get("source") or ""),
                "chunk_count": len(series_items),
                "avg_quality_score": round(mean(quality_scores), 4) if quality_scores else 0.0,
                "avg_rag_readiness": avg_rag,
                "avg_training_readiness": round(mean(training_scores), 4) if training_scores else 0.0,
                "available_fact_types": fact_types,
                "suggested_questions": _build_suggested_questions(series_name, fact_types),
                "supporting_preview": merged_text[:400],
                "metadata": {
                    "route_confidence_avg": avg_route,
                    "model_count": _safe_int(sample_metadata.get("model_count")),
                    "news_count": _safe_int(sample_metadata.get("news_count")),
                    "asset_split": "eval_candidates",
                },
            }
        )

    candidates.sort(key=lambda row: (-row["avg_rag_readiness"], -row["avg_quality_score"], row["series_name"]))
    return candidates


def split_processed_assets(
    items: list[dict[str, Any]],
    min_rag_readiness: float = 0.45,
    min_training_readiness: float = 0.6,
    min_quality_score: float = 0.65,
    min_eval_readiness: float = 0.6,
    max_training_chunks_per_series: int = 8,
    min_eval_chunks_per_series: int = 2,
) -> dict[str, Any]:
    rag_rows = build_rag_corpus(items=items, min_rag_readiness=min_rag_readiness)
    training_rows = build_training_corpus(
        items=items,
        min_training_readiness=min_training_readiness,
        min_quality_score=min_quality_score,
        max_chunks_per_series=max_training_chunks_per_series,
    )
    eval_candidates = build_eval_candidates(
        items=items,
        min_eval_readiness=min_eval_readiness,
        min_chunks_per_series=min_eval_chunks_per_series,
    )
    return {
        "rag_corpus": rag_rows,
        "training_corpus": training_rows,
        "eval_candidates": eval_candidates,
        "summary": {
            "input_rows": len(items),
            "rag_rows": len(rag_rows),
            "training_rows": len(training_rows),
            "eval_candidate_rows": len(eval_candidates),
            "rag_series": len({_series_key(item) for item in rag_rows}),
            "training_series": len({_series_key(item) for item in training_rows}),
            "eval_series": len({str(item.get('series_id') or '') for item in eval_candidates}),
            "thresholds": {
                "min_rag_readiness": min_rag_readiness,
                "min_training_readiness": min_training_readiness,
                "min_quality_score": min_quality_score,
                "min_eval_readiness": min_eval_readiness,
                "max_training_chunks_per_series": max_training_chunks_per_series,
                "min_eval_chunks_per_series": min_eval_chunks_per_series,
            },
        },
    }
