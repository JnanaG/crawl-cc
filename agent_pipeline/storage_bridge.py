from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from agent_pipeline.types import GovernanceResult


def ensure_governance_dirs(base_dir: str = "data") -> dict[str, str]:
    dirs = {
        "audit": os.path.join(base_dir, "audit", "agent_pipeline"),
        "review": os.path.join(base_dir, "review", "agent_pipeline"),
        "repair": os.path.join(base_dir, "repair", "agent_pipeline"),
        "feedback": os.path.join(base_dir, "feedback", "agent_pipeline"),
        "manifests": os.path.join(base_dir, "state", "agent_pipeline"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def load_manifest(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_manifest(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def persist_governance_result(
    result: GovernanceResult,
    clean_record: dict[str, Any],
    markdown_text: str,
    dirs: dict[str, str],
) -> None:
    input_summary = _build_input_summary(clean_record, markdown_text)
    step_metrics = _flatten_step_metrics(result)
    audit_row = {
        "trace_id": result.trace_id,
        "batch_id": result.batch_id,
        "series_id": result.series_id,
        "decision": result.decision_result.decision,
        "input_summary": input_summary,
        "metadata": result.metadata,
        "route": result.route_result.model_dump(),
        "rule_dedup": result.rule_dedup_result.model_dump(),
        "semantic_dedup": result.semantic_dedup_result.model_dump(),
        "quality": result.quality_result.model_dump(),
        "step_metrics": step_metrics,
        "audit_logs": result.audit_logs,
    }
    append_jsonl(os.path.join(dirs["audit"], "governance_audit.jsonl"), audit_row)
    for step_metric in step_metrics:
        append_jsonl(os.path.join(dirs["audit"], "governance_steps.jsonl"), step_metric)
    _update_governance_summary(result, step_metrics, dirs)

    decision = result.decision_result.decision
    if decision == "review":
        append_jsonl(
            os.path.join(dirs["review"], "review_queue.jsonl"),
            {
                "trace_id": result.trace_id,
                "batch_id": result.batch_id,
                "series_id": result.series_id,
                "reason": result.decision_result.reason,
                "quality_tier": result.quality_result.quality_tier,
                "issue_groups": result.quality_result.issue_groups,
                "clean_record": clean_record,
                "markdown_text": markdown_text,
            },
        )
    elif decision == "repair":
        append_jsonl(
            os.path.join(dirs["repair"], "repair_queue.jsonl"),
            {
                "trace_id": result.trace_id,
                "batch_id": result.batch_id,
                "series_id": result.series_id,
                "reason": result.decision_result.reason,
                "quality_tier": result.quality_result.quality_tier,
                "issue_groups": result.quality_result.issue_groups,
                "repair_suggestion": result.quality_result.repair_suggestion,
                "clean_record": clean_record,
            },
        )


def persist_governance_failure(
    trace_id: str,
    batch_id: str,
    clean_record: dict[str, Any],
    markdown_text: str,
    error: Exception,
    dirs: dict[str, str],
) -> None:
    input_summary = _build_input_summary(clean_record, markdown_text)
    series = clean_record.get("series", {}) or {}
    error_row = {
        "trace_id": trace_id,
        "batch_id": batch_id,
        "series_id": str(series.get("series_id") or ""),
        "decision": "failed",
        "input_summary": input_summary,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    append_jsonl(os.path.join(dirs["audit"], "governance_failures.jsonl"), error_row)


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


def _flatten_step_metrics(result: GovernanceResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in result.audit_logs:
        rows.append(
            {
                "trace_id": result.trace_id,
                "batch_id": result.batch_id,
                "series_id": result.series_id,
                "step": item.get("step"),
                "at": item.get("at"),
                "status": item.get("status", "ok"),
                "duration_ms": item.get("duration_ms", 0.0),
                "input_summary": item.get("input_summary", {}),
                "output_summary": item.get("output_summary", {}),
                "error": item.get("error"),
            }
        )
    return rows


def _update_governance_summary(
    result: GovernanceResult,
    step_metrics: list[dict[str, Any]],
    dirs: dict[str, str],
) -> None:
    summary_path = os.path.join(dirs["audit"], "governance_summary.json")
    summary = load_manifest(summary_path)
    summary.setdefault("total_records", 0)
    summary.setdefault("decisions", {})
    summary.setdefault("quality", {"sum": 0.0, "rag_sum": 0.0, "training_sum": 0.0})
    summary.setdefault("step_stats", {})

    summary["total_records"] += 1
    decision = result.decision_result.decision
    summary["decisions"][decision] = summary["decisions"].get(decision, 0) + 1
    summary["quality"]["sum"] += float(result.quality_result.quality_score)
    summary["quality"]["rag_sum"] += float(result.quality_result.rag_readiness)
    summary["quality"]["training_sum"] += float(result.quality_result.training_readiness)

    for row in step_metrics:
        step = str(row.get("step") or "unknown")
        step_summary = summary["step_stats"].setdefault(
            step,
            {"count": 0, "total_duration_ms": 0.0, "max_duration_ms": 0.0, "error_count": 0},
        )
        duration = float(row.get("duration_ms") or 0.0)
        step_summary["count"] += 1
        step_summary["total_duration_ms"] += duration
        step_summary["max_duration_ms"] = max(float(step_summary["max_duration_ms"]), duration)
        if row.get("status") != "ok":
            step_summary["error_count"] += 1

    total_records = max(int(summary["total_records"]), 1)
    summary["averages"] = {
        "quality_score": round(summary["quality"]["sum"] / total_records, 4),
        "rag_readiness": round(summary["quality"]["rag_sum"] / total_records, 4),
        "training_readiness": round(summary["quality"]["training_sum"] / total_records, 4),
    }
    summary["updated_at"] = datetime.now().isoformat()
    save_manifest(summary_path, summary)
