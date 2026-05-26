from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from agent_pipeline.storage_bridge import append_jsonl, ensure_governance_dirs, load_manifest, save_manifest

METRIC_THRESHOLDS = {
    "faithfulness": 0.70,
    "answer_relevancy": 0.65,
    "context_precision": 0.60,
    "context_recall": 0.55,
}


def load_eval_results(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_feedback_items(results: dict[str, Any], results_path: str) -> list[dict[str, Any]]:
    backend = results.get("backend", "unknown")
    rows = results.get("raw_result", []) or []
    items: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        item = classify_eval_row(row=row, row_index=idx, backend=backend, results_path=results_path)
        if item is not None:
            items.append(item)
    return items


def classify_eval_row(
    row: dict[str, Any],
    row_index: int,
    backend: str,
    results_path: str,
) -> dict[str, Any] | None:
    metrics = {
        name: float(row.get(name, 0.0) or 0.0)
        for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    }
    query_error = row.get("query_error")
    question = row.get("question", "") or ""
    answer = row.get("answer", "") or ""
    contexts = row.get("contexts", []) or []
    reference_contexts = row.get("reference_contexts", []) or []
    metadata = row.get("metadata", {}) or {}
    failed_metrics = [
        name for name, threshold in METRIC_THRESHOLDS.items() if metrics.get(name, 0.0) < threshold
    ]
    if not query_error and not failed_metrics:
        return None

    reason_tags: list[str] = []
    suggested_actions: list[str] = []
    target_queue = "repair"
    target_decision = "repair"

    if query_error or answer == "查询失败":
        reason_tags.append("query_failure")
        suggested_actions.append("检查检索链路、embedding 维度和索引可用性")
        target_queue = "review"
        target_decision = "review"

    if not contexts:
        reason_tags.append("empty_context")
        suggested_actions.append("补采候选语料或重建索引，避免空上下文")
        target_queue = "review"
        target_decision = "review"

    if metrics["context_recall"] < 0.25:
        reason_tags.append("coverage_gap")
        suggested_actions.append("补充覆盖 ground truth 的车型/配置语料")

    if metrics["context_precision"] < 0.35:
        reason_tags.append("retrieval_noise")
        suggested_actions.append("优化 chunk 结构和检索筛选，减少无关上下文")

    if metrics["faithfulness"] < 0.30:
        reason_tags.append("grounding_weak")
        suggested_actions.append("增强答案与检索上下文的绑定，降低幻觉")
        if target_queue != "review":
            target_queue = "review"
            target_decision = "review"

    if metrics["answer_relevancy"] < 0.25:
        reason_tags.append("answer_mismatch")
        suggested_actions.append("针对问题类型优化检索意图和回答生成模板")

    if not reason_tags:
        reason_tags.append("below_threshold")
        suggested_actions.append("按低分指标回查清洗、分块与检索策略")

    severity = derive_severity(metrics, query_error=bool(query_error), contexts=contexts)
    issue_groups = build_issue_groups(reason_tags)

    return {
        "feedback_id": f"eval-feedback-{row_index:04d}",
        "created_at": datetime.now().isoformat(),
        "source": "ragas_eval",
        "backend": backend,
        "results_path": os.path.abspath(results_path),
        "row_index": row_index,
        "question": question,
        "ground_truth": row.get("ground_truth", "") or "",
        "answer": answer,
        "query_error": query_error,
        "metrics": metrics,
        "failed_metrics": failed_metrics,
        "severity": severity,
        "target_queue": target_queue,
        "target_decision": target_decision,
        "reason_tags": reason_tags,
        "issue_groups": issue_groups,
        "suggested_actions": sorted(set(suggested_actions)),
        "metadata": metadata,
        "reference_contexts": reference_contexts,
        "contexts_preview": [str(ctx)[:240] for ctx in contexts[:3]],
        "answer_preview": answer[:240],
    }


def derive_severity(metrics: dict[str, float], query_error: bool, contexts: list[str]) -> str:
    if query_error or not contexts:
        return "critical"
    lowest = min(metrics.values()) if metrics else 0.0
    if lowest < 0.15:
        return "high"
    if lowest < 0.4:
        return "medium"
    return "low"


def build_issue_groups(reason_tags: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "retrieval": [],
        "quality": [],
        "coverage": [],
        "generation": [],
    }
    for tag in reason_tags:
        if tag in {"query_failure", "empty_context", "retrieval_noise"}:
            groups["retrieval"].append(tag)
        elif tag in {"coverage_gap"}:
            groups["coverage"].append(tag)
        elif tag in {"grounding_weak", "answer_mismatch"}:
            groups["generation"].append(tag)
        else:
            groups["quality"].append(tag)
    return {name: values for name, values in groups.items() if values}


def persist_feedback_items(items: list[dict[str, Any]], base_dir: str = "data") -> dict[str, Any]:
    dirs = ensure_governance_dirs(base_dir)
    summary = {
        "total_feedback": 0,
        "by_queue": {},
        "by_severity": {},
        "by_reason_tag": {},
        "updated_at": datetime.now().isoformat(),
    }

    for item in items:
        append_jsonl(os.path.join(dirs["feedback"], "ragas_feedback.jsonl"), item)
        queue_name = item["target_queue"]
        if queue_name == "review":
            append_jsonl(os.path.join(dirs["review"], "review_queue.jsonl"), item)
        else:
            append_jsonl(os.path.join(dirs["repair"], "repair_queue.jsonl"), item)

        summary["total_feedback"] += 1
        summary["by_queue"][queue_name] = summary["by_queue"].get(queue_name, 0) + 1
        severity = item["severity"]
        summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
        for tag in item["reason_tags"]:
            summary["by_reason_tag"][tag] = summary["by_reason_tag"].get(tag, 0) + 1

    summary_path = os.path.join(dirs["feedback"], "ragas_feedback_summary.json")
    previous = load_manifest(summary_path)
    if previous:
        summary["history_count"] = int(previous.get("history_count", 0) or 0) + 1
    else:
        summary["history_count"] = 1
    save_manifest(summary_path, summary)
    return summary
