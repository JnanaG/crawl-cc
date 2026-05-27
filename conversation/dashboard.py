from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _read_jsonl_count(path: Path) -> int | None:
    if not path.exists():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _latest_file(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return existing[0]


def _glob_files(pattern: str) -> list[Path]:
    return list(DATA_DIR.glob(pattern))


def get_reports_summary() -> dict[str, Any]:
    task_summary = _read_json(DATA_DIR / "reports" / "task_summary.json") or {}
    quality_report = _read_json(DATA_DIR / "reports" / "quality_report.json") or {}
    summary = quality_report.get("summary", {}) or {}
    return {
        "task_summary": task_summary,
        "quality_summary": summary,
        "quality_generated_at": quality_report.get("generated_at", ""),
    }


def get_assets_summary() -> dict[str, Any]:
    summary = _read_json(DATA_DIR / "assets" / "asset_split_summary.json") or {}
    return {
        "summary": summary,
        "files": {
            "rag": str(DATA_DIR / "assets" / "rag" / "rag_corpus.jsonl"),
            "training": str(DATA_DIR / "assets" / "training" / "training_corpus.jsonl"),
            "evaluation": str(DATA_DIR / "assets" / "evaluation" / "eval_candidates.jsonl"),
        },
    }


def get_evaluation_summary() -> dict[str, Any]:
    results = _read_json(DATA_DIR / "evaluation" / "ragas_results.json") or {}
    raw_result = results.get("raw_result", []) or []
    low_score_examples = []
    for item in raw_result[:5]:
        low_score_examples.append(
            {
                "question": item.get("question", ""),
                "faithfulness": item.get("faithfulness"),
                "answer_relevancy": item.get("answer_relevancy"),
                "context_precision": item.get("context_precision"),
                "context_recall": item.get("context_recall"),
            }
        )
    return {
        "backend": results.get("backend", ""),
        "metrics": results.get("metrics", {}) or {},
        "test_cases_count": results.get("test_cases_count", 0),
        "samples": low_score_examples,
    }


def get_feedback_summary() -> dict[str, Any]:
    summary = _read_json(DATA_DIR / "feedback" / "agent_pipeline" / "ragas_feedback_summary.json") or {}
    details_count = _read_jsonl_count(DATA_DIR / "feedback" / "agent_pipeline" / "ragas_feedback.jsonl")
    review_count = _read_jsonl_count(DATA_DIR / "review_queue.jsonl")
    repair_count = _read_jsonl_count(DATA_DIR / "repair_queue.jsonl")
    return {
        "summary": summary,
        "detail_count": details_count,
        "review_queue_count": review_count,
        "repair_queue_count": repair_count,
    }


def get_workflow_summary() -> dict[str, Any]:
    latest = _latest_file(_glob_files("state/workflow_runs/workflow_*.json"))
    if latest is None:
        return {"latest_state_file": "", "state": None}
    state = _read_json(latest) or {}
    steps = state.get("steps", []) or []
    latest_step = steps[-1] if steps else {}
    return {
        "latest_state_file": str(latest),
        "run_id": state.get("run_id", ""),
        "success": (state.get("results", {}) or {}).get("success"),
        "failed_step": (state.get("results", {}) or {}).get("failed_step", ""),
        "step_count": len(steps),
        "latest_step": {
            "name": latest_step.get("name", ""),
            "status": latest_step.get("status", ""),
            "return_code": latest_step.get("return_code"),
            "duration_sec": latest_step.get("duration_sec"),
        },
        "steps": [
            {
                "name": step.get("name", ""),
                "status": step.get("status", ""),
                "return_code": step.get("return_code"),
                "duration_sec": step.get("duration_sec"),
            }
            for step in steps
        ],
    }


def get_overview_payload(session_count: int) -> dict[str, Any]:
    reports = get_reports_summary()
    assets = get_assets_summary()
    evaluation = get_evaluation_summary()
    feedback = get_feedback_summary()
    workflow = get_workflow_summary()
    task_summary = reports.get("task_summary", {}) or {}
    quality_summary = reports.get("quality_summary", {}) or {}
    asset_summary = assets.get("summary", {}) or {}
    metrics = evaluation.get("metrics", {}) or {}
    feedback_summary = feedback.get("summary", {}) or {}
    return {
        "cards": {
            "session_count": session_count,
            "task_total": task_summary.get("total_tasks", 0),
            "task_success_count": task_summary.get("success_count", 0),
            "training_records_total": quality_summary.get("training_records_total", 0),
            "rag_rows": asset_summary.get("rag_rows", 0),
            "training_rows": asset_summary.get("training_rows", 0),
            "eval_candidate_rows": asset_summary.get("eval_candidate_rows", 0),
            "faithfulness": metrics.get("faithfulness"),
            "answer_relevancy": metrics.get("answer_relevancy"),
            "feedback_total": feedback_summary.get("total_feedback", 0),
            "workflow_success": workflow.get("success"),
        },
        "modules": {
            "reports": reports,
            "assets": assets,
            "evaluation": evaluation,
            "feedback": feedback,
            "workflow": workflow,
        },
    }
