from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def get_latest_workflow_state(state_dir: Path) -> Path | None:
    candidates = sorted(state_dir.glob("workflow_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def validate_python_syntax(script_path: Path) -> None:
    assert_true(script_path.exists(), f"缺少脚本: {script_path}")
    source = script_path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(script_path))
    except SyntaxError as exc:
        raise AssertionError(f"{script_path.name} 当前仍存在语法错误: {exc}") from exc


def is_stale_syntax_error(
    script_path: Path,
    workflow: dict[str, Any],
    ci_gate_step: dict[str, Any],
) -> bool:
    workflow_finished_at = (
        parse_iso_datetime(ci_gate_step.get("finished_at"))
        or parse_iso_datetime(workflow.get("finished_at"))
    )
    if workflow_finished_at is None:
        return False
    script_mtime = datetime.fromtimestamp(script_path.stat().st_mtime, tz=workflow_finished_at.tzinfo)
    return script_mtime > workflow_finished_at


def validate_processed(data_dir: Path) -> dict[str, Any]:
    processed_jsonl = data_dir / "processed" / "dongchedi_training_data.jsonl"
    processed_parquet = data_dir / "processed" / "dongchedi_training_data.parquet"
    assert_true(processed_jsonl.exists(), "缺少 processed JSONL")
    assert_true(processed_parquet.exists(), "缺少 processed Parquet")
    processed_count = count_jsonl(processed_jsonl)
    assert_true(processed_count > 0, "processed JSONL 为空")
    return {
        "processed_jsonl": str(processed_jsonl),
        "processed_parquet": str(processed_parquet),
        "processed_count": processed_count,
    }


def validate_vector_store(data_dir: Path) -> dict[str, Any]:
    faiss_dir = data_dir / "vector_store" / "faiss"
    index_path = faiss_dir / "dongchedi.index"
    records_path = faiss_dir / "dongchedi_records.jsonl"
    meta_path = faiss_dir / "dongchedi_meta.json"
    assert_true(index_path.exists(), "缺少 FAISS index")
    assert_true(records_path.exists(), "缺少 FAISS records")
    assert_true(meta_path.exists(), "缺少 FAISS meta")
    record_count = count_jsonl(records_path)
    assert_true(record_count > 0, "FAISS records 为空")
    meta = read_json(meta_path)
    return {
        "index": str(index_path),
        "records": str(records_path),
        "meta": str(meta_path),
        "record_count": record_count,
        "meta_dim": meta.get("dim"),
    }


def validate_eval_and_feedback(data_dir: Path) -> dict[str, Any]:
    eval_path = data_dir / "evaluation" / "ragas_results.json"
    feedback_summary_path = data_dir / "feedback" / "agent_pipeline" / "ragas_feedback_summary.json"
    feedback_jsonl_path = data_dir / "feedback" / "agent_pipeline" / "ragas_feedback.jsonl"
    assert_true(eval_path.exists(), "缺少 ragas_results.json")
    assert_true(feedback_summary_path.exists(), "缺少 ragas_feedback_summary.json")
    assert_true(feedback_jsonl_path.exists(), "缺少 ragas_feedback.jsonl")

    eval_results = read_json(eval_path)
    feedback_summary = read_json(feedback_summary_path)
    feedback_count = count_jsonl(feedback_jsonl_path)

    assert_true(eval_results.get("test_cases_count", 0) > 0, "评测结果缺少 test_cases_count")
    assert_true("metrics" in eval_results, "评测结果缺少 metrics")
    summary_total = int(feedback_summary.get("total_feedback", 0) or 0)
    assert_true(summary_total > 0, "feedback summary total_feedback 非法")
    assert_true(feedback_count >= summary_total, "feedback 明细数量少于最近一次 summary 统计")

    return {
        "eval_path": str(eval_path),
        "backend": eval_results.get("backend"),
        "metrics": eval_results.get("metrics", {}),
        "test_cases_count": eval_results.get("test_cases_count", 0),
        "feedback_count": feedback_count,
        "feedback_last_run_count": summary_total,
        "feedback_summary": feedback_summary,
    }


def validate_reports(data_dir: Path) -> dict[str, Any]:
    task_summary_path = data_dir / "reports" / "task_summary.json"
    quality_report_path = data_dir / "reports" / "quality_report.json"
    assert_true(task_summary_path.exists(), "缺少 task_summary.json")
    assert_true(quality_report_path.exists(), "缺少 quality_report.json")
    task_summary = read_json(task_summary_path)
    quality_report = read_json(quality_report_path)
    return {
        "task_summary": task_summary,
        "quality_summary": quality_report.get("summary", {}),
    }


def validate_workflow_state(state_dir: Path) -> dict[str, Any]:
    latest = get_latest_workflow_state(state_dir)
    assert_true(latest is not None, "缺少 workflow state 文件")
    workflow = read_json(latest)
    steps = workflow.get("steps", [])
    assert_true(len(steps) >= 1, "workflow steps 为空")

    ci_gate_script = REPO_ROOT / "scripts" / "ci_eval_gate.py"
    validate_python_syntax(ci_gate_script)

    ci_gate_step = next((step for step in steps if step.get("name") == "ci_gate"), None)
    stale_syntax_error = False
    if ci_gate_step:
        stderr_tail = ci_gate_step.get("stderr_tail", "") or ""
        if "SyntaxError" in stderr_tail:
            stale_syntax_error = is_stale_syntax_error(ci_gate_script, workflow, ci_gate_step)
            assert_true(
                stale_syntax_error,
                "ci_gate workflow 记录显示语法错误，且当前脚本未体现为修复后的版本",
            )

    return {
        "state_path": str(latest),
        "run_id": workflow.get("run_id"),
        "success": workflow.get("results", {}).get("success"),
        "failed_step": workflow.get("results", {}).get("failed_step"),
        "step_count": len(steps),
        "ci_gate_script": str(ci_gate_script),
        "ci_gate_current_syntax_ok": True,
        "ci_gate_state_stale_syntax_error": stale_syntax_error,
        "ci_gate_status": ci_gate_step.get("status") if ci_gate_step else None,
        "ci_gate_return_code": ci_gate_step.get("return_code") if ci_gate_step else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="端到端验收脚本：检查主链路产物和工作流状态")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--state-dir", default=str(REPO_ROOT / "data" / "state" / "workflow_runs"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    state_dir = Path(args.state_dir).resolve()

    try:
        result = {
            "processed": validate_processed(data_dir),
            "vector_store": validate_vector_store(data_dir),
            "eval_feedback": validate_eval_and_feedback(data_dir),
            "reports": validate_reports(data_dir),
            "workflow": validate_workflow_state(state_dir),
        }
        print("[PASS] 端到端验收通过")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"[FAIL] 端到端验收失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
