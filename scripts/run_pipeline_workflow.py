from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "data" / "evaluation" / "ragas_results.json"
DEFAULT_GOLDEN_SET = REPO_ROOT / "data" / "evaluation" / "golden_set.jsonl"
DEFAULT_WORKFLOW_STATE_DIR = REPO_ROOT / "data" / "state" / "workflow_runs"
DEFAULT_ALERTS_DIR = REPO_ROOT / "data" / "alerts" / "workflow"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_workflow_state(path: Path) -> dict[str, Any] | None:
    return load_json(path)


def emit_alert(
    alerts_dir: Path,
    run_id: str,
    alert_type: str,
    severity: str,
    message: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    alert = {
        "run_id": run_id,
        "alert_type": alert_type,
        "severity": severity,
        "message": message,
        "context": context,
        "created_at": datetime.now().isoformat(),
    }
    append_jsonl(alerts_dir / "workflow_alerts.jsonl", alert)
    with (alerts_dir / "latest_alert.json").open("w", encoding="utf-8") as f:
        json.dump(alert, f, ensure_ascii=False, indent=2)
    return alert


def build_report_snapshot(
    task_summary_path: Path,
    quality_report_path: Path,
    eval_results_path: Path,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "task_summary_path": str(task_summary_path),
        "quality_report_path": str(quality_report_path),
        "eval_results_path": str(eval_results_path),
    }
    task_summary = load_json(task_summary_path) or {}
    quality_report = load_json(quality_report_path) or {}
    eval_results = load_json(eval_results_path) or {}

    total_tasks = int(task_summary.get("total_tasks", 0) or 0)
    success_count = int(task_summary.get("success_count", 0) or 0)
    summary = quality_report.get("summary", {}) or {}

    snapshot["task_success_rate"] = round(success_count / total_tasks, 4) if total_tasks > 0 else None
    snapshot["training_records_total"] = int(summary.get("training_records_total", 0) or 0) or None
    snapshot["covered_series_count"] = int(summary.get("covered_series_count", 0) or 0) or None
    series_total = int(summary.get("series_total", 0) or 0)
    series_valid = int(summary.get("series_valid", 0) or 0)
    snapshot["series_valid_rate"] = round(series_valid / series_total, 4) if series_total > 0 else None

    metrics = eval_results.get("metrics", {}) or {}
    if metrics:
        snapshot["eval_metrics"] = {
            key: round(float(value), 4)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
    else:
        snapshot["eval_metrics"] = {}
    return snapshot


def compare_snapshots(
    baseline: dict[str, Any],
    current: dict[str, Any],
    success_rate_drop_threshold: float,
    series_valid_rate_drop_threshold: float,
    training_record_drop_threshold: float,
    covered_series_drop_threshold: float,
    eval_metric_drop_threshold: float,
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []

    def check_ratio_drop(key: str, threshold: float, label: str) -> None:
        baseline_value = baseline.get(key)
        current_value = current.get(key)
        if baseline_value is None or current_value is None:
            return
        if float(current_value) < float(baseline_value) - threshold:
            alerts.append(
                {
                    "metric": key,
                    "label": label,
                    "baseline": baseline_value,
                    "current": current_value,
                    "threshold": threshold,
                }
            )

    def check_relative_drop(key: str, threshold: float, label: str) -> None:
        baseline_value = baseline.get(key)
        current_value = current.get(key)
        if baseline_value in (None, 0) or current_value is None:
            return
        drop_ratio = (float(baseline_value) - float(current_value)) / max(float(baseline_value), 1e-6)
        if drop_ratio > threshold:
            alerts.append(
                {
                    "metric": key,
                    "label": label,
                    "baseline": baseline_value,
                    "current": current_value,
                    "threshold": threshold,
                    "drop_ratio": round(drop_ratio, 4),
                }
            )

    check_ratio_drop("task_success_rate", success_rate_drop_threshold, "任务成功率下降")
    check_ratio_drop("series_valid_rate", series_valid_rate_drop_threshold, "有效车系占比下降")
    check_relative_drop("training_records_total", training_record_drop_threshold, "训练样本量下降")
    check_relative_drop("covered_series_count", covered_series_drop_threshold, "覆盖车系数下降")

    baseline_metrics = baseline.get("eval_metrics", {}) or {}
    current_metrics = current.get("eval_metrics", {}) or {}
    for metric, baseline_value in baseline_metrics.items():
        current_value = current_metrics.get(metric)
        if current_value is None:
            continue
        if float(current_value) < float(baseline_value) - eval_metric_drop_threshold:
            alerts.append(
                {
                    "metric": metric,
                    "label": f"评测指标 {metric} 下降",
                    "baseline": baseline_value,
                    "current": current_value,
                    "threshold": eval_metric_drop_threshold,
                }
            )

    return alerts


def maybe_emit_quality_drift_alerts(
    alerts_dir: Path,
    baseline_path: Path,
    run_id: str,
    task_summary_path: Path,
    quality_report_path: Path,
    eval_results_path: Path,
    success_rate_drop_threshold: float,
    series_valid_rate_drop_threshold: float,
    training_record_drop_threshold: float,
    covered_series_drop_threshold: float,
    eval_metric_drop_threshold: float,
) -> list[dict[str, Any]]:
    current_snapshot = build_report_snapshot(
        task_summary_path=task_summary_path,
        quality_report_path=quality_report_path,
        eval_results_path=eval_results_path,
    )
    if all(
        current_snapshot.get(key) is None for key in ["task_success_rate", "training_records_total", "covered_series_count", "series_valid_rate"]
    ) and not current_snapshot.get("eval_metrics"):
        return []

    baseline_snapshot = load_json(baseline_path) or {}
    alerts: list[dict[str, Any]] = []
    if baseline_snapshot:
        drift_items = compare_snapshots(
            baseline=baseline_snapshot,
            current=current_snapshot,
            success_rate_drop_threshold=success_rate_drop_threshold,
            series_valid_rate_drop_threshold=series_valid_rate_drop_threshold,
            training_record_drop_threshold=training_record_drop_threshold,
            covered_series_drop_threshold=covered_series_drop_threshold,
            eval_metric_drop_threshold=eval_metric_drop_threshold,
        )
        for item in drift_items:
            alerts.append(
                emit_alert(
                    alerts_dir=alerts_dir,
                    run_id=run_id,
                    alert_type="quality_drift",
                    severity="medium",
                    message=item["label"],
                    context=item,
                )
            )

    ensure_parent(baseline_path)
    with baseline_path.open("w", encoding="utf-8") as f:
        json.dump(current_snapshot, f, ensure_ascii=False, indent=2)
    return alerts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批处理调度编排脚本：串联 main/build/eval/feedback")
    parser.add_argument("--python", default=sys.executable, help="执行各步骤的 Python 解释器路径")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="本次工作流运行 ID")
    parser.add_argument("--state-dir", default=str(DEFAULT_WORKFLOW_STATE_DIR), help="工作流状态输出目录")
    parser.add_argument("--alerts-dir", default=str(DEFAULT_ALERTS_DIR), help="告警输出目录")
    parser.add_argument("--dry-run", action="store_true", help="仅打印步骤，不真正执行")
    parser.add_argument("--resume", action="store_true", help="基于已有同 run-id 状态断点续跑")
    parser.add_argument("--step-retries", type=int, default=0, help="单个步骤失败后的重试次数")
    parser.add_argument("--retry-delay-sec", type=float, default=1.0, help="步骤重试前等待秒数")

    parser.add_argument("--skip-main", action="store_true", help="跳过 ingest/govern/chunk 主流程")
    parser.add_argument("--skip-build", action="store_true", help="跳过向量构建")
    parser.add_argument("--skip-eval", action="store_true", help="跳过 ragas 评测")
    parser.add_argument("--skip-feedback", action="store_true", help="跳过评测结果回流")
    parser.add_argument("--run-ci-gate", action="store_true", help="在评测后执行 CI gate 检查")

    parser.add_argument("--target-training-records", type=int, default=1000)
    parser.add_argument("--target-series-pool", type=int, default=300)
    parser.add_argument("--max-chunk-tokens", type=int, default=100)
    parser.add_argument("--max-series-expand-requests", type=int, default=40)

    parser.add_argument(
        "--embedding-provider",
        default="fastembed",
        choices=["openai_compatible", "ollama", "fastembed", "sentence_transformers"],
    )
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-api-base", default=None)
    parser.add_argument("--embedding-api-key", default=None)

    parser.add_argument(
        "--llm-provider",
        default="ollama",
        choices=["openai_compatible", "ollama"],
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-base", default=None)
    parser.add_argument("--llm-api-key", default=None)

    parser.add_argument("--golden-set", default=str(DEFAULT_GOLDEN_SET))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--eval-backend", default="auto", choices=["auto", "ragas", "lightweight"])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--task-summary", default=str(REPO_ROOT / "data" / "reports" / "task_summary.json"))
    parser.add_argument("--quality-report", default=str(REPO_ROOT / "data" / "reports" / "quality_report.json"))
    parser.add_argument("--success-rate-drop-threshold", type=float, default=0.1)
    parser.add_argument("--series-valid-rate-drop-threshold", type=float, default=0.1)
    parser.add_argument("--training-record-drop-threshold", type=float, default=0.15)
    parser.add_argument("--covered-series-drop-threshold", type=float, default=0.15)
    parser.add_argument("--eval-metric-drop-threshold", type=float, default=0.1)
    return parser


def add_optional_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    command.extend([flag, str(value)])


def build_steps(args: argparse.Namespace) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []

    if not args.skip_main:
        env_overrides = {
            "TARGET_TRAINING_RECORDS": str(args.target_training_records),
            "TARGET_SERIES_POOL": str(args.target_series_pool),
            "MAX_CHUNK_TOKENS": str(args.max_chunk_tokens),
            "MAX_SERIES_EXPAND_REQUESTS": str(args.max_series_expand_requests),
        }
        steps.append(
            {
                "name": "main",
                "description": "执行 scrape/clean/govern/chunk 主流程",
                "command": [args.python, str(REPO_ROOT / "main.py")],
                "env_overrides": env_overrides,
            }
        )

    if not args.skip_build:
        command = [
            args.python,
            str(REPO_ROOT / "rag_llm_demo.py"),
            "build",
            "--embedding-provider",
            args.embedding_provider,
        ]
        add_optional_arg(command, "--embedding-model", args.embedding_model)
        add_optional_arg(command, "--embedding-api-base", args.embedding_api_base)
        add_optional_arg(command, "--embedding-api-key", args.embedding_api_key)
        steps.append(
            {
                "name": "build",
                "description": "构建向量索引",
                "command": command,
                "env_overrides": {},
            }
        )

    if not args.skip_eval:
        command = [
            args.python,
            str(REPO_ROOT / "rag_ragas_eval.py"),
            "--golden-set",
            args.golden_set,
            "--output",
            args.results,
            "--eval-backend",
            args.eval_backend,
            "--embedding-provider",
            args.embedding_provider,
            "--llm-provider",
            args.llm_provider,
            "--sleep-sec",
            str(args.sleep_sec),
        ]
        add_optional_arg(command, "--embedding-model", args.embedding_model)
        add_optional_arg(command, "--embedding-api-base", args.embedding_api_base)
        add_optional_arg(command, "--embedding-api-key", args.embedding_api_key)
        add_optional_arg(command, "--llm-model", args.llm_model)
        add_optional_arg(command, "--llm-api-base", args.llm_api_base)
        add_optional_arg(command, "--llm-api-key", args.llm_api_key)
        if args.max_cases is not None:
            command.extend(["--max-cases", str(args.max_cases)])
        steps.append(
            {
                "name": "eval",
                "description": "执行 RAG 评测",
                "command": command,
                "env_overrides": {},
            }
        )

    if not args.skip_feedback:
        steps.append(
            {
                "name": "feedback",
                "description": "将低分样本回流到治理队列",
                "command": [
                    args.python,
                    str(REPO_ROOT / "scripts" / "sync_ragas_feedback.py"),
                    "--results",
                    args.results,
                ],
                "env_overrides": {},
            }
        )

    if args.run_ci_gate:
        steps.append(
            {
                "name": "ci_gate",
                "description": "执行评测阈值守门",
                "command": [
                    args.python,
                    str(REPO_ROOT / "scripts" / "ci_eval_gate.py"),
                    "--results",
                    args.results,
                ],
                "env_overrides": {},
            }
        )
    return steps


def save_workflow_state(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def init_workflow_state(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "created_at": datetime.now().isoformat(),
        "python": args.python,
        "dry_run": args.dry_run,
        "step_retries": args.step_retries,
        "resume_enabled": args.resume,
        "alerts_dir": str(Path(args.alerts_dir).resolve()),
        "steps": [],
        "resume_history": [],
        "alerts": [],
        "results": {
            "success": False,
            "failed_step": None,
        },
    }


def resolve_resume_steps(
    planned_steps: list[dict[str, Any]],
    workflow_state: dict[str, Any],
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not resume:
        return planned_steps, []

    existing_steps = {step.get("name"): step for step in workflow_state.get("steps", [])}
    failed_step = workflow_state.get("results", {}).get("failed_step")
    if not failed_step and workflow_state.get("results", {}).get("success") is True:
        return [], workflow_state.get("steps", [])

    preserved_steps: list[dict[str, Any]] = []
    remaining_steps: list[dict[str, Any]] = []
    rerun_started = False
    for step in planned_steps:
        previous = existing_steps.get(step["name"])
        if not rerun_started and previous and previous.get("status") in {"success", "dry_run"} and step["name"] != failed_step:
            preserved_steps.append(previous)
            continue
        rerun_started = True
        remaining_steps.append(step)
    return remaining_steps, preserved_steps


def run_step_once(step: dict[str, Any], cwd: Path, dry_run: bool) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(step.get("env_overrides", {}) or {})
    started_at = datetime.now().isoformat()
    started_perf = time.perf_counter()
    if dry_run:
        return {
            "name": step["name"],
            "description": step["description"],
            "command": step["command"],
            "env_overrides": step.get("env_overrides", {}),
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "duration_sec": 0.0,
            "status": "dry_run",
            "return_code": 0,
        }

    proc = subprocess.run(
        step["command"],
        cwd=str(cwd),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    duration_sec = round(time.perf_counter() - started_perf, 3)
    return {
        "name": step["name"],
        "description": step["description"],
        "command": step["command"],
        "env_overrides": step.get("env_overrides", {}),
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
        "duration_sec": duration_sec,
        "status": "success" if proc.returncode == 0 else "failed",
        "return_code": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
    }


def run_step(
    step: dict[str, Any],
    cwd: Path,
    dry_run: bool,
    step_retries: int,
    retry_delay_sec: float,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    max_attempts = max(1, int(step_retries) + 1)
    for attempt in range(1, max_attempts + 1):
        step_result = run_step_once(step, cwd=cwd, dry_run=dry_run)
        step_result["attempt"] = attempt
        attempts.append(step_result)
        if step_result["status"] in {"success", "dry_run"}:
            break
        if attempt < max_attempts and retry_delay_sec > 0:
            time.sleep(retry_delay_sec)

    final_result = dict(attempts[-1])
    final_result["attempts"] = attempts
    final_result["retry_count"] = len(attempts) - 1
    final_result["status"] = attempts[-1]["status"]
    return final_result


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    planned_steps = build_steps(args)
    state_dir = Path(args.state_dir).resolve()
    state_path = state_dir / f"workflow_{args.run_id}.json"
    alerts_dir = Path(args.alerts_dir).resolve()
    alerts_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = state_dir / "workflow_quality_baseline.json"
    history_path = state_dir / "workflow_history.jsonl"

    existing_state = load_workflow_state(state_path)
    if args.resume and existing_state:
        workflow_state = existing_state
        workflow_state["resume_enabled"] = True
        workflow_state.setdefault("resume_history", []).append(
            {
                "resumed_at": datetime.now().isoformat(),
                "previous_success": existing_state.get("results", {}).get("success"),
                "previous_failed_step": existing_state.get("results", {}).get("failed_step"),
            }
        )
    else:
        workflow_state = init_workflow_state(args)

    steps, preserved_steps = resolve_resume_steps(
        planned_steps=planned_steps,
        workflow_state=workflow_state,
        resume=args.resume,
    )
    workflow_state["steps"] = preserved_steps
    workflow_state["results"]["success"] = False
    workflow_state["results"]["failed_step"] = None
    save_workflow_state(state_path, workflow_state)

    print(f"[WORKFLOW] run_id={args.run_id}")
    print(f"[WORKFLOW] state={state_path}")
    if args.resume and preserved_steps:
        print(f"[WORKFLOW] resume=on, 复用已完成步骤 {len(preserved_steps)} 个")
    if not steps:
        print("[WORKFLOW] 没有可执行步骤")
        workflow_state["results"]["success"] = True
        workflow_state["finished_at"] = datetime.now().isoformat()
        save_workflow_state(state_path, workflow_state)
        append_jsonl(
            history_path,
            {
                "run_id": args.run_id,
                "finished_at": workflow_state["finished_at"],
                "success": True,
                "failed_step": None,
                "dry_run": args.dry_run,
                "resumed": args.resume,
            },
        )
        return 0

    for idx, step in enumerate(steps, start=1):
        print(f"[STEP {idx}/{len(steps)}] {step['name']} - {step['description']}")
        print("  command:", " ".join(step["command"]))
        if step.get("env_overrides"):
            print("  env:", json.dumps(step["env_overrides"], ensure_ascii=False))

        step_result = run_step(
            step,
            cwd=REPO_ROOT,
            dry_run=args.dry_run,
            step_retries=args.step_retries,
            retry_delay_sec=args.retry_delay_sec,
        )
        workflow_state["steps"].append(step_result)
        save_workflow_state(state_path, workflow_state)

        if step_result["status"] not in {"success", "dry_run"}:
            workflow_state["results"]["failed_step"] = step["name"]
            workflow_state["results"]["success"] = False
            workflow_state["finished_at"] = datetime.now().isoformat()
            failure_alert = emit_alert(
                alerts_dir=alerts_dir,
                run_id=args.run_id,
                alert_type="workflow_failure",
                severity="high",
                message=f"工作流步骤失败: {step['name']}",
                context={
                    "step": step["name"],
                    "return_code": step_result["return_code"],
                    "retry_count": step_result.get("retry_count", 0),
                    "stderr_tail": step_result.get("stderr_tail", ""),
                },
            )
            workflow_state.setdefault("alerts", []).append(failure_alert)
            save_workflow_state(state_path, workflow_state)
            append_jsonl(
                history_path,
                {
                    "run_id": args.run_id,
                    "finished_at": workflow_state["finished_at"],
                    "success": False,
                    "failed_step": step["name"],
                    "dry_run": args.dry_run,
                    "resumed": args.resume,
                },
            )
            print(f"[FAIL] step={step['name']} return_code={step_result['return_code']}", file=sys.stderr)
            return int(step_result["return_code"] or 1)

    workflow_state["results"]["success"] = True
    workflow_state["results"]["failed_step"] = None
    workflow_state["finished_at"] = datetime.now().isoformat()
    if not args.dry_run:
        drift_alerts = maybe_emit_quality_drift_alerts(
            alerts_dir=alerts_dir,
            baseline_path=baseline_path,
            run_id=args.run_id,
            task_summary_path=Path(args.task_summary).resolve(),
            quality_report_path=Path(args.quality_report).resolve(),
            eval_results_path=Path(args.results).resolve(),
            success_rate_drop_threshold=args.success_rate_drop_threshold,
            series_valid_rate_drop_threshold=args.series_valid_rate_drop_threshold,
            training_record_drop_threshold=args.training_record_drop_threshold,
            covered_series_drop_threshold=args.covered_series_drop_threshold,
            eval_metric_drop_threshold=args.eval_metric_drop_threshold,
        )
        workflow_state.setdefault("alerts", []).extend(drift_alerts)
    save_workflow_state(state_path, workflow_state)
    append_jsonl(
        history_path,
        {
            "run_id": args.run_id,
            "finished_at": workflow_state["finished_at"],
            "success": True,
            "failed_step": None,
            "dry_run": args.dry_run,
            "resumed": args.resume,
            "alert_count": len(workflow_state.get("alerts", [])),
        },
    )
    print("[PASS] 工作流执行完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
