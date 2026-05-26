from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = r"D:\workplace\Crawl\.venv\Scripts\python.exe"


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_workflow(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )


def main() -> int:
    artifact_dir = REPO_ROOT / "data" / "test_runs" / "workflow_resilience"
    state_dir = artifact_dir / "state"
    alerts_dir = artifact_dir / "alerts"
    reports_dir = artifact_dir / "reports"
    results_path = artifact_dir / "evaluation" / "ragas_results.json"

    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    try:
        write_json(
            results_path,
            {
                "metrics": {
                    "faithfulness": 0.2,
                    "answer_relevancy": 0.2,
                    "context_precision": 0.2,
                    "context_recall": 0.2,
                }
            },
        )
        write_json(
            reports_dir / "task_summary.json",
            {
                "total_tasks": 100,
                "success_count": 95,
                "failed_count": 5,
                "skipped_count": 0,
                "retry_total": 1,
            },
        )
        write_json(
            reports_dir / "quality_report.json",
            {
                "summary": {
                    "series_total": 50,
                    "series_valid": 40,
                    "training_records_total": 800,
                    "covered_series_count": 35,
                }
            },
        )

        base_command = [
            PYTHON,
            str(REPO_ROOT / "scripts" / "run_pipeline_workflow.py"),
            "--run-id",
            "workflow-resilience-a",
            "--state-dir",
            str(state_dir),
            "--alerts-dir",
            str(alerts_dir),
            "--results",
            str(results_path),
            "--task-summary",
            str(reports_dir / "task_summary.json"),
            "--quality-report",
            str(reports_dir / "quality_report.json"),
            "--skip-main",
            "--skip-build",
            "--skip-eval",
            "--skip-feedback",
            "--run-ci-gate",
            "--step-retries",
            "1",
            "--retry-delay-sec",
            "0",
        ]

        first_run = run_workflow(base_command)
        assert_true(first_run.returncode == 1, f"首次失败运行返回码异常: {first_run.returncode}")
        state_path = state_dir / "workflow_workflow-resilience-a.json"
        state = read_json(state_path)
        assert_true(state["results"]["success"] is False, "首次运行应失败")
        assert_true(state["results"]["failed_step"] == "ci_gate", "首次失败步骤应为 ci_gate")
        assert_true(len(state["steps"]) == 1, "首次运行应只有一个步骤记录")
        assert_true(state["steps"][0]["retry_count"] == 1, "ci_gate 应重试 1 次")
        assert_true(len(state["steps"][0]["attempts"]) == 2, "ci_gate 应存在 2 次尝试记录")

        alert_rows = read_jsonl(alerts_dir / "workflow_alerts.jsonl")
        assert_true(any(row["alert_type"] == "workflow_failure" for row in alert_rows), "缺少失败告警")

        write_json(
            results_path,
            {
                "metrics": {
                    "faithfulness": 0.9,
                    "answer_relevancy": 0.9,
                    "context_precision": 0.9,
                    "context_recall": 0.9,
                }
            },
        )

        resume_run = run_workflow(base_command + ["--resume"])
        assert_true(resume_run.returncode == 0, f"resume 运行返回码异常: {resume_run.returncode}\n{resume_run.stderr}")
        resumed_state = read_json(state_path)
        assert_true(resumed_state["results"]["success"] is True, "resume 后应成功")
        assert_true(resumed_state["results"]["failed_step"] is None, "resume 后 failed_step 应清空")
        assert_true(len(resumed_state["steps"]) == 1, "resume 后步骤数应保持 1")
        assert_true(resumed_state["steps"][0]["status"] == "success", "resume 后 ci_gate 应成功")
        assert_true(len(resumed_state.get("resume_history", [])) >= 1, "应记录 resume 历史")

        write_json(
            reports_dir / "task_summary.json",
            {
                "total_tasks": 100,
                "success_count": 60,
                "failed_count": 40,
                "skipped_count": 0,
                "retry_total": 3,
            },
        )
        write_json(
            reports_dir / "quality_report.json",
            {
                "summary": {
                    "series_total": 50,
                    "series_valid": 20,
                    "training_records_total": 500,
                    "covered_series_count": 20,
                }
            },
        )
        write_json(
            results_path,
            {
                "metrics": {
                    "faithfulness": 0.7,
                    "answer_relevancy": 0.75,
                    "context_precision": 0.74,
                    "context_recall": 0.72,
                }
            },
        )

        second_run = run_workflow(
            [
                PYTHON,
                str(REPO_ROOT / "scripts" / "run_pipeline_workflow.py"),
                "--run-id",
                "workflow-resilience-b",
                "--state-dir",
                str(state_dir),
                "--alerts-dir",
                str(alerts_dir),
                "--results",
                str(results_path),
                "--task-summary",
                str(reports_dir / "task_summary.json"),
                "--quality-report",
                str(reports_dir / "quality_report.json"),
                "--skip-main",
                "--skip-build",
                "--skip-eval",
                "--skip-feedback",
                "--run-ci-gate",
            ]
        )
        assert_true(second_run.returncode == 0, f"第二次成功运行返回码异常: {second_run.returncode}\n{second_run.stderr}")
        alert_rows = read_jsonl(alerts_dir / "workflow_alerts.jsonl")
        assert_true(any(row["alert_type"] == "quality_drift" for row in alert_rows), "缺少质量波动告警")

        history_rows = read_jsonl(state_dir / "workflow_history.jsonl")
        assert_true(len(history_rows) >= 3, "workflow 历史记录数量不正确")

        print("[PASS] 工作流韧性与告警测试通过")
        print(
            json.dumps(
                {
                    "state_path": str(state_path),
                    "alerts_dir": str(alerts_dir),
                    "history_count": len(history_rows),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        print(f"[FAIL] 工作流韧性与告警测试失败: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
