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


def main() -> int:
    artifact_dir = REPO_ROOT / "data" / "test_runs" / "workflow_smoke"
    state_dir = artifact_dir / "state"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    command = [
        PYTHON,
        str(REPO_ROOT / "scripts" / "run_pipeline_workflow.py"),
        "--run-id",
        "workflow-smoke",
        "--state-dir",
        str(state_dir),
        "--results",
        str(REPO_ROOT / "data" / "evaluation" / "ragas_results.json"),
        "--dry-run",
        "--run-ci-gate",
        "--embedding-provider",
        "fastembed",
        "--llm-provider",
        "ollama",
        "--llm-model",
        "qwen2.5:3b",
    ]

    try:
        proc = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        assert_true(proc.returncode == 0, f"工作流 dry-run 返回码异常: {proc.returncode}")
        state_path = state_dir / "workflow_workflow-smoke.json"
        assert_true(state_path.exists(), "缺少工作流状态文件")

        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        assert_true(state["results"]["success"] is True, "工作流结果应为 success")
        assert_true(len(state["steps"]) == 5, "默认 dry-run 应包含 5 个步骤")
        step_names = [step["name"] for step in state["steps"]]
        assert_true(step_names == ["main", "build", "eval", "feedback", "ci_gate"], "步骤顺序不正确")
        for step in state["steps"]:
            assert_true(step["status"] == "dry_run", f"步骤 {step['name']} 状态应为 dry_run")
            assert_true(len(step["command"]) >= 2, f"步骤 {step['name']} 命令不完整")

        eval_step = next(step for step in state["steps"] if step["name"] == "eval")
        assert_true("--golden-set" in eval_step["command"], "评测步骤缺少 golden-set")
        feedback_step = next(step for step in state["steps"] if step["name"] == "feedback")
        assert_true(str(REPO_ROOT / "scripts" / "sync_ragas_feedback.py") in feedback_step["command"], "回流步骤命令不正确")

        print("[PASS] 调度编排脚本测试通过")
        print(json.dumps({"steps": step_names, "state_path": str(state_path)}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"[FAIL] 调度编排脚本测试失败: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
