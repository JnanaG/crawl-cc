from __future__ import annotations

import json
import importlib.util
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


def load_verify_module():
    module_path = REPO_ROOT / "scripts" / "verify_end_to_end.py"
    spec = importlib.util.spec_from_file_location("verify_end_to_end", module_path)
    assert_true(spec is not None and spec.loader is not None, "无法加载 verify_end_to_end 模块")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    artifact_dir = REPO_ROOT / "data" / "test_runs" / "verify_end_to_end"
    data_dir = artifact_dir / "data"
    state_dir = artifact_dir / "state"

    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    try:
        (data_dir / "processed").mkdir(parents=True, exist_ok=True)
        (data_dir / "vector_store" / "faiss").mkdir(parents=True, exist_ok=True)
        (data_dir / "evaluation").mkdir(parents=True, exist_ok=True)
        (data_dir / "feedback" / "agent_pipeline").mkdir(parents=True, exist_ok=True)
        (data_dir / "reports").mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)

        (data_dir / "processed" / "dongchedi_training_data.jsonl").write_text(
            json.dumps({"id": "demo", "text": "hello"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (data_dir / "processed" / "dongchedi_training_data.parquet").write_bytes(b"PAR1")
        (data_dir / "vector_store" / "faiss" / "dongchedi.index").write_bytes(b"FAISS")
        (data_dir / "vector_store" / "faiss" / "dongchedi_records.jsonl").write_text(
            json.dumps({"id": "chunk-1"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_json(data_dir / "vector_store" / "faiss" / "dongchedi_meta.json", {"dim": 512})

        write_json(
            data_dir / "evaluation" / "ragas_results.json",
            {
                "backend": "lightweight_fallback",
                "test_cases_count": 1,
                "metrics": {
                    "faithfulness": 0.1,
                    "answer_relevancy": 0.2,
                    "context_precision": 0.3,
                    "context_recall": 0.4,
                },
            },
        )
        (data_dir / "feedback" / "agent_pipeline" / "ragas_feedback.jsonl").write_text(
            json.dumps({"source": "ragas_eval", "queue": "review"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_json(
            data_dir / "feedback" / "agent_pipeline" / "ragas_feedback_summary.json",
            {"total_feedback": 1},
        )
        write_json(data_dir / "reports" / "task_summary.json", {"processed": 1})
        write_json(data_dir / "reports" / "quality_report.json", {"summary": {"accepted": 1}})

        write_json(
            state_dir / "workflow_stale-ci-gate.json",
            {
                "run_id": "stale-ci-gate",
                "finished_at": "2000-01-01T00:00:00",
                "steps": [
                    {
                        "name": "ci_gate",
                        "status": "failed",
                        "return_code": 1,
                        "finished_at": "2000-01-01T00:00:00",
                        "stderr_tail": (
                            "File \"scripts/ci_eval_gate.py\", line 1\n"
                            "SyntaxError: unterminated string literal\n"
                        ),
                    }
                ],
                "results": {"success": False, "failed_step": "ci_gate"},
            },
        )

        command = [
            PYTHON,
            str(REPO_ROOT / "scripts" / "verify_end_to_end.py"),
            "--data-dir",
            str(data_dir),
            "--state-dir",
            str(state_dir),
        ]
        proc = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )

        assert_true(proc.returncode == 0, f"验收脚本返回码异常: {proc.returncode}\n{proc.stderr}")
        verify_module = load_verify_module()
        workflow_result = verify_module.validate_workflow_state(state_dir)
        assert_true(workflow_result["ci_gate_current_syntax_ok"] is True, "当前 ci_gate 语法检查应通过")
        assert_true(
            workflow_result["ci_gate_state_stale_syntax_error"] is True,
            "旧 workflow 中的语法错误应被识别为 stale 记录",
        )

        print("[PASS] 端到端验收脚本测试通过")
        print(
            json.dumps(
                {"state_dir": str(state_dir), "workflow_result": workflow_result},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        print(f"[FAIL] 端到端验收脚本测试失败: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
