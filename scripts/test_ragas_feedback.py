from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_pipeline import build_feedback_items, load_eval_results, persist_feedback_items


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    artifact_dir = REPO_ROOT / "data" / "test_runs" / "ragas_feedback_smoke"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    results_path = REPO_ROOT / "data" / "evaluation" / "ragas_results.json"
    if not results_path.exists():
        print(f"[FAIL] 缺少评测结果文件: {results_path}", file=sys.stderr)
        return 1

    try:
        results = load_eval_results(str(results_path))
        items = build_feedback_items(results, str(results_path))
        assert_true(len(items) > 0, "应该至少生成一条回流任务")
        summary = persist_feedback_items(items, base_dir=str(artifact_dir))

        feedback_path = artifact_dir / "feedback" / "agent_pipeline" / "ragas_feedback.jsonl"
        review_path = artifact_dir / "review" / "agent_pipeline" / "review_queue.jsonl"
        repair_path = artifact_dir / "repair" / "agent_pipeline" / "repair_queue.jsonl"
        summary_path = artifact_dir / "feedback" / "agent_pipeline" / "ragas_feedback_summary.json"

        assert_true(feedback_path.exists(), "缺少 ragas_feedback.jsonl")
        assert_true(summary_path.exists(), "缺少 ragas_feedback_summary.json")
        assert_true(review_path.exists() or repair_path.exists(), "review/repair 队列至少应有一个存在")

        with feedback_path.open("r", encoding="utf-8") as f:
            first_item = json.loads(next(line for line in f if line.strip()))
        assert_true(first_item["source"] == "ragas_eval", "feedback source 不正确")
        assert_true(first_item["target_queue"] in {"review", "repair"}, "target_queue 不正确")
        assert_true(len(first_item["reason_tags"]) >= 1, "reason_tags 不能为空")
        assert_true(summary["total_feedback"] == len(items), "summary total_feedback 不匹配")

        print("[PASS] ragas 回流测试通过")
        print(json.dumps({"items": len(items), "summary": summary, "sample": first_item}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"[FAIL] ragas 回流测试失败: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
