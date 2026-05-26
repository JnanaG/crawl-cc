from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_pipeline import build_feedback_items, load_eval_results, persist_feedback_items


def main() -> int:
    parser = argparse.ArgumentParser(description="将 RAG 评测结果回流到治理队列")
    parser.add_argument(
        "--results",
        default="data/evaluation/ragas_results.json",
        help="评测结果 JSON 路径",
    )
    parser.add_argument(
        "--base-dir",
        default="data",
        help="治理输出根目录，默认写入 data/feedback|review|repair/agent_pipeline",
    )
    args = parser.parse_args()

    results_path = Path(args.results).resolve()
    if not results_path.exists():
        print(f"[FAIL] 评测结果文件不存在: {results_path}", file=sys.stderr)
        return 1

    results = load_eval_results(str(results_path))
    items = build_feedback_items(results=results, results_path=str(results_path))
    summary = persist_feedback_items(items, base_dir=args.base_dir)

    print("[PASS] 评测结果已回流到治理队列")
    print(f"results={results_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
