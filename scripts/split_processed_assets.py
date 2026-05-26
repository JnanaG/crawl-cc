from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_pipeline.dataset_splitter import load_jsonl, split_processed_assets, write_json, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将 processed 数据分流为 RAG/训练/评测候选三类资产")
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "data" / "processed" / "dongchedi_training_data.jsonl"),
        help="processed JSONL 输入路径",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "assets"),
        help="多目标资产输出目录",
    )
    parser.add_argument("--min-rag-readiness", type=float, default=0.45)
    parser.add_argument("--min-training-readiness", type=float, default=0.6)
    parser.add_argument("--min-quality-score", type=float, default=0.65)
    parser.add_argument("--min-eval-readiness", type=float, default=0.6)
    parser.add_argument("--max-training-chunks-per-series", type=int, default=8)
    parser.add_argument("--min-eval-chunks-per-series", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    items = load_jsonl(str(input_path))
    result = split_processed_assets(
        items=items,
        min_rag_readiness=args.min_rag_readiness,
        min_training_readiness=args.min_training_readiness,
        min_quality_score=args.min_quality_score,
        min_eval_readiness=args.min_eval_readiness,
        max_training_chunks_per_series=args.max_training_chunks_per_series,
        min_eval_chunks_per_series=args.min_eval_chunks_per_series,
    )

    rag_path = output_dir / "rag" / "rag_corpus.jsonl"
    training_path = output_dir / "training" / "training_corpus.jsonl"
    eval_path = output_dir / "evaluation" / "eval_candidates.jsonl"
    summary_path = output_dir / "asset_split_summary.json"

    write_jsonl(str(rag_path), result["rag_corpus"])
    write_jsonl(str(training_path), result["training_corpus"])
    write_jsonl(str(eval_path), result["eval_candidates"])
    write_json(str(summary_path), result["summary"])

    print("[PASS] processed 数据分流完成")
    print(
        json.dumps(
            {
                **result["summary"],
                "input_path": str(input_path),
                "rag_path": str(rag_path),
                "training_path": str(training_path),
                "eval_path": str(eval_path),
                "summary_path": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
