from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multimodal.caption_builder import (  # noqa: E402
    build_image_caption_corpus,
    load_jsonl,
    write_image_caption_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将图片资产转成可进入 RAG 的 caption 文本语料")
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "data" / "multimodal" / "image_assets.jsonl"),
        help="图片资产 JSONL 输入路径",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "multimodal"),
        help="caption 语料输出目录",
    )
    parser.add_argument(
        "--caption-provider",
        default="heuristic",
        choices=["heuristic", "openai_compatible", "ollama"],
    )
    parser.add_argument("--caption-model", default=None)
    parser.add_argument("--caption-api-base", default=None)
    parser.add_argument("--caption-api-key", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--all-assets", action="store_true", help="默认仅处理 is_usable=true 的图片")
    parser.add_argument("--no-fail-open", action="store_true", help="模型失败时不回退 heuristic")
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    image_assets = load_jsonl(input_path)
    result = build_image_caption_corpus(
        image_assets=image_assets,
        provider=args.caption_provider,
        model=args.caption_model,
        api_base=args.caption_api_base,
        api_key=args.caption_api_key,
        only_usable=not args.all_assets,
        limit=args.limit,
        fail_open=not args.no_fail_open,
        temperature=args.temperature,
    )
    paths = write_image_caption_outputs(result, output_dir)

    print("[PASS] 图片 caption 语料构建完成")
    print(
        json.dumps(
            {
                **result["summary"],
                **paths,
                "input_path": str(input_path),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
