from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multimodal.image_pipeline import (  # noqa: E402
    build_image_dataset,
    download_image_assets,
    write_image_dataset_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从懂车帝 raw/cleaned 数据构建图片数据集 manifest")
    parser.add_argument(
        "--raw-dir",
        default=str(REPO_ROOT / "data" / "raw" / "dongchedi"),
        help="原始 JSON 目录",
    )
    parser.add_argument(
        "--cleaned-dir",
        default=str(REPO_ROOT / "data" / "cleaned" / "dongchedi" / "json"),
        help="cleaned JSON 目录，可选用于补充车系元数据",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "multimodal"),
        help="图片数据链路输出目录",
    )
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个 raw JSON")
    parser.add_argument(
        "--include-contextual-images",
        action="store_true",
        help="是否纳入新闻、同品牌推荐等上下文图片",
    )
    parser.add_argument(
        "--download-assets",
        action="store_true",
        help="是否下载高质量可用图片到本地",
    )
    parser.add_argument(
        "--download-limit-per-series",
        type=int,
        default=20,
        help="每个车系最多下载图片数",
    )
    parser.add_argument("--timeout-sec", type=int, default=15, help="下载图片超时时间")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    cleaned_dir = Path(args.cleaned_dir).resolve() if args.cleaned_dir else None
    output_dir = Path(args.output_dir).resolve()

    result = build_image_dataset(
        raw_dir=raw_dir,
        cleaned_dir=cleaned_dir if cleaned_dir and cleaned_dir.exists() else None,
        include_contextual_images=args.include_contextual_images,
        limit=args.limit,
    )
    paths = write_image_dataset_outputs(result, output_dir)

    response = {
        **result["summary"],
        **paths,
        "raw_dir": str(raw_dir),
        "cleaned_dir": str(cleaned_dir) if cleaned_dir else "",
        "output_dir": str(output_dir),
    }

    if args.download_assets:
        download_result = download_image_assets(
            result["image_assets"],
            output_dir=output_dir / "downloaded_images",
            limit_per_series=args.download_limit_per_series,
            timeout_sec=args.timeout_sec,
        )
        response["download"] = download_result

    print("[PASS] 图片数据链路构建完成")
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
