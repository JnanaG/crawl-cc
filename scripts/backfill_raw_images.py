from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scraper.dcd_scraper import DongchediScraper  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于已有 raw json 回填原始图片和图片 manifest")
    parser.add_argument(
        "--raw-dir",
        default=str(REPO_ROOT / "data" / "raw" / "dongchedi"),
        help="raw json 目录",
    )
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 个车系")
    parser.add_argument("--skip-existing", action="store_true", help="若已存在 series_<id>_images.json 则跳过")
    parser.add_argument("--include-contextual-images", action="store_true", help="是否回填新闻/推荐等上下文图片")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    scraper = DongchediScraper()

    raw_files = sorted(
        path
        for path in raw_dir.glob("series_*.json")
        if not path.name.endswith("_images.json")
    )
    if args.limit is not None:
        raw_files = raw_files[: max(args.limit, 0)]

    summary = {
        "raw_files_scanned": len(raw_files),
        "processed_series_count": 0,
        "skipped_existing_count": 0,
        "image_success_count": 0,
        "image_failed_count": 0,
    }

    for raw_path in raw_files:
        series_id = raw_path.stem.replace("series_", "", 1)
        manifest_path = raw_path.with_name(f"{raw_path.stem}_images.json")
        if args.skip_existing and manifest_path.exists():
            summary["skipped_existing_count"] += 1
            continue

        with raw_path.open("r", encoding="utf-8") as f:
            raw_json = json.load(f)
        if raw_json.get("page") == "/_error":
            continue

        result = scraper.save_series_images(
            series_id=series_id,
            raw_json=raw_json,
            include_contextual_images=args.include_contextual_images,
        )
        summary["processed_series_count"] += 1
        summary["image_success_count"] += int(result.get("image_success_count", 0) or 0)
        summary["image_failed_count"] += int(result.get("image_failed_count", 0) or 0)

    print("[PASS] 原始图片回填完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
