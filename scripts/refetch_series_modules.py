from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cleaner.dcd_cleaner import DongchediCleaner  # noqa: E402
from scraper.dcd_scraper import DongchediScraper  # noqa: E402


DEFAULT_CLEAR_TARGETS = [
    REPO_ROOT / "data" / "raw" / "dongchedi",
    REPO_ROOT / "data" / "cleaned" / "dongchedi",
    REPO_ROOT / "data" / "processed",
    REPO_ROOT / "data" / "multimodal",
    REPO_ROOT / "data" / "assets",
    REPO_ROOT / "data" / "state" / "agent_pipeline",
    REPO_ROOT / "data" / "state" / "job_state.json",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按新三模块策略重爬懂车帝车系数据")
    parser.add_argument(
        "--series-ids",
        default="",
        help="逗号分隔的 series_id 列表；为空时自动收集",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=20,
        help="自动收集模式下目标抓取车系数",
    )
    parser.add_argument(
        "--max-expand-requests",
        type=int,
        default=20,
        help="自动收集车系池时的扩展请求数",
    )
    parser.add_argument(
        "--clear-data",
        action="store_true",
        help="重爬前清理旧的 raw/cleaned/processed/multimodal 和状态目录",
    )
    parser.add_argument(
        "--summary-path",
        default=str(REPO_ROOT / "data" / "reports" / "refetch_series_summary.json"),
        help="重爬汇总输出路径",
    )
    parser.add_argument(
        "--min-interval-sec",
        type=float,
        default=0.8,
        help="请求最小间隔秒数",
    )
    parser.add_argument(
        "--max-retry",
        type=int,
        default=3,
        help="单请求重试次数",
    )
    return parser


def ensure_dirs() -> tuple[Path, Path]:
    cleaned_json_dir = REPO_ROOT / "data" / "cleaned" / "dongchedi" / "json"
    cleaned_md_dir = REPO_ROOT / "data" / "cleaned" / "dongchedi" / "markdown"
    cleaned_json_dir.mkdir(parents=True, exist_ok=True)
    cleaned_md_dir.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "data" / "reports").mkdir(parents=True, exist_ok=True)
    return cleaned_json_dir, cleaned_md_dir


def parse_series_ids(raw: str) -> list[str]:
    ids = []
    for item in (raw or "").split(","):
        sid = item.strip()
        if sid:
            ids.append(sid)
    return ids


def clear_old_data() -> list[str]:
    cleared = []
    for path in DEFAULT_CLEAR_TARGETS:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        cleared.append(str(path))
    return cleared


def main() -> int:
    args = build_parser().parse_args()
    cleaned_json_dir, cleaned_md_dir = ensure_dirs()

    cleared_paths: list[str] = []
    if args.clear_data:
        cleared_paths = clear_old_data()
        cleaned_json_dir, cleaned_md_dir = ensure_dirs()

    scraper = DongchediScraper(
        min_interval_sec=args.min_interval_sec,
        max_retry=args.max_retry,
    )
    cleaner = DongchediCleaner()

    series_ids = parse_series_ids(args.series_ids)
    if not series_ids:
        series_ids = scraper.collect_series_ids(
            target_count=max(args.target_count, 1),
            max_expand_requests=max(args.max_expand_requests, 1),
        )
    else:
        series_ids = series_ids[: max(args.target_count, 1)]

    summary = {
        "started_at": datetime.now().isoformat(),
        "clear_data": bool(args.clear_data),
        "cleared_paths": cleared_paths,
        "requested_series_count": len(series_ids),
        "success_count": 0,
        "failed_count": 0,
        "series_ids": series_ids,
        "items": [],
    }

    for series_id in series_ids:
        raw_json, fetch_meta = scraper.fetch_series_data(series_id)
        if not raw_json:
            summary["failed_count"] += 1
            summary["items"].append(
                {
                    "series_id": str(series_id),
                    "status": "failed",
                    "error": fetch_meta.get("error", ""),
                    "http_status": fetch_meta.get("http_status"),
                }
            )
            continue

        clean_record = cleaner.extract_clean_series_record(raw_json)
        markdown = cleaner.clean_record_to_markdown(clean_record)

        clean_json_path = cleaned_json_dir / f"series_{series_id}.json"
        clean_md_path = cleaned_md_dir / f"series_{series_id}.md"
        clean_json_path.write_text(json.dumps(clean_record, ensure_ascii=False, indent=2), encoding="utf-8")
        clean_md_path.write_text(markdown, encoding="utf-8")

        selected_modules = ((raw_json.get("props") or {}).get("pageProps") or {}).get("selectedImageModules") or {}
        summary["success_count"] += 1
        summary["items"].append(
            {
                "series_id": str(series_id),
                "status": "success",
                "series_name": (clean_record.get("series") or {}).get("series_name", ""),
                "image_success_count": fetch_meta.get("image_success_count", 0),
                "wg_sample_count": len((selected_modules.get("wg") or {}).get("selected_images", [])),
                "ns_sample_count": len((selected_modules.get("ns") or {}).get("selected_images", [])),
                "clean_json_path": str(clean_json_path),
                "clean_md_path": str(clean_md_path),
            }
        )

    summary["finished_at"] = datetime.now().isoformat()
    summary_path = Path(args.summary_path).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[PASS] 新三模块重爬完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
