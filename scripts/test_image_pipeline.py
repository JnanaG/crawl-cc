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


def write_binary(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_raw_payload() -> dict:
    return {
        "props": {
            "pageProps": {
                "seriesId": "900001",
                "seriesName": "测试车系 Alpha",
                "seriesHomeHead": {
                    "series_id": 900001,
                    "series_name": "测试车系 Alpha",
                    "brand_name": "测试品牌",
                    "cover_url": "https://example.com/cover-alpha.image",
                    "series_image_info_list": [],
                    "pics_summary_info": [],
                },
                "selectedImageModules": {
                    "wg": {
                        "module_key": "wg",
                        "category_name": "外观",
                        "required_view": "正面前脸图",
                        "selected_images": [
                            {
                                "rank": 0,
                                "car_id": 1001,
                                "car_name": "2026款 405 Air",
                                "image_url": "https://example.com/alpha-wg-1.jpg",
                                "available_color_names": ["皓月白", "流金紫"],
                            },
                            {
                                "rank": 1,
                                "car_id": 1002,
                                "car_name": "2026款 506 Max",
                                "image_url": "https://example.com/alpha-wg-2.jpg",
                                "available_color_names": ["松野绿"],
                            },
                        ],
                        "available_colors": [
                            {"color_name": "皓月白", "hex_color": "#ffffff"},
                            {"color_name": "流金紫", "hex_color": "#aaaaaa"},
                        ],
                    },
                    "ns": {
                        "module_key": "ns",
                        "category_name": "内饰",
                        "required_view": "正面主副驾拍摄图",
                        "selected_images": [
                            {
                                "rank": 0,
                                "car_id": 1001,
                                "car_name": "2026款 405 Air",
                                "image_url": "https://example.com/alpha-ns-1.jpg",
                                "available_color_names": ["墨玉黑", "玉石白"],
                            },
                            {
                                "rank": 1,
                                "car_id": 1002,
                                "car_name": "2026款 506 Max",
                                "image_url": "https://example.com/alpha-ns-2.jpg",
                                "available_color_names": ["玉石白"],
                            },
                        ],
                        "available_colors": [
                            {"color_name": "墨玉黑", "hex_color": "#111111"},
                            {"color_name": "玉石白", "hex_color": "#f4f4f4"},
                        ],
                    },
                },
                "imageFloorData": {"floor_head_list": [], "floor_image_list": []},
            }
        }
    }


def main() -> int:
    artifact_dir = REPO_ROOT / "data" / "test_runs" / "image_pipeline"
    raw_dir = artifact_dir / "raw"
    output_dir = artifact_dir / "multimodal"

    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    try:
        write_json(raw_dir / "series_900001.json", build_raw_payload())
        image_dir = raw_dir / "images" / "series_900001"
        local_cover = image_dir / "00000_series_cover_cover001.jpg"
        local_wg = image_dir / "00000_module_wg_wg001.jpg"
        local_ns = image_dir / "00000_module_ns_ns001.jpg"
        write_binary(local_cover, b"cover-image")
        write_binary(local_wg, b"wg-image")
        write_binary(local_ns, b"ns-image")
        write_json(
            raw_dir / "series_900001_images.json",
            {
                "series_id": "900001",
                "image_count": 3,
                "success_count": 3,
                "failed_count": 0,
                "items": [
                    {
                        "asset_id": "cover001",
                        "series_id": "900001",
                        "image_url": "https://example.com/cover-alpha.image",
                        "source_section": "series_cover",
                        "image_role": "cover",
                        "rank": 0,
                        "status": "success",
                        "local_path": str(local_cover.resolve()),
                        "content_type": "image/jpeg",
                        "bytes": 11,
                    },
                    {
                        "asset_id": "wg001",
                        "series_id": "900001",
                        "image_url": "https://example.com/alpha-wg-1.jpg",
                        "source_section": "module_wg",
                        "image_role": "module_sample",
                        "rank": 0,
                        "status": "success",
                        "local_path": str(local_wg.resolve()),
                        "content_type": "image/jpeg",
                        "bytes": 8,
                    },
                    {
                        "asset_id": "ns001",
                        "series_id": "900001",
                        "image_url": "https://example.com/alpha-ns-1.jpg",
                        "source_section": "module_ns",
                        "image_role": "module_sample",
                        "rank": 0,
                        "status": "success",
                        "local_path": str(local_ns.resolve()),
                        "content_type": "image/jpeg",
                        "bytes": 8,
                    },
                ],
            },
        )

        command = [
            PYTHON,
            str(REPO_ROOT / "scripts" / "build_image_dataset.py"),
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(output_dir),
        ]
        proc = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )

        assert_true(proc.returncode == 0, f"图片数据链路脚本返回码异常: {proc.returncode}\n{proc.stderr}")

        assets = read_jsonl(output_dir / "image_assets.jsonl")
        classification = read_jsonl(output_dir / "image_classification_manifest.jsonl")
        summary = read_json(output_dir / "image_dataset_summary.json")

        assert_true(len(assets) >= 5, f"图片资产数量过少: {len(assets)}")
        assert_true(summary["series_count"] == 1, "series_count 不正确")
        assert_true(summary["asset_with_local_file_count"] >= 3, "本地原图关联数量不正确")
        assert_true(summary["usable_asset_count"] >= 5, "usable_asset_count 偏低")
        assert_true(any(row["category_name"] == "外观" for row in classification), "缺少外观分类样本")
        assert_true(any(row["category_name"] == "内饰" for row in classification), "缺少内饰分类样本")
        assert_true(all(row["dataset_split"] in {"train", "val", "test"} for row in classification), "分类样本未正确打 split")
        assert_true(any(row.get("local_exists") for row in assets), "图片资产未回填本地原图路径")
        assert_true((output_dir / "image_assets.parquet").exists(), "缺少 image_assets.parquet")
        assert_true((output_dir / "image_series_summary.parquet").exists(), "缺少 image_series_summary.parquet")

        print("[PASS] 图片数据链路测试通过")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"[FAIL] 图片数据链路测试失败: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
