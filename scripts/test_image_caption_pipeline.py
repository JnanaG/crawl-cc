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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def build_image_assets() -> list[dict]:
    return [
        {
            "asset_id": "img001",
            "series_id": "s-alpha",
            "series_name": "测试车系 Alpha",
            "brand_name": "测试品牌",
            "image_url": "https://example.com/alpha-wg-1.jpg",
            "category": "wg",
            "category_name": "外观",
            "source_section": "series_gallery",
            "image_role": "gallery",
            "rank": 0,
            "width": 1280,
            "height": 720,
            "car_id": "1001",
            "color_id": "501",
            "color_name": "皓月白",
            "raw_ref": "mock",
            "quality_score": 0.95,
            "quality_flags": [],
            "is_usable": True,
        },
        {
            "asset_id": "img002",
            "series_id": "s-alpha",
            "series_name": "测试车系 Alpha",
            "brand_name": "测试品牌",
            "image_url": "https://example.com/alpha-ns-1.jpg",
            "category": "ns",
            "category_name": "内饰",
            "source_section": "image_floor",
            "image_role": "gallery",
            "rank": 1,
            "width": 960,
            "height": 640,
            "car_id": "1001",
            "color_id": "",
            "color_name": "",
            "raw_ref": "mock",
            "quality_score": 0.89,
            "quality_flags": [],
            "is_usable": True,
        },
    ]


def build_text_corpus() -> list[dict]:
    return [
        {
            "metadata": {
                "source": "dongchedi",
                "url": "https://example.com/series/alpha",
                "title": "测试车系 Alpha",
                "series_id": "s-alpha",
                "brand_name": "测试品牌",
                "car_type": "SUV",
                "chunk_index": 0,
                "total_chunks": 1,
                "tokens": 32,
                "modality": "text",
            },
            "text": "测试车系 Alpha 官方指导价 20 万左右，定位紧凑型 SUV，偏年轻化设计。",
        }
    ]


def main() -> int:
    artifact_dir = REPO_ROOT / "data" / "test_runs" / "image_caption_pipeline"
    multimodal_dir = artifact_dir / "multimodal"
    vector_dir = artifact_dir / "vector_store"

    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    try:
        write_jsonl(multimodal_dir / "image_assets.jsonl", build_image_assets())
        write_jsonl(artifact_dir / "text_corpus.jsonl", build_text_corpus())

        caption_cmd = [
            PYTHON,
            str(REPO_ROOT / "scripts" / "build_image_caption_corpus.py"),
            "--input",
            str(multimodal_dir / "image_assets.jsonl"),
            "--output-dir",
            str(multimodal_dir),
            "--caption-provider",
            "heuristic",
        ]
        caption_proc = subprocess.run(
            caption_cmd,
            cwd=str(REPO_ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        assert_true(
            caption_proc.returncode == 0,
            f"caption 构建脚本返回码异常: {caption_proc.returncode}\n{caption_proc.stderr}",
        )

        caption_rows = read_jsonl(multimodal_dir / "image_caption_corpus.jsonl")
        caption_summary = read_json(multimodal_dir / "image_caption_summary.json")
        assert_true(len(caption_rows) == 2, f"caption 语料数量不正确: {len(caption_rows)}")
        assert_true(
            all(row["metadata"]["modality"] == "image_caption" for row in caption_rows),
            "caption 语料 modality 未正确标记",
        )
        assert_true(
            caption_summary["caption_record_count"] == 2,
            "caption summary 统计不正确",
        )

        build_cmd = [
            PYTHON,
            str(REPO_ROOT / "rag_llm_demo.py"),
            "build",
            "--input",
            str(artifact_dir / "text_corpus.jsonl"),
            "--extra-inputs",
            str(multimodal_dir / "image_caption_corpus.jsonl"),
            "--embedding-provider",
            "hash",
            "--index",
            str(vector_dir / "test.index"),
            "--records",
            str(vector_dir / "test_records.jsonl"),
            "--meta",
            str(vector_dir / "test_meta.json"),
            "--disable-storage",
        ]
        build_proc = subprocess.run(
            build_cmd,
            cwd=str(REPO_ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        assert_true(
            build_proc.returncode == 0,
            f"RAG build 返回码异常: {build_proc.returncode}\n{build_proc.stderr}",
        )

        records = read_jsonl(vector_dir / "test_records.jsonl")
        meta = read_json(vector_dir / "test_meta.json")
        assert_true(len(records) == 3, f"合并后的 records 数量不正确: {len(records)}")
        assert_true(meta["size"] == 3, "向量库 meta size 不正确")
        assert_true(
            any(row["metadata"].get("modality") == "image_caption" for row in records),
            "向量库中缺少 image_caption 语料",
        )

        print("[PASS] 图片 caption -> RAG 链路测试通过")
        print(json.dumps({"caption_summary": caption_summary, "vector_meta": meta}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"[FAIL] 图片 caption -> RAG 链路测试失败: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
