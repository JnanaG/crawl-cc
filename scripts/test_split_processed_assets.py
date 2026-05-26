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
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_item(
    series_id: str,
    title: str,
    chunk_index: int,
    *,
    quality_score: float,
    rag_readiness: float,
    training_readiness: float,
    route_confidence: float = 0.9,
    governance_decision: str = "accept",
    model_count: int = 3,
    news_count: int = 2,
    brand_name: str = "测试品牌",
    text: str = "",
) -> dict:
    return {
        "metadata": {
            "source": "dongchedi",
            "title": title,
            "series_id": series_id,
            "brand_name": brand_name,
            "car_type": "SUV",
            "model_count": model_count,
            "news_count": news_count,
            "chunk_index": chunk_index,
            "governance_decision": governance_decision,
            "quality_score": quality_score,
            "rag_readiness": rag_readiness,
            "training_readiness": training_readiness,
            "route_confidence": route_confidence,
            "dedup_duplicate_type": "none",
        },
        "text": text or f"{title} 官方指导价 20-25万，配置齐全，动力和续航表现稳定，近期有改款消息。 chunk={chunk_index}",
    }


def main() -> int:
    artifact_dir = REPO_ROOT / "data" / "test_runs" / "split_assets"
    input_path = artifact_dir / "processed" / "dongchedi_training_data.jsonl"
    output_dir = artifact_dir / "assets"

    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    try:
        rows = [
            build_item("s1", "星舰1", 0, quality_score=0.92, rag_readiness=0.91, training_readiness=0.89),
            build_item("s1", "星舰1", 1, quality_score=0.9, rag_readiness=0.87, training_readiness=0.86),
            build_item("s1", "星舰1", 2, quality_score=0.88, rag_readiness=0.82, training_readiness=0.8),
            build_item("s2", "银河2", 0, quality_score=0.78, rag_readiness=0.66, training_readiness=0.7),
            build_item("s2", "银河2", 1, quality_score=0.74, rag_readiness=0.64, training_readiness=0.68),
            build_item("s3", "边缘3", 0, quality_score=0.7, rag_readiness=0.4, training_readiness=0.58, route_confidence=0.55),
        ]
        write_jsonl(input_path, rows)

        command = [
            PYTHON,
            str(REPO_ROOT / "scripts" / "split_processed_assets.py"),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--max-training-chunks-per-series",
            "2",
        ]
        proc = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )

        assert_true(proc.returncode == 0, f"分流脚本返回码异常: {proc.returncode}\n{proc.stderr}")
        rag_rows = read_jsonl(output_dir / "rag" / "rag_corpus.jsonl")
        training_rows = read_jsonl(output_dir / "training" / "training_corpus.jsonl")
        eval_rows = read_jsonl(output_dir / "evaluation" / "eval_candidates.jsonl")
        summary = read_json(output_dir / "asset_split_summary.json")

        assert_true(len(rag_rows) == 5, f"RAG 语料数量不正确: {len(rag_rows)}")
        assert_true(len(training_rows) == 4, f"训练语料数量不正确: {len(training_rows)}")
        assert_true(len(eval_rows) == 2, f"评测候选数量不正确: {len(eval_rows)}")
        assert_true(summary["rag_rows"] == 5, "summary rag_rows 不正确")
        assert_true(summary["training_rows"] == 4, "summary training_rows 不正确")
        assert_true(summary["eval_candidate_rows"] == 2, "summary eval_candidate_rows 不正确")
        assert_true(all(row["metadata"]["asset_split"] == "rag_corpus" for row in rag_rows), "RAG 语料未打上 asset_split")
        assert_true(all(row["metadata"]["asset_split"] == "training_corpus" for row in training_rows), "训练语料未打上 asset_split")
        assert_true(all(row["series_name"] in {"星舰1", "银河2"} for row in eval_rows), "评测候选车系不正确")
        assert_true(
            any(question["question"].startswith("星舰1") for question in eval_rows[0]["suggested_questions"] + eval_rows[1]["suggested_questions"]),
            "评测候选未生成建议问题",
        )

        print("[PASS] processed 数据分流测试通过")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"[FAIL] processed 数据分流测试失败: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
