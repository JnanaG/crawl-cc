from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_pipeline import GovernanceOrchestrator
from cleaner.dcd_cleaner import DongchediCleaner


DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "data" / "test_runs"
SUPPORTED_EMBEDDING_PROVIDERS = {
    "sentence_transformers",
    "fastembed",
    "openai_compatible",
    "ollama",
}


class FakeSemanticStore:
    def __init__(self):
        self.similarity_threshold = 0.94
        self.same_series_threshold = 0.88
        self.records: list[dict] = []

    def find_candidates(self, clean_record: dict, markdown_text: str) -> dict:
        series = clean_record.get("series", {}) or {}
        series_name = str(series.get("series_name") or "")
        hits = []
        for item in self.records[:3]:
            score = 0.99 if item["series_name"] == series_name else 0.2
            hits.append(
                {
                    "score": score,
                    "series_id": item["series_id"],
                    "series_name": item["series_name"],
                    "content_hash": item["metadata"].get("content_hash"),
                    "normalized_hash": item["metadata"].get("normalized_hash"),
                    "record_hash": item["metadata"].get("record_hash"),
                }
            )
        hits.sort(key=lambda row: row["score"], reverse=True)
        return {
            "query_text": markdown_text or "",
            "query_dim": 8,
            "hits": hits,
        }

    def add_record(self, clean_record: dict, markdown_text: str, metadata: dict) -> None:
        series = clean_record.get("series", {}) or {}
        self.records.append(
            {
                "series_id": str(series.get("series_id") or ""),
                "series_name": str(series.get("series_name") or ""),
                "markdown_text": markdown_text,
                "metadata": dict(metadata),
            }
        )


class BrokenSemanticStore:
    similarity_threshold = 0.94
    same_series_threshold = 0.88

    def find_candidates(self, clean_record: dict, markdown_text: str) -> dict:
        raise RuntimeError("broken semantic store")

    def add_record(self, clean_record: dict, markdown_text: str, metadata: dict) -> None:
        raise RuntimeError("broken semantic store")


def make_record(
    *,
    series_id: str,
    series_name: str,
    brand_name: str = "演示品牌",
    car_type: str = "SUV",
    dealer_price_range: str = "20-30万",
    official_price_range: str = "22-32万",
    model_count: int = 2,
    news_count: int = 2,
    image_group_count: int = 1,
    body_variant: str = "default",
) -> dict:
    models = []
    for idx in range(max(model_count, 0)):
        models.append(
            {
                "car_id": f"{series_id}-m{idx + 1}",
                "name": f"{series_name} 车型{idx + 1}",
                "year": "2025",
                "official_price": f"{24 + idx}.0万",
                "dealer_price": f"{23 + idx}.5万",
                "tags": ["智能驾驶", "舒适座舱"] if idx == 0 else ["长续航"],
            }
        )

    news = []
    for idx in range(max(news_count, 0)):
        news.append(
            {
                "category": "guide",
                "title": f"{series_name} 新闻 {idx + 1}",
                "publish_time": f"2026-05-{idx + 1:02d}",
                "watch_or_read_count": 1000 + idx * 100,
                "has_video": False,
                "author": "测试作者",
            }
        )

    images = []
    for idx in range(max(image_group_count, 0)):
        images.append(
            {
                "category": "appearance",
                "category_name": "外观",
                "color_count": 2 + idx,
                "sample_colors": ["黑色", "白色"],
            }
        )

    scores = {"total_score": 4.3, "total_review_count": 123}
    dimensions = [
        {
            "length_mm": 4900,
            "width_mm": 1900,
            "height_mm": 1700,
            "wheelbase_mm": 2900,
            "car_count": max(model_count, 0),
        }
    ]

    if body_variant == "rich":
        news[0]["title"] = f"{series_name} 空间、能耗与配置解析"
    elif body_variant == "similar":
        news[0]["title"] = f"{series_name} 空间配置解读"
    elif body_variant == "poor":
        models = []
        news = []
        images = []
        dealer_price_range = ""
        official_price_range = ""
        brand_name = ""

    return {
        "source": "dongchedi",
        "entity_type": "car_series",
        "series": {
            "series_id": series_id,
            "series_name": series_name,
            "brand_name": brand_name,
            "car_type": car_type,
            "city_name": "全国",
        },
        "pricing": {
            "dealer_price_range": dealer_price_range,
            "official_price_range": official_price_range,
        },
        "scores": scores,
        "dimensions": dimensions,
        "images": images,
        "models": models,
        "news": news,
        "stats": {
            "model_count": model_count,
            "dimension_group_count": len(dimensions),
            "image_group_count": image_group_count,
            "news_count": news_count,
        },
    }


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


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


def run_accept_case(orchestrator: GovernanceOrchestrator, cleaner: DongchediCleaner) -> dict:
    record = make_record(series_id="accept-001", series_name="治理测试车系A", body_variant="rich")
    markdown = cleaner.clean_record_to_markdown(record)
    result = orchestrator.govern(record, markdown, "test-batch")
    assert_true(result.decision_result.decision == "accept", "高质量样本应被 accept")
    assert_true(result.quality_result.quality_tier in {"high", "medium"}, "高质量样本 tier 不合理")
    assert_true(result.metadata.get("governance_duration_ms", 0.0) >= 0.0, "缺少治理耗时")
    assert_true(len(result.audit_logs) == 5, "治理节点数量应为 5")
    return result.model_dump()


def run_review_case(orchestrator: GovernanceOrchestrator, cleaner: DongchediCleaner) -> dict:
    record = make_record(
        series_id="review-001",
        series_name="治理测试车系B",
        model_count=0,
        news_count=0,
        image_group_count=0,
        body_variant="poor",
    )
    markdown = cleaner.clean_record_to_markdown(record)
    result = orchestrator.govern(record, markdown, "test-batch")
    assert_true(result.decision_result.decision == "review", "低质量样本应被 review")
    assert_true(result.quality_result.quality_tier in {"low", "critical"}, "低质量样本 tier 不合理")
    assert_true(result.quality_result.rag_readiness < 0.45, "低质量样本 rag_readiness 应较低")
    return result.model_dump()


def run_dedup_case(orchestrator: GovernanceOrchestrator, cleaner: DongchediCleaner) -> dict:
    first = make_record(series_id="dedup-001", series_name="治理测试车系C", body_variant="rich")
    second = make_record(series_id="dedup-002", series_name="治理测试车系C", body_variant="rich")
    first_result = orchestrator.govern(first, cleaner.clean_record_to_markdown(first), "test-batch")
    second_result = orchestrator.govern(second, cleaner.clean_record_to_markdown(second), "test-batch")
    assert_true(first_result.decision_result.decision == "accept", "去重首条样本应先入库")
    assert_true(second_result.semantic_dedup_result.is_duplicate, "重复样本应命中去重")
    assert_true(second_result.decision_result.decision == "drop", "重复样本应被 drop")
    assert_true(
        second_result.semantic_dedup_result.duplicate_type in {"exact_content", "normalized_text", "semantic_same_series", "semantic_similarity"},
        "重复样本 duplicate_type 不符合预期",
    )
    return {
        "first": first_result.model_dump(),
        "second": second_result.model_dump(),
    }


def run_failure_audit_case(base_dir: Path, cleaner: DongchediCleaner) -> dict:
    orchestrator = GovernanceOrchestrator(base_dir=str(base_dir), semantic_store=BrokenSemanticStore())
    record = make_record(series_id="fail-001", series_name="治理测试车系D")
    markdown = cleaner.clean_record_to_markdown(record)
    try:
        orchestrator.govern(record, markdown, "test-batch-fail")
    except Exception as exc:
        failure_path = base_dir / "audit" / "agent_pipeline" / "governance_failures.jsonl"
        assert_true(failure_path.exists(), "失败审计文件未生成")
        rows = read_jsonl(failure_path)
        assert_true(len(rows) >= 1, "失败审计文件为空")
        assert_true(rows[-1]["error_type"] in {"ValueError", "RuntimeError"}, "失败错误类型异常")
        return {"error": str(exc), "failure_row": rows[-1]}
    raise AssertionError("损坏的 semantic store 应触发异常")


def validate_audit_outputs(base_dir: Path) -> dict:
    audit_dir = base_dir / "audit" / "agent_pipeline"
    audit_path = audit_dir / "governance_audit.jsonl"
    steps_path = audit_dir / "governance_steps.jsonl"
    summary_path = audit_dir / "governance_summary.json"
    review_path = base_dir / "review" / "agent_pipeline" / "review_queue.jsonl"

    assert_true(audit_path.exists(), "缺少 governance_audit.jsonl")
    assert_true(steps_path.exists(), "缺少 governance_steps.jsonl")
    assert_true(summary_path.exists(), "缺少 governance_summary.json")
    assert_true(review_path.exists(), "缺少 review_queue.jsonl")

    audits = read_jsonl(audit_path)
    steps = read_jsonl(steps_path)
    summary = read_json(summary_path)
    reviews = read_jsonl(review_path)

    assert_true(len(audits) >= 4, "审计行数不足")
    assert_true(len(steps) >= 15, "节点级审计行数不足")
    assert_true(summary.get("total_records", 0) >= 4, "审计汇总 total_records 不正确")
    assert_true("semantic_dedup" in summary.get("step_stats", {}), "缺少 semantic_dedup 汇总")
    assert_true(len(reviews) >= 1, "review 队列应至少有一条")

    return {
        "audit_rows": len(audits),
        "step_rows": len(steps),
        "review_rows": len(reviews),
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="智能治理链路烟测脚本")
    parser.add_argument(
        "--artifact-dir",
        default=str(DEFAULT_ARTIFACT_ROOT / f"governance_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        help="测试产物目录，默认写到 data/test_runs 下",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="保留测试产物目录，便于人工查看审计结果",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).resolve()
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    cleaner = DongchediCleaner()
    orchestrator = GovernanceOrchestrator(base_dir=str(artifact_dir), semantic_store=FakeSemanticStore())

    try:
        results = {
            "accept_case": run_accept_case(orchestrator, cleaner),
            "review_case": run_review_case(orchestrator, cleaner),
            "dedup_case": run_dedup_case(orchestrator, cleaner),
            "failure_case": run_failure_audit_case(artifact_dir, cleaner),
            "audit_validation": validate_audit_outputs(artifact_dir),
        }
        print("[PASS] 治理链路测试通过")
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"[FAIL] 治理链路测试失败: {exc}", file=sys.stderr)
        if args.keep_artifacts:
            print(f"[INFO] 保留测试产物: {artifact_dir}", file=sys.stderr)
        return 1
    finally:
        if not args.keep_artifacts and artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
