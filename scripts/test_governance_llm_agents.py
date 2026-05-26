from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_pipeline import GovernanceOrchestrator
from cleaner.dcd_cleaner import DongchediCleaner


class FakeSemanticStore:
    def __init__(self):
        self.similarity_threshold = 0.94
        self.same_series_threshold = 0.88
        self.records: list[dict] = []

    def find_candidates(self, clean_record: dict, markdown_text: str) -> dict:
        hits = []
        series_name = str((clean_record.get("series", {}) or {}).get("series_name") or "")
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
        return {"query_text": markdown_text or "", "query_dim": 8, "hits": hits}

    def add_record(self, clean_record: dict, markdown_text: str, metadata: dict) -> None:
        series = clean_record.get("series", {}) or {}
        self.records.append(
            {
                "series_id": str(series.get("series_id") or ""),
                "series_name": str(series.get("series_name") or ""),
                "metadata": dict(metadata),
            }
        )


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_record() -> dict:
    return {
        "source": "unknown_source",
        "entity_type": "unknown_entity",
        "series": {
            "series_id": "llm-001",
            "series_name": "LLM治理测试车系",
            "brand_name": "演示品牌",
            "car_type": "SUV",
            "city_name": "全国",
        },
        "pricing": {
            "dealer_price_range": "20-25万",
            "official_price_range": "21-26万",
        },
        "scores": {"total_score": 4.5, "total_review_count": 128},
        "dimensions": [{"length_mm": 4900, "width_mm": 1900, "height_mm": 1700, "wheelbase_mm": 2900, "car_count": 2}],
        "images": [{"category": "appearance", "category_name": "外观", "color_count": 3, "sample_colors": ["黑色", "白色"]}],
        "models": [
            {"car_id": "llm-001-m1", "name": "LLM治理测试车系 车型1", "year": "2025", "official_price": "23.0万", "dealer_price": "22.5万", "tags": ["智能驾驶", "舒适座舱"]},
            {"car_id": "llm-001-m2", "name": "LLM治理测试车系 车型2", "year": "2025", "official_price": "24.0万", "dealer_price": "23.5万", "tags": ["长续航"]},
        ],
        "news": [
            {"category": "guide", "title": "LLM治理测试车系 空间、续航与配置解析", "publish_time": "2026-05-01", "watch_or_read_count": 1200, "has_video": False, "author": "测试作者"},
            {"category": "guide", "title": "LLM治理测试车系 改款信息", "publish_time": "2026-05-02", "watch_or_read_count": 1400, "has_video": False, "author": "测试作者"},
        ],
        "stats": {
            "model_count": 2,
            "dimension_group_count": 1,
            "image_group_count": 1,
            "news_count": 2,
        },
    }


def fake_llm_response(agent_name: str, system_prompt: str, user_prompt: str) -> str:
    if agent_name == "route":
        return json.dumps(
            {
                "channel": "dongchedi_llm",
                "route_decision": "llm_enhanced_pipeline",
                "template_version": "dongchedi_series_v2",
                "confidence": 0.86,
                "reason": "识别到这是汽车车系资料，建议走增强模板",
            },
            ensure_ascii=False,
        )
    if agent_name == "quality":
        return json.dumps(
            {
                "quality_score": 0.82,
                "rag_readiness": 0.8,
                "training_readiness": 0.78,
                "extra_issues": ["llm:comparison_dimension_missing"],
                "issue_groups": {"content": ["llm:comparison_dimension_missing"]},
                "repair_suggestion": "补充同价位对比维度与竞品差异描述",
                "reason": "正文信息完整，但缺少对比型信息",
            },
            ensure_ascii=False,
        )
    raise AssertionError(f"未知 agent_name: {agent_name}")


def main() -> int:
    artifact_dir = REPO_ROOT / "data" / "test_runs" / "governance_llm_agents"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    try:
        cleaner = DongchediCleaner()
        record = make_record()
        markdown = cleaner.clean_record_to_markdown(record)

        llm_orchestrator = GovernanceOrchestrator(
            base_dir=str(artifact_dir / "llm_on"),
            enable_llm_agents=True,
            agent_llm_provider="ollama",
            agent_llm_model="mock-agent",
            llm_response_provider=fake_llm_response,
            semantic_store=FakeSemanticStore(),
        )
        llm_result = llm_orchestrator.govern(record, markdown, "llm-batch")
        assert_true(llm_result.metadata.get("agent_llm_enabled") is True, "LLM 模式应启用")
        assert_true(llm_result.metadata.get("route_llm_used") is True, "route agent 应使用 LLM")
        assert_true(llm_result.metadata.get("quality_llm_used") is True, "quality agent 应使用 LLM")
        assert_true(llm_result.route_result.route_decision == "llm_enhanced_pipeline", "route 决策未使用 LLM 输出")
        assert_true(llm_result.route_result.template_version == "dongchedi_series_v2", "route 模板未使用 LLM 输出")
        assert_true("llm:comparison_dimension_missing" in llm_result.quality_result.issues, "quality 未合并 LLM 问题标签")
        assert_true(
            llm_result.quality_result.repair_suggestion == "补充同价位对比维度与竞品差异描述",
            "quality 未使用 LLM 修复建议",
        )
        assert_true(any(log.get("step") == "route" and (log.get("llm_agent", {}) or {}).get("llm_used") for log in llm_result.audit_logs), "route audit 缺少 llm_used")
        assert_true(any(log.get("step") == "quality" and (log.get("llm_agent", {}) or {}).get("llm_used") for log in llm_result.audit_logs), "quality audit 缺少 llm_used")

        no_llm_orchestrator = GovernanceOrchestrator(base_dir=str(artifact_dir / "llm_off"), semantic_store=FakeSemanticStore())
        no_llm_result = no_llm_orchestrator.govern(record, markdown, "rule-batch")
        assert_true(no_llm_result.metadata.get("agent_llm_enabled") is False, "规则模式不应启用 LLM")
        assert_true(no_llm_result.metadata.get("route_llm_used") is False, "规则模式 route 不应使用 LLM")
        assert_true(no_llm_result.metadata.get("quality_llm_used") is False, "规则模式 quality 不应使用 LLM")

        print("[PASS] LLM Agent 接入测试通过")
        print(
            json.dumps(
                {
                    "llm_route": llm_result.route_result.model_dump(),
                    "llm_quality": llm_result.quality_result.model_dump(),
                    "rule_route": no_llm_result.route_result.model_dump(),
                    "rule_quality": no_llm_result.quality_result.model_dump(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        print(f"[FAIL] LLM Agent 接入测试失败: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
