import json
import os
import time
from datetime import datetime
from loguru import logger

from agent_pipeline import GovernanceOrchestrator
from scraper.dcd_scraper import DongchediScraper
from cleaner.dcd_cleaner import DongchediCleaner
from utils.llm_processor import LLMDataProcessor
from utils.parquet_exporter import ParquetExporter
from utils.job_state import JobStateStore
from utils.stats_collector import StatsCollector
from quality.data_quality import DataQualityEngine


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default)).strip()
    try:
        return int(value)
    except ValueError:
        logger.warning(f"环境变量 {name}={value} 不是合法整数，回退默认值 {default}")
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main():
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join("data", "logs"), exist_ok=True)
    logger.remove()
    logger.add(
        os.path.join("data", "logs", f"run_{batch_id}.log"),
        level="DEBUG",
        encoding="utf-8",
    )
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.info("启动懂车帝汽车数据爬虫流水线...")

    max_retry = 3
    target_training_records = _env_int("TARGET_TRAINING_RECORDS", 1000)
    target_series_pool = _env_int("TARGET_SERIES_POOL", 300)
    max_chunk_tokens = _env_int("MAX_CHUNK_TOKENS", 100)
    max_expand_requests = _env_int("MAX_SERIES_EXPAND_REQUESTS", 40)

    scraper = DongchediScraper(min_interval_sec=0.8, max_retry=max_retry)
    cleaner = DongchediCleaner()
    processor = LLMDataProcessor()
    parquet_exporter = ParquetExporter()
    quality_engine = DataQualityEngine()
    state_store = JobStateStore(os.path.join("data", "state", "job_state.json"))
    stats_collector = StatsCollector()
    governance_orchestrator = GovernanceOrchestrator(
        base_dir="data",
        enable_llm_agents=_env_bool("AGENT_ENABLE_LLM", False),
        agent_llm_provider=os.getenv("AGENT_LLM_PROVIDER", "ollama").strip() or "ollama",
        agent_llm_model=(os.getenv("AGENT_LLM_MODEL") or "").strip() or None,
        agent_llm_api_base=(os.getenv("AGENT_LLM_API_BASE") or "").strip() or None,
        agent_llm_api_key=(os.getenv("AGENT_LLM_API_KEY") or "").strip() or None,
        llm_fail_open=_env_bool("AGENT_LLM_FAIL_OPEN", True),
    )

    cleaned_json_dir = os.path.join("data", "cleaned", "dongchedi", "json")
    cleaned_md_dir = os.path.join("data", "cleaned", "dongchedi", "markdown")
    report_dir = os.path.join("data", "reports")
    os.makedirs(cleaned_json_dir, exist_ok=True)
    os.makedirs(cleaned_md_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # 1. 收集车系池，用于扩量抓取
    target_ids = scraper.collect_series_ids(
        target_count=target_series_pool,
        max_expand_requests=max_expand_requests,
    )
    if not target_ids:
        logger.error("未能获取到任何车系ID，程序退出。")
        return
    stats_collector.set_total_tasks(len(target_ids))
    logger.info(
        f"本次目标: 至少 {target_training_records} 条训练样本，"
        f"车系池上限 {target_series_pool}，chunk上限 {max_chunk_tokens} tokens"
    )

    structured_data = []
    clean_records = []
    clean_validations = []
    dropped_training_items = 0
    governance_counters = {"accept": 0, "repair": 0, "review": 0, "drop": 0}

    def append_training_items(
        clean_record: dict,
        series_id: str,
        crawl_time: str,
        governance_result,
    ) -> tuple[int, int, int]:
        nonlocal dropped_training_items
        markdown_text = cleaner.clean_record_to_markdown(clean_record)
        series_info = clean_record.get("series", {})
        stats = clean_record.get("stats", {})
        series_name = series_info.get("series_name", f"未知车系_{series_id}")
        chunks = processor.chunk_text(markdown_text, max_chunk_tokens=max_chunk_tokens)

        kept = 0
        dropped = 0
        for c_idx, chunk in enumerate(chunks):
            item = {
                "metadata": {
                    "source": "dongchedi",
                    "url": f"https://www.dongchedi.com/auto/series/{series_id}",
                    "title": series_name,
                    "series_id": series_id,
                    "brand_name": series_info.get("brand_name"),
                    "car_type": series_info.get("car_type"),
                    "model_count": stats.get("model_count", 0),
                    "news_count": stats.get("news_count", 0),
                    "crawl_timestamp": crawl_time,
                    "chunk_index": c_idx,
                    "total_chunks": len(chunks),
                    "tokens": processor.count_tokens(chunk),
                    "trace_id": governance_result.trace_id,
                    "batch_id": batch_id,
                    "governance_decision": governance_result.decision_result.decision,
                    "governance_reason": governance_result.decision_result.reason,
                    "quality_score": governance_result.quality_result.quality_score,
                    "quality_issues": governance_result.quality_result.issues,
                    "quality_tier": governance_result.quality_result.quality_tier,
                    "quality_dimensions": governance_result.quality_result.dimensions,
                    "quality_issue_groups": governance_result.quality_result.issue_groups,
                    "rag_readiness": governance_result.quality_result.rag_readiness,
                    "training_readiness": governance_result.quality_result.training_readiness,
                    "repair_suggestion": governance_result.quality_result.repair_suggestion,
                    "route_channel": governance_result.route_result.channel,
                    "route_confidence": governance_result.route_result.confidence,
                    "dedup_duplicate_type": governance_result.semantic_dedup_result.duplicate_type,
                    "dedup_confidence": governance_result.semantic_dedup_result.confidence,
                    "semantic_evidence": governance_result.semantic_dedup_result.evidence,
                },
                "text": chunk,
            }
            training_qc = quality_engine.validate_training_item(item)
            if training_qc["is_valid"]:
                structured_data.append(item)
                kept += 1
            else:
                dropped += 1
                dropped_training_items += 1
                logger.debug(
                    f"训练样本被过滤 series_id={series_id}, chunk={c_idx}, reason={training_qc['errors']}"
                )
        return kept, dropped, len(chunks)

    # 2. 依次抓取车系详情并做清洗、Chunking
    for idx, series_id in enumerate(target_ids, start=1):
        if len(structured_data) >= target_training_records:
            logger.info(
                f"已达到目标训练样本数 {target_training_records}，提前结束抓取。"
            )
            break

        logger.info(f"[{idx}/{len(target_ids)}] 正在处理车系: {series_id}")
        can_process, reason = state_store.can_process(series_id, max_retry=max_retry)
        if not can_process:
            logger.warning(f"  > 跳过车系 {series_id}: {reason}")
            state_store.mark_skipped(series_id, reason)
            stats_collector.add_skipped()
            # 断点续跑时，若该车系历史成功，尝试复用 cleaned JSON 回填训练样本
            if reason == "already_success":
                clean_json_path = os.path.join(cleaned_json_dir, f"series_{series_id}.json")
                if os.path.exists(clean_json_path):
                    try:
                        with open(clean_json_path, "r", encoding="utf-8") as f:
                            clean_record = json.load(f)
                        clean_qc = quality_engine.validate_clean_record(clean_record)
                        clean_validations.append(clean_qc)
                        clean_records.append(clean_record)
                        markdown_text = cleaner.clean_record_to_markdown(clean_record)
                        governance_result = governance_orchestrator.govern(
                            clean_record=clean_record,
                            markdown_text=markdown_text,
                            batch_id=batch_id,
                        )
                        governance_counters[governance_result.decision_result.decision] += 1
                        if governance_result.decision_result.decision in {"accept", "repair"}:
                            kept, _, chunk_count = append_training_items(
                                clean_record=clean_record,
                                series_id=str(series_id),
                                crawl_time=datetime.now().isoformat(),
                                governance_result=governance_result,
                            )
                            logger.info(
                                f"  > 回填历史车系 {series_id}: decision={governance_result.decision_result.decision}, "
                                f"chunk={chunk_count}, kept={kept}"
                            )
                        else:
                            logger.info(
                                f"  > 回填历史车系 {series_id}: decision={governance_result.decision_result.decision}, "
                                "跳过 processed 输出"
                            )
                    except Exception as e:
                        logger.warning(f"  > 回填历史车系 {series_id} 失败: {e}")
            continue

        # 2.1 抓取原始数据并保存 HTML 和 JSON (已经在 scraper 内部完成)
        started_at = time.monotonic()
        state_store.mark_running(series_id)
        raw_json, fetch_meta = scraper.fetch_series_data(series_id)
        if not raw_json:
            elapsed = time.monotonic() - started_at
            state_store.mark_failed(
                series_id,
                fetch_meta.get("error", "unknown_error"),
                meta={
                    "url": fetch_meta.get("url"),
                    "http_status": fetch_meta.get("http_status"),
                    "retry_count": fetch_meta.get("retry_count", 0),
                    "stage": "fetch",
                },
            )
            stats_collector.add_failed(
                duration_sec=elapsed,
                retries=fetch_meta.get("retry_count", 0),
                http_status=fetch_meta.get("http_status"),
                error=fetch_meta.get("error", "unknown_error"),
            )
            continue

        # 2.2 数据清洗：落地 cleaned JSON 和 Markdown
        clean_record = cleaner.extract_clean_series_record(raw_json)
        markdown_text = cleaner.clean_record_to_markdown(clean_record)
        series_info = clean_record.get("series", {})
        stats = clean_record.get("stats", {})

        # 2.2.1 质量规则：cleaned record 字段完整性与异常检测
        clean_qc = quality_engine.validate_clean_record(clean_record)
        clean_validations.append(clean_qc)
        clean_records.append(clean_record)
        if not clean_qc["is_valid"]:
            logger.warning(f"  > 车系 {series_id} 清洗数据未通过校验: {clean_qc['issues']}")

        governance_result = governance_orchestrator.govern(
            clean_record=clean_record,
            markdown_text=markdown_text,
            batch_id=batch_id,
        )
        governance_counters[governance_result.decision_result.decision] += 1
        logger.info(
            f"  > 治理决策: {governance_result.decision_result.decision}, "
            f"quality={governance_result.quality_result.quality_score:.2f}, "
            f"tier={governance_result.quality_result.quality_tier}, "
            f"rag={governance_result.quality_result.rag_readiness:.2f}, "
            f"reason={governance_result.decision_result.reason}"
        )

        clean_json_path = os.path.join(cleaned_json_dir, f"series_{series_id}.json")
        with open(clean_json_path, "w", encoding="utf-8") as f:
            json.dump(clean_record, f, ensure_ascii=False, indent=2)

        clean_md_path = os.path.join(cleaned_md_dir, f"series_{series_id}.md")
        with open(clean_md_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        # 2.3 大模型数据处理：Chunking 分块
        total_tokens = processor.count_tokens(markdown_text)
        if governance_result.decision_result.decision in {"accept", "repair"}:
            kept_count, dropped_count, chunk_count = append_training_items(
                clean_record=clean_record,
                series_id=str(series_id),
                crawl_time=datetime.now().isoformat(),
                governance_result=governance_result,
            )
        else:
            kept_count, dropped_count, chunk_count = 0, 0, 0
        logger.info(
            f"  > 清洗后长度: {len(markdown_text)} 字符，约 {total_tokens} tokens，"
            f"切分 {chunk_count} 块，保留 {kept_count}，过滤 {dropped_count}"
        )

        elapsed = time.monotonic() - started_at
        state_store.mark_success(
            series_id,
            meta={
                "url": fetch_meta.get("url"),
                "http_status": fetch_meta.get("http_status"),
                "retry_count": fetch_meta.get("retry_count", 0),
                "model_count": stats.get("model_count", 0),
                "news_count": stats.get("news_count", 0),
                "raw_image_count": fetch_meta.get("image_count", 0),
                "raw_image_success_count": fetch_meta.get("image_success_count", 0),
                "raw_image_failed_count": fetch_meta.get("image_failed_count", 0),
                "raw_image_manifest_path": fetch_meta.get("image_manifest_path", ""),
                "raw_image_dir": fetch_meta.get("image_dir", ""),
                "chunks": chunk_count,
                "records_kept": kept_count,
                "governance_decision": governance_result.decision_result.decision,
                "governance_quality_score": governance_result.quality_result.quality_score,
                "governance_quality_tier": governance_result.quality_result.quality_tier,
                "governance_rag_readiness": governance_result.quality_result.rag_readiness,
                "governance_training_readiness": governance_result.quality_result.training_readiness,
                "governance_duration_ms": governance_result.metadata.get("governance_duration_ms", 0.0),
                "governance_input_summary": governance_result.metadata.get("input_summary", {}),
            },
        )
        stats_collector.add_success(
            duration_sec=elapsed,
            retries=fetch_meta.get("retry_count", 0),
            http_status=fetch_meta.get("http_status"),
        )

        time.sleep(1)  # 礼貌延迟

    # 3. 输出 JSONL 到 processed 目录
    output_path = os.path.join("data", "processed", "dongchedi_training_data.jsonl")
    processor.format_as_jsonl(output_path, structured_data)
    training_parquet_path = os.path.join("data", "processed", "dongchedi_training_data.parquet")
    parquet_exporter.export_training_items(structured_data, training_parquet_path)
    cleaned_parquet_path = os.path.join("data", "cleaned", "dongchedi", "cleaned_series.parquet")
    parquet_exporter.export_clean_summary(clean_records, cleaned_parquet_path)

    # 4. 输出质量报告 (JSON + Markdown)
    report = quality_engine.generate_quality_report(
        clean_validations=clean_validations,
        clean_records=clean_records,
        training_items=structured_data,
        dropped_training_items=dropped_training_items,
        output_dir=report_dir,
    )
    task_summary = stats_collector.export(output_dir=report_dir)
    if len(structured_data) < target_training_records:
        logger.warning(
            f"当前仅生成 {len(structured_data)} 条训练样本，未达到目标 {target_training_records}。"
            "可增大 TARGET_SERIES_POOL 或减小 MAX_CHUNK_TOKENS 后重跑。"
        )

    logger.info(
        "✅ 懂车帝流水线执行完毕！"
        f"共生成 {len(structured_data)} 条大模型训练数据，"
        f"丢弃 {dropped_training_items} 条低质量样本，"
        f"治理决策 accept/repair/review/drop = "
        f"{governance_counters['accept']}/{governance_counters['repair']}/"
        f"{governance_counters['review']}/{governance_counters['drop']}，"
        f"有效车系 {report['summary']['series_valid']}/{report['summary']['series_total']}，"
        f"任务成功/失败/跳过 = {task_summary['success_count']}/{task_summary['failed_count']}/{task_summary['skipped_count']}。"
    )

if __name__ == "__main__":
    main()
