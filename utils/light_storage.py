import json
import os
from datetime import datetime


class LightRAGStorage:
    """
    轻量版存储层（DuckDB）：
    - chunks: 用于回溯 chunk 元数据
    - build_runs: 记录索引构建任务
    - query_logs: 记录检索与回答日志
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        try:
            import duckdb  # type: ignore
        except ImportError as e:
            raise RuntimeError("缺少 duckdb 依赖，请安装 requirements 后重试。") from e
        self._duckdb = duckdb
        self._conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                record_id BIGINT,
                source VARCHAR,
                series_id VARCHAR,
                title VARCHAR,
                url VARCHAR,
                chunk_index INTEGER,
                total_chunks INTEGER,
                tokens INTEGER,
                text VARCHAR,
                metadata_json VARCHAR,
                loaded_at TIMESTAMP
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS build_runs (
                run_id VARCHAR,
                built_at TIMESTAMP,
                records_count BIGINT,
                index_path VARCHAR,
                records_path VARCHAR,
                meta_path VARCHAR,
                embedding_provider VARCHAR,
                embedding_model VARCHAR
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_logs (
                query_id VARCHAR,
                asked_at TIMESTAMP,
                question VARCHAR,
                retrieval_mode VARCHAR,
                top_k INTEGER,
                llm_provider VARCHAR,
                llm_model VARCHAR,
                answer_text VARCHAR,
                hits_json VARCHAR
            )
            """
        )

    def refresh_chunks(self, records: list[dict]) -> None:
        now = datetime.now()
        self._conn.execute("DELETE FROM chunks")
        for idx, record in enumerate(records):
            metadata = record.get("metadata", {}) or {}
            self._conn.execute(
                """
                INSERT INTO chunks (
                    record_id, source, series_id, title, url,
                    chunk_index, total_chunks, tokens, text, metadata_json, loaded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    idx,
                    str(metadata.get("source", "dongchedi")),
                    str(metadata.get("series_id", "")),
                    str(metadata.get("title", "")),
                    str(metadata.get("url", "")),
                    int(metadata.get("chunk_index", -1)) if metadata.get("chunk_index") is not None else -1,
                    int(metadata.get("total_chunks", 0)) if metadata.get("total_chunks") is not None else 0,
                    int(metadata.get("tokens", 0)) if metadata.get("tokens") is not None else 0,
                    record.get("text", ""),
                    json.dumps(metadata, ensure_ascii=False),
                    now,
                ],
            )

    def log_build_run(
        self,
        run_id: str,
        records_count: int,
        index_path: str,
        records_path: str,
        meta_path: str,
        embedding_provider: str,
        embedding_model: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO build_runs (
                run_id, built_at, records_count, index_path, records_path, meta_path, embedding_provider, embedding_model
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                datetime.now(),
                int(records_count),
                index_path,
                records_path,
                meta_path,
                embedding_provider,
                embedding_model,
            ],
        )

    def log_query(
        self,
        query_id: str,
        question: str,
        retrieval_mode: str,
        top_k: int,
        llm_provider: str,
        llm_model: str,
        answer_text: str,
        hits: list[dict],
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO query_logs (
                query_id, asked_at, question, retrieval_mode, top_k, llm_provider, llm_model, answer_text, hits_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                query_id,
                datetime.now(),
                question,
                retrieval_mode,
                int(top_k),
                llm_provider,
                llm_model,
                answer_text,
                json.dumps(hits, ensure_ascii=False),
            ],
        )
