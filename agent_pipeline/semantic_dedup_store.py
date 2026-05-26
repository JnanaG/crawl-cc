from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from utils.embedding_client import EmbeddingClient
from utils.faiss_vector_store import FaissVectorStore


class SemanticDedupStore:
    def __init__(
        self,
        base_dir: str = "data",
        provider: str | None = None,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        similarity_threshold: float = 0.94,
        same_series_threshold: float = 0.88,
        top_k: int = 3,
    ):
        self.base_dir = base_dir
        self.store_dir = os.path.join(base_dir, "state", "agent_pipeline", "semantic_dedup")
        os.makedirs(self.store_dir, exist_ok=True)

        self.provider = provider or os.getenv("AGENT_EMBEDDING_PROVIDER") or "fastembed"
        self.model = model or os.getenv("AGENT_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL")
        self.api_base = api_base or os.getenv("AGENT_EMBEDDING_API_BASE")
        self.api_key = api_key or os.getenv("AGENT_EMBEDDING_API_KEY")
        self.similarity_threshold = similarity_threshold
        self.same_series_threshold = same_series_threshold
        self.top_k = top_k

        self.index_path = os.path.join(self.store_dir, "semantic.index")
        self.records_path = os.path.join(self.store_dir, "semantic_records.jsonl")
        self.meta_path = os.path.join(self.store_dir, "semantic_meta.json")
        self.vectors_path = os.path.join(self.store_dir, "semantic_vectors.npy")

        self.embedding_client: EmbeddingClient | None = None
        self.store: FaissVectorStore | None = None
        self.vectors: list[list[float]] = []
        self.records: list[dict[str, Any]] = []
        self._load_existing_store()

    def _load_existing_store(self) -> None:
        if not (
            os.path.exists(self.index_path)
            and os.path.exists(self.records_path)
            and os.path.exists(self.meta_path)
            and os.path.exists(self.vectors_path)
        ):
            return

        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        dim = int(meta["dim"])
        self.store = FaissVectorStore(dim=dim)
        self.store.load(self.index_path, self.records_path, self.meta_path)
        self.records = list(self.store.records)

        matrix = np.load(self.vectors_path)
        if matrix.ndim == 1:
            matrix = np.expand_dims(matrix, axis=0)
        self.vectors = matrix.astype(np.float32).tolist()

    def _get_embedding_client(self) -> EmbeddingClient:
        if self.embedding_client is None:
            self.embedding_client = EmbeddingClient(
                provider=self.provider,
                model=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
            )
        return self.embedding_client

    def _compose_semantic_text(self, clean_record: dict[str, Any], markdown_text: str) -> str:
        series = clean_record.get("series", {}) or {}
        prefix = " ".join(
            [
                str(series.get("brand_name") or "").strip(),
                str(series.get("series_name") or "").strip(),
                str(series.get("car_type") or "").strip(),
            ]
        ).strip()
        body = (markdown_text or "").strip()
        return f"{prefix}\n{body}".strip()

    def _embed(self, text: str) -> list[float]:
        client = self._get_embedding_client()
        vector = client.embed_query(text)
        if not vector:
            raise RuntimeError("semantic dedup embedding 返回空向量")
        return vector

    def find_candidates(self, clean_record: dict[str, Any], markdown_text: str) -> dict[str, Any]:
        text = self._compose_semantic_text(clean_record, markdown_text)
        if not text:
            return {"query_text": "", "query_dim": 0, "hits": []}

        query_vector = self._embed(text)
        if self.store is None or not self.records:
            return {"query_text": text, "query_dim": len(query_vector), "hits": []}

        hits = self.store.search(query_vector=query_vector, top_k=self.top_k)
        hit_rows = []
        for hit in hits:
            hit_rows.append(
                {
                    "score": float(hit.score),
                    "series_id": hit.metadata.get("series_id"),
                    "series_name": hit.metadata.get("series_name"),
                    "content_hash": hit.metadata.get("content_hash"),
                    "normalized_hash": hit.metadata.get("normalized_hash"),
                    "record_hash": hit.metadata.get("record_hash"),
                }
            )

        return {
            "query_text": text,
            "query_dim": len(query_vector),
            "hits": hit_rows,
        }

    def add_record(self, clean_record: dict[str, Any], markdown_text: str, metadata: dict[str, Any]) -> None:
        text = self._compose_semantic_text(clean_record, markdown_text)
        if not text:
            return

        vector = self._embed(text)
        dim = len(vector)
        if self.store is None:
            self.store = FaissVectorStore(dim=dim)
        elif self.store.dim != dim:
            raise ValueError(f"semantic dedup 向量维度不匹配: store={self.store.dim}, current={dim}")

        series = clean_record.get("series", {}) or {}
        record = {
            "text": text,
            "metadata": {
                "series_id": str(series.get("series_id") or ""),
                "series_name": str(series.get("series_name") or ""),
                "brand_name": str(series.get("brand_name") or ""),
                "content_hash": metadata.get("content_hash"),
                "normalized_hash": metadata.get("normalized_hash"),
                "record_hash": metadata.get("record_hash"),
            },
        }
        self.records.append(record)
        self.vectors.append(vector)
        self.store.build(self.vectors, self.records)
        self._persist()

    def _persist(self) -> None:
        if self.store is None:
            return
        self.store.save(self.index_path, self.records_path, self.meta_path)
        matrix = np.asarray(self.vectors, dtype=np.float32)
        np.save(self.vectors_path, matrix)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta.update(
            {
                "provider": self.provider,
                "model": self.model,
                "similarity_threshold": self.similarity_threshold,
                "same_series_threshold": self.same_series_threshold,
                "top_k": self.top_k,
            }
        )
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

