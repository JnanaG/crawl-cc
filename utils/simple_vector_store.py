import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class SearchHit:
    score: float
    text: str
    metadata: dict


class SimpleVectorStore:
    """
    轻量本地向量库:
    - 向量存储: .npz
    - 文本/元数据存储: .jsonl
    - 检索: 余弦相似度(top-k)
    """

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.embeddings = None
        self.records = []

    def _hash_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        text = (text or "").strip()
        if not text:
            return vec

        for token in text:
            h = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % self.dim
            sign = -1.0 if (int(h[8:10], 16) % 2) else 1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def build(self, items: list[dict], progress_callback=None) -> None:
        records = []
        vectors = []
        total = len(items)
        start_time = time.perf_counter()
        for idx, item in enumerate(items, start=1):
            text = item.get("text", "")
            metadata = item.get("metadata", {})
            if not text.strip():
                continue
            emb = self._hash_embed(text)
            vectors.append(emb)
            records.append({"text": text, "metadata": metadata})
            if progress_callback:
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                rate = idx / elapsed
                remain = max(total - idx, 0)
                eta_sec = int(remain / max(rate, 1e-6))
                progress_callback(idx, total, elapsed, eta_sec)

        self.records = records
        if vectors:
            self.embeddings = np.vstack(vectors).astype(np.float32)
        else:
            self.embeddings = np.zeros((0, self.dim), dtype=np.float32)

    def save(self, index_path: str, records_path: str) -> None:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(records_path), exist_ok=True)
        np.savez_compressed(index_path, embeddings=self.embeddings, dim=self.dim)
        with open(records_path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def load(self, index_path: str, records_path: str) -> None:
        npz = np.load(index_path)
        self.dim = int(npz["dim"])
        self.embeddings = npz["embeddings"].astype(np.float32)
        self.records = []
        with open(records_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if self.embeddings is None or len(self.records) == 0:
            return []
        query_vec = self._hash_embed(query)
        scores = self.embeddings @ query_vec
        if scores.size == 0:
            return []
        idxs = np.argsort(-scores)[:top_k]
        hits = []
        for idx in idxs:
            record = self.records[int(idx)]
            hits.append(
                SearchHit(
                    score=float(scores[int(idx)]),
                    text=record.get("text", ""),
                    metadata=record.get("metadata", {}),
                )
            )
        return hits
