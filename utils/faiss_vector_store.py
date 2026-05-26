import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class SearchHit:
    score: float
    text: str
    metadata: dict
    record_id: int = -1


class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = None
        self.records = []

    def _require_faiss(self):
        try:
            import faiss  # type: ignore
        except ImportError as e:
            raise RuntimeError("缺少 faiss-cpu 依赖，请安装 requirements 后重试。") from e
        return faiss

    def build(self, vectors: list[list[float]], records: list[dict]) -> None:
        faiss = self._require_faiss()
        if len(vectors) != len(records):
            raise ValueError("vectors 与 records 数量不一致")

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors 必须是二维数组")
        if matrix.shape[0] == 0:
            self.index = faiss.IndexFlatIP(self.dim)
            self.records = []
            return
        if matrix.shape[1] != self.dim:
            raise ValueError(f"向量维度不匹配: expected={self.dim}, got={matrix.shape[1]}")

        # 归一化后使用内积等价于余弦相似度
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(self.dim)
        index.add(matrix)
        self.index = index
        self.records = records

    def save(self, index_path: str, records_path: str, meta_path: str) -> None:
        faiss = self._require_faiss()
        if self.index is None:
            raise ValueError("索引为空，请先 build 或 load")

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(records_path), exist_ok=True)
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(records_path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "size": len(self.records)}, f, ensure_ascii=False, indent=2)

    def load(self, index_path: str, records_path: str, meta_path: str) -> None:
        faiss = self._require_faiss()
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.dim = int(meta["dim"])
        self.index = faiss.read_index(index_path)

        self.records = []
        with open(records_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchHit]:
        if self.index is None or len(self.records) == 0:
            return []
        if not query_vector:
            return []

        q = np.asarray([query_vector], dtype=np.float32)
        faiss = self._require_faiss()
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, top_k)
        hits = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.records):
                continue
            record = self.records[int(idx)]
            hits.append(
                SearchHit(
                    score=float(score),
                    text=record.get("text", ""),
                    metadata=record.get("metadata", {}),
                    record_id=int(idx),
                )
            )
        return hits
