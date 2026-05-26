import os
from typing import Sequence

import requests


class EmbeddingClient:
    """
    真实 Embedding 客户端，支持:
    1) sentence_transformers (本地模型)
    2) fastembed (本地ONNX模型, 无torch依赖)
    3) openai_compatible (远端API)
    4) ollama (本地/远端Ollama Embeddings API)
    """

    def __init__(
        self,
        provider: str = "sentence_transformers",
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout_sec: int = 60,
    ):
        self.provider = provider
        self.timeout_sec = timeout_sec

        if provider == "sentence_transformers":
            self.model_name = model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise RuntimeError(
                    f"加载 sentence-transformers 失败: {e}"
                ) from e
            self._model = SentenceTransformer(self.model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()
            self.api_base = None
            self.api_key = None
            return

        if provider == "fastembed":
            self.model_name = model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
            try:
                from fastembed import TextEmbedding
            except Exception as e:
                raise RuntimeError(f"加载 fastembed 失败: {e}") from e
            self._model = TextEmbedding(model_name=self.model_name)
            self.dimension = None
            self.api_base = None
            self.api_key = None
            return

        if provider == "openai_compatible":
            self.model_name = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            self.api_base = (api_base or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
            if not self.api_base:
                raise ValueError("openai_compatible 需要 OPENAI_BASE_URL 或 --embedding-api-base")
            if not self.api_key:
                raise ValueError("openai_compatible 需要 OPENAI_API_KEY 或 --embedding-api-key")
            self.dimension = None
            self._model = None
            return

        if provider == "ollama":
            self.model_name = model or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
            self.api_base = (api_base or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
            self.api_key = ""
            self.dimension = None
            self._model = None
            return

        raise ValueError(f"不支持的 embedding provider: {provider}")

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if self.provider == "sentence_transformers":
            vectors = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
            return vectors.tolist()

        if self.provider == "fastembed":
            vectors = list(self._model.embed(list(texts)))
            return [v.tolist() for v in vectors]

        if self.provider == "ollama":
            vectors = []
            endpoints = [
                (f"{self.api_base}/api/embed", "input", "embeddings"),
                (f"{self.api_base}/api/embeddings", "prompt", "embedding"),
            ]
            for text in texts:
                vector = None
                last_error = None
                for url, input_key, output_key in endpoints:
                    payload = {"model": self.model_name, input_key: text}
                    try:
                        resp = requests.post(url, json=payload, timeout=self.timeout_sec)
                        # 端点不存在时返回404，尝试下一个兼容端点
                        if resp.status_code == 404:
                            last_error = resp.text
                            continue
                        resp.raise_for_status()
                        body = resp.json()
                        if output_key == "embeddings":
                            values = body.get("embeddings", [])
                            if values and isinstance(values[0], list):
                                vector = values[0]
                            elif isinstance(values, list):
                                vector = values
                        else:
                            vector = body.get("embedding", [])
                        if vector:
                            break
                    except requests.RequestException as e:
                        last_error = str(e)
                if not vector:
                    raise RuntimeError(
                        "调用 Ollama Embedding 失败，请检查 Ollama 版本、模型名称和服务状态。"
                        f" model={self.model_name}, base={self.api_base}, error={last_error}"
                    )
                vectors.append(vector)
            return vectors

        url = f"{self.api_base}/v1/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model_name, "input": list(texts)}
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        body = resp.json()
        data = body.get("data", [])
        if not data:
            return []
        ordered = sorted(data, key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in ordered]

    def embed_query(self, query: str) -> list[float]:
        vectors = self.embed_texts([query])
        return vectors[0] if vectors else []
