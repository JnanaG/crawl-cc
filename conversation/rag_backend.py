from __future__ import annotations

import json
import logging
import time
from typing import Any, Protocol

from rag_llm_demo import build_context, hybrid_search, iter_unique_series
from utils.embedding_client import EmbeddingClient
from utils.faiss_vector_store import FaissVectorStore
from utils.llm_client import LLMClient
from utils.simple_vector_store import SimpleVectorStore

logger = logging.getLogger(__name__)


DEFAULT_INDEX = "data/vector_store/faiss/dongchedi.index"
DEFAULT_RECORDS = "data/vector_store/faiss/dongchedi_records.jsonl"
DEFAULT_META = "data/vector_store/faiss/dongchedi_meta.json"


class RAGBackend(Protocol):
    def ask(self, question: str) -> dict[str, Any]:
        ...


def create_rag_backend(
    *,
    backend_mode: str = "hash",
    index_path: str = DEFAULT_INDEX,
    records_path: str = DEFAULT_RECORDS,
    meta_path: str = DEFAULT_META,
    embedding_provider: str = "fastembed",
    embedding_model: str | None = None,
    embedding_api_base: str | None = None,
    embedding_api_key: str | None = None,
    llm_provider: str = "ollama",
    llm_model: str | None = None,
    llm_api_base: str | None = None,
    llm_api_key: str | None = None,
    retrieval_mode: str = "hybrid",
    top_k: int = 6,
    use_llm: bool = False,
) -> RAGBackend:
    logger.info(
        "rag_backend_create_started backend_mode=%s embedding_provider=%s retrieval_mode=%s top_k=%s use_llm=%s llm_provider=%s llm_model=%s",
        backend_mode,
        embedding_provider,
        retrieval_mode,
        top_k,
        use_llm,
        llm_provider,
        llm_model or "",
    )
    if backend_mode == "faiss":
        return FaissRAGBackend(
            index_path=index_path,
            records_path=records_path,
            meta_path=meta_path,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_api_base=embedding_api_base,
            embedding_api_key=embedding_api_key,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_base=llm_api_base,
            llm_api_key=llm_api_key,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            use_llm=use_llm,
        )
    return HashRAGBackend(
        records_path=records_path,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_base=llm_api_base,
        llm_api_key=llm_api_key,
        top_k=top_k,
        use_llm=use_llm,
    )


def _load_jsonl_records(records_path: str) -> list[dict[str, Any]]:
    rows = []
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_fallback_answer(question: str, hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "根据当前检索内容无法确认。"
    lines = [f"基于检索内容，和“{question}”最相关的信息如下："]
    for idx, hit in enumerate(hits[:3], start=1):
        title = hit.get("title") or "未知车系"
        snippet = hit.get("text_snippet") or ""
        lines.append(f"{idx}. {title}: {snippet[:120].strip()}")
    lines.append("如果你愿意，我可以继续围绕这几款车做价格、配置或对比分析。")
    return "\n".join(lines)


class FaissRAGBackend:
    def __init__(
        self,
        *,
        index_path: str = DEFAULT_INDEX,
        records_path: str = DEFAULT_RECORDS,
        meta_path: str = DEFAULT_META,
        embedding_provider: str = "fastembed",
        embedding_model: str | None = None,
        embedding_api_base: str | None = None,
        embedding_api_key: str | None = None,
        llm_provider: str = "ollama",
        llm_model: str | None = None,
        llm_api_base: str | None = None,
        llm_api_key: str | None = None,
        retrieval_mode: str = "hybrid",
        top_k: int = 6,
        max_context_chars: int = 2800,
        use_llm: bool = False,
    ):
        self.index_path = index_path
        self.records_path = records_path
        self.meta_path = meta_path
        self.backend_mode = "faiss"
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.use_llm = use_llm
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model or ""
        self.llm_provider = llm_provider
        self.llm_model = llm_model or ""
        logger.info(
            "rag_backend_loading_embeddings backend_mode=%s provider=%s model=%s",
            self.backend_mode,
            self.embedding_provider,
            self.embedding_model,
        )
        self.embedding_client = EmbeddingClient(
            provider=embedding_provider,
            model=embedding_model,
            api_base=embedding_api_base,
            api_key=embedding_api_key,
        )
        self.store = FaissVectorStore(dim=1)
        logger.info(
            "rag_backend_loading_vector_store backend_mode=%s index_path=%s records_path=%s meta_path=%s",
            self.backend_mode,
            index_path,
            records_path,
            meta_path,
        )
        self.store.load(index_path=index_path, records_path=records_path, meta_path=meta_path)
        self.llm_client = None
        if use_llm:
            self.llm_client = LLMClient(
                provider=llm_provider,
                model=llm_model,
                api_base=llm_api_base,
                api_key=llm_api_key,
            )
        logger.info(
            "rag_backend_initialized runtime=%s",
            self.describe_runtime(),
        )

    def ask(self, question: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        runtime = self.describe_runtime()
        logger.info(
            "rag_backend_ask_started question=%r runtime=%s",
            question[:120],
            runtime,
        )
        query_vec = self.embedding_client.embed_query(question)
        if self.retrieval_mode == "dense":
            raw_hits = self.store.search(query_vector=query_vec, top_k=self.top_k)
        else:
            raw_hits, _ = hybrid_search(
                question=question,
                query_vec=query_vec,
                store=self.store,
                top_k=self.top_k,
                dense_top_k=max(24, self.top_k),
                sparse_top_k=max(40, self.top_k),
                rrf_k=60,
                rerank_top_n=max(12, self.top_k),
                reranker=None,
                reranker_weight=0.0,
            )
        hits = [
            {
                "title": hit.metadata.get("title", ""),
                "url": hit.metadata.get("url", ""),
                "series_id": str(hit.metadata.get("series_id", "")),
                "score": float(hit.score),
                "text_snippet": hit.text[:240],
            }
            for hit in raw_hits
        ]
        logger.info(
            "rag_backend_retrieval_finished hits=%s runtime=%s",
            len(hits),
            runtime,
        )
        if not raw_hits:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info(
                "rag_backend_no_hits latency_ms=%s runtime=%s",
                latency_ms,
                runtime,
            )
            return {
                "answer": "根据当前检索内容无法确认。",
                "hits": [],
                "memory_context": "",
            }

        if not self.llm_client:
            answer = build_fallback_answer(question, hits)
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info(
                "rag_backend_fallback_answer hits=%s latency_ms=%s runtime=%s",
                len(hits),
                latency_ms,
                runtime,
            )
            return {"answer": answer, "hits": hits, "memory_context": ""}

        context = build_context(raw_hits, self.max_context_chars)
        system_prompt = (
            "你是汽车多轮对话助手。必须仅基于给定上下文回答，不要编造。"
            "如果上下文不足，请明确说“根据当前检索内容无法确认”。"
        )
        user_prompt = f"用户问题:\n{question}\n\n检索上下文:\n{context}\n\n请给出简洁准确的中文回答。"
        logger.info(
            "rag_backend_llm_started hits=%s context_chars=%s runtime=%s",
            len(hits),
            len(context),
            runtime,
        )
        answer = self.llm_client.chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2)
        sources = []
        for hit in iter_unique_series(raw_hits, max_series=4):
            sources.append(f"{hit.metadata.get('title', '未知车系')}({hit.metadata.get('url', 'N/A')})")
        if sources:
            answer = f"{answer.strip()}\n\n参考来源: " + "; ".join(sources)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "rag_backend_llm_finished hits=%s latency_ms=%s runtime=%s answer_preview=%r",
            len(hits),
            latency_ms,
            runtime,
            answer[:120],
        )
        return {"answer": answer, "hits": hits, "memory_context": context}

    def describe_runtime(self) -> dict[str, Any]:
        return {
            "backend_mode": self.backend_mode,
            "retrieval_mode": self.retrieval_mode,
            "top_k": self.top_k,
            "use_llm": bool(self.llm_client),
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
        }


class HashRAGBackend:
    def __init__(
        self,
        *,
        records_path: str = DEFAULT_RECORDS,
        llm_provider: str = "ollama",
        llm_model: str | None = None,
        llm_api_base: str | None = None,
        llm_api_key: str | None = None,
        top_k: int = 6,
        use_llm: bool = False,
    ):
        self.top_k = top_k
        self.records_path = records_path
        self.backend_mode = "hash"
        self.retrieval_mode = "hash"
        self.use_llm = use_llm
        self.embedding_provider = "hash"
        self.embedding_model = ""
        self.llm_provider = llm_provider
        self.llm_model = llm_model or ""
        self.store = SimpleVectorStore(dim=384)
        rows = _load_jsonl_records(records_path)
        self.store.build(rows)
        self.llm_client = None
        if use_llm:
            self.llm_client = LLMClient(
                provider=llm_provider,
                model=llm_model,
                api_base=llm_api_base,
                api_key=llm_api_key,
            )
        logger.info(
            "rag_backend_initialized runtime=%s",
            self.describe_runtime(),
        )

    def ask(self, question: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        runtime = self.describe_runtime()
        logger.info(
            "rag_backend_ask_started question=%r runtime=%s",
            question[:120],
            runtime,
        )
        raw_hits = self.store.search(question, top_k=self.top_k)
        hits = [
            {
                "title": hit.metadata.get("title", ""),
                "url": hit.metadata.get("url", ""),
                "series_id": str(hit.metadata.get("series_id", "")),
                "score": float(hit.score),
                "text_snippet": hit.text[:240],
            }
            for hit in raw_hits
        ]
        logger.info(
            "rag_backend_retrieval_finished hits=%s runtime=%s",
            len(hits),
            runtime,
        )
        if not hits:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info(
                "rag_backend_no_hits latency_ms=%s runtime=%s",
                latency_ms,
                runtime,
            )
            return {"answer": "根据当前检索内容无法确认。", "hits": [], "memory_context": ""}

        if not self.llm_client:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info(
                "rag_backend_fallback_answer hits=%s latency_ms=%s runtime=%s",
                len(hits),
                latency_ms,
                runtime,
            )
            return {
                "answer": build_fallback_answer(question, hits),
                "hits": hits,
                "memory_context": "",
            }

        context_lines = []
        for idx, hit in enumerate(hits[:4], start=1):
            context_lines.append(
                f"[{idx}] title={hit['title']}; score={hit['score']:.4f}; url={hit['url']}\n{hit['text_snippet']}"
            )
        context = "\n\n".join(context_lines)
        system_prompt = (
            "你是汽车多轮对话助手。必须仅基于给定上下文回答，不要编造。"
            "如果上下文不足，请明确说“根据当前检索内容无法确认”。"
        )
        user_prompt = f"用户问题:\n{question}\n\n检索上下文:\n{context}\n\n请给出简洁准确的中文回答。"
        logger.info(
            "rag_backend_llm_started hits=%s context_chars=%s runtime=%s",
            len(hits),
            len(context),
            runtime,
        )
        answer = self.llm_client.chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "rag_backend_llm_finished hits=%s latency_ms=%s runtime=%s answer_preview=%r",
            len(hits),
            latency_ms,
            runtime,
            answer[:120],
        )
        return {"answer": answer.strip(), "hits": hits, "memory_context": context}

    def describe_runtime(self) -> dict[str, Any]:
        return {
            "backend_mode": self.backend_mode,
            "retrieval_mode": self.retrieval_mode,
            "top_k": self.top_k,
            "use_llm": bool(self.llm_client),
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
        }
