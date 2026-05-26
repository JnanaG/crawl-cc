import json
import os
import re
import time
import uuid
from collections import Counter
from typing import Iterable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from loguru import logger

from utils.embedding_client import EmbeddingClient
from utils.faiss_vector_store import SearchHit
from utils.light_storage import LightRAGStorage
from utils.llm_client import LLMClient
from utils.reranker_client import RerankerClient


def load_training_items(jsonl_path: str) -> list[dict]:
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_records(records_path: str) -> list[dict]:
    records = []
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def maybe_init_storage(args) -> LightRAGStorage | None:
    if getattr(args, "disable_storage", False):
        return None
    try:
        return LightRAGStorage(args.storage_db)
    except Exception as e:
        logger.warning(f"初始化轻量存储层失败，已跳过存储写入: {e}")
        return None


def tokenize_zh_en(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower())


class ProjectEmbeddings(Embeddings):
    def __init__(self, provider: str, model: str | None, api_base: str | None, api_key: str | None, batch_size: int):
        self.client = EmbeddingClient(provider=provider, model=model, api_base=api_base, api_key=api_key)
        self.batch_size = max(1, batch_size)
        self.model_name = self.client.model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        total = len(texts)
        total_batches = (total + self.batch_size - 1) // max(1, self.batch_size)
        start_time = time.perf_counter()
        for batch_idx, start in enumerate(range(0, total, self.batch_size), start=1):
            vectors.extend(self.client.embed_texts(texts[start : start + self.batch_size]))
            done = min(start + self.batch_size, total)
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            rate = done / elapsed
            remain = max(total - done, 0)
            eta_sec = int(remain / max(rate, 1e-6))
            progress = (done / max(total, 1)) * 100
            logger.info(
                f"[build进度] {done}/{total} ({progress:.1f}%) | "
                f"batch={batch_idx}/{total_batches} | elapsed={elapsed:.1f}s | eta={eta_sec}s"
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed_query(text)


def normalize_metadata(metadata: dict | None) -> dict:
    metadata = metadata or {}
    return {
        "source": metadata.get("source", "dongchedi"),
        "series_id": str(metadata.get("series_id", "")),
        "title": metadata.get("title", ""),
        "url": metadata.get("url", ""),
        "brand_name": metadata.get("brand_name", ""),
        "car_type": metadata.get("car_type", ""),
        "chunk_index": metadata.get("chunk_index", -1),
        "total_chunks": metadata.get("total_chunks", 0),
        "tokens": metadata.get("tokens", 0),
    }


def records_to_documents(records: list[dict]) -> list[Document]:
    docs = []
    for r in records:
        text = (r.get("text") or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata=normalize_metadata(r.get("metadata", {}))))
    return docs


def training_items_to_records(items: list[dict]) -> list[dict]:
    records = []
    for item in items:
        text = (item.get("text") or "").strip()
        if text:
            records.append({"text": text, "metadata": normalize_metadata(item.get("metadata", {}))})
    return records


def save_records(records_path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(records_path), exist_ok=True)
    with open(records_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_meta(meta_path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _doc_key(text: str, metadata: dict) -> str:
    return json.dumps({"text": text, "metadata": metadata or {}}, ensure_ascii=False, sort_keys=True)


def build_doc_id_map(documents: list[Document]) -> dict[str, int]:
    return {_doc_key(doc.page_content, doc.metadata or {}): idx for idx, doc in enumerate(documents)}


class LightweightBM25:
    def __init__(self, records: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tf = []
        self.doc_len = []
        self.idf = {}
        df = Counter()
        for record in records:
            tf = Counter(tokenize_zh_en(record.get("text", "")))
            self.doc_tf.append(tf)
            self.doc_len.append(sum(tf.values()))
            for t in tf:
                df[t] += 1
        n_docs = max(1, len(records))
        self.avgdl = sum(self.doc_len) / n_docs if self.doc_len else 0.0
        for t, freq in df.items():
            self.idf[t] = max(0.0, float((n_docs - freq + 0.5) / (freq + 0.5)))

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        q_terms = tokenize_zh_en(query)
        if not q_terms:
            return []
        scores = []
        for idx, tf in enumerate(self.doc_tf):
            dl = self.doc_len[idx]
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f <= 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-6)))
                score += idf * ((f * (self.k1 + 1.0)) / max(denom, 1e-6))
            if score > 0:
                scores.append((idx, float(score)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def _minmax_norm(v: float, min_v: float, max_v: float) -> float:
    if max_v - min_v <= 1e-8:
        return 0.0
    return (v - min_v) / (max_v - min_v)


def _load_vector_store(store_dir: str, embeddings: ProjectEmbeddings, index_name: str) -> FAISS:
    try:
        return FAISS.load_local(
            folder_path=store_dir,
            embeddings=embeddings,
            index_name=index_name,
            allow_dangerous_deserialization=True,
        )
    except TypeError:
        return FAISS.load_local(folder_path=store_dir, embeddings=embeddings, index_name=index_name)


def dense_search(question: str, store: FAISS, doc_id_map: dict[str, int], top_k: int) -> list[SearchHit]:
    dense_pairs = store.similarity_search_with_score(question, k=top_k)
    hits = []
    for doc, raw_score in dense_pairs:
        metadata = doc.metadata or {}
        hits.append(
            SearchHit(
                score=-float(raw_score),
                text=doc.page_content,
                metadata=metadata,
                record_id=doc_id_map.get(_doc_key(doc.page_content, metadata), -1),
            )
        )
    return hits


def hybrid_search(
    question: str,
    store: FAISS,
    documents: list[Document],
    doc_id_map: dict[str, int],
    top_k: int,
    dense_top_k: int,
    sparse_top_k: int,
    rrf_k: int,
    rerank_top_n: int,
    reranker: RerankerClient | None,
    reranker_weight: float,
) -> tuple[list[SearchHit], list[dict]]:
    dense_hits = dense_search(question, store, doc_id_map, max(dense_top_k, top_k))
    records = [{"text": d.page_content, "metadata": d.metadata or {}} for d in documents]
    sparse_hits = LightweightBM25(records).search(question, top_k=max(sparse_top_k, top_k))

    dense_rank, dense_score = {}, {}
    for rank, h in enumerate(dense_hits, start=1):
        if h.record_id >= 0:
            dense_rank[h.record_id] = rank
            dense_score[h.record_id] = h.score
    sparse_rank, sparse_score = {}, {}
    for rank, (doc_id, score) in enumerate(sparse_hits, start=1):
        sparse_rank[doc_id] = rank
        sparse_score[doc_id] = score

    candidate_ids = set(dense_rank) | set(sparse_rank)
    if not candidate_ids:
        return [], []

    dense_vals = list(dense_score.values()) or [0.0]
    sparse_vals = list(sparse_score.values()) or [0.0]
    min_dense, max_dense = min(dense_vals), max(dense_vals)
    min_sparse, max_sparse = min(sparse_vals), max(sparse_vals)
    q_tokens = set(tokenize_zh_en(question))

    ranked = []
    for doc_id in candidate_ids:
        d_rank = dense_rank.get(doc_id)
        s_rank = sparse_rank.get(doc_id)
        d_score = dense_score.get(doc_id, 0.0)
        s_score = sparse_score.get(doc_id, 0.0)
        rrf = (1.0 / (rrf_k + d_rank) if d_rank is not None else 0.0) + (
            1.0 / (rrf_k + s_rank) if s_rank is not None else 0.0
        )
        d_norm = _minmax_norm(d_score, min_dense, max_dense)
        s_norm = _minmax_norm(s_score, min_sparse, max_sparse)
        doc_tokens = set(tokenize_zh_en(records[doc_id].get("text", "")))
        cov = (len(q_tokens & doc_tokens) / len(q_tokens)) if q_tokens else 0.0
        final_score = 0.55 * rrf + 0.25 * d_norm + 0.15 * s_norm + 0.05 * cov
        ranked.append(
            {
                "doc_id": doc_id,
                "final_score": float(final_score),
                "rrf": float(rrf),
                "dense_score": float(d_score),
                "sparse_score": float(s_score),
                "term_coverage": float(cov),
            }
        )
    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    ranked = ranked[: max(rerank_top_n, top_k)]

    if reranker and reranker.enabled() and ranked:
        rerank_scores = reranker.score(question, [records[r["doc_id"]]["text"] for r in ranked])
        if rerank_scores:
            min_r, max_r = min(rerank_scores), max(rerank_scores)
            alpha = max(0.0, min(1.0, reranker_weight))
            for i, row in enumerate(ranked):
                base = row["final_score"]
                model_norm = _minmax_norm(rerank_scores[i], min_r, max_r)
                row["base_score"] = float(base)
                row["model_rerank_score"] = float(rerank_scores[i])
                row["final_score"] = float((1.0 - alpha) * base + alpha * model_norm)
            ranked.sort(key=lambda x: x["final_score"], reverse=True)

    top_rows = ranked[:top_k]
    hits = [
        SearchHit(
            score=row["final_score"],
            text=documents[row["doc_id"]].page_content,
            metadata=documents[row["doc_id"]].metadata or {},
            record_id=row["doc_id"],
        )
        for row in top_rows
    ]
    return hits, top_rows


def build_context(hits: list[SearchHit], max_chars: int) -> str:
    blocks, used = [], 0
    for i, hit in enumerate(hits, start=1):
        block = (
            f"[{i}] title={hit.metadata.get('title','未知车系')}; "
            f"series_id={hit.metadata.get('series_id','')}; "
            f"score={hit.score:.4f}; url={hit.metadata.get('url','N/A')}\n{hit.text}\n"
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n".join(blocks)


def iter_unique_series(hits: list[SearchHit], max_series: int = 5) -> Iterable[SearchHit]:
    seen = set()
    for hit in hits:
        sid = str(hit.metadata.get("series_id", ""))
        if sid in seen:
            continue
        seen.add(sid)
        yield hit
        if len(seen) >= max_series:
            break


def build_langchain_answer_chain(llm_client: LLMClient, temperature: float, no_thinking_prompt: bool):
    system_prompt = (
        "你是汽车知识RAG助手。"
        "必须仅基于给定上下文回答，不要编造。"
        "若上下文不足，明确说'根据当前检索内容无法确认'。"
        "回答后附上引用编号，例如[1][3]。"
    )
    if no_thinking_prompt:
        system_prompt += "不要输出分析过程、推理过程或thinking内容，只输出最终答案。"
    return (
        RunnableLambda(
            lambda payload: {
                **payload,
                "user_prompt": (
                    f"用户问题:\n{payload['question']}\n\n"
                    f"检索上下文:\n{payload['context']}\n\n"
                    "请给出简洁准确的中文答案，并保留引用编号。"
                ),
            }
        )
        | RunnableLambda(
            lambda payload: {
                **payload,
                "answer": llm_client.chat(
                    system_prompt=system_prompt, user_prompt=payload["user_prompt"], temperature=temperature
                ),
            }
        )
    )


def build_store(args) -> None:
    logger.info(f"加载训练数据: {args.input}")
    items = load_training_items(args.input)
    records = training_items_to_records(items)
    if not records:
        raise RuntimeError("没有可用于构建 LangChain 向量库的文档")
    docs = records_to_documents(records)
    embeddings = ProjectEmbeddings(
        provider=args.embedding_provider,
        model=args.embedding_model,
        api_base=args.embedding_api_base,
        api_key=args.embedding_api_key,
        batch_size=args.batch_size,
    )
    logger.info(
        f"开始构建 LangChain FAISS: provider={args.embedding_provider}, "
        f"model={embeddings.model_name}, docs={len(docs)}"
    )
    store = FAISS.from_documents(documents=docs, embedding=embeddings)
    os.makedirs(args.store_dir, exist_ok=True)
    store.save_local(folder_path=args.store_dir, index_name=args.index_name)
    save_records(args.records, records)
    save_meta(
        args.meta,
        {
            "store_type": "langchain_faiss",
            "store_dir": args.store_dir,
            "index_name": args.index_name,
            "records_count": len(records),
            "embedding_provider": args.embedding_provider,
            "embedding_model": embeddings.model_name,
        },
    )
    storage = maybe_init_storage(args)
    if storage:
        run_id = str(uuid.uuid4())
        storage.refresh_chunks(records)
        storage.log_build_run(
            run_id=run_id,
            records_count=len(records),
            index_path=os.path.join(args.store_dir, f"{args.index_name}.faiss"),
            records_path=args.records,
            meta_path=args.meta,
            embedding_provider=args.embedding_provider,
            embedding_model=embeddings.model_name,
        )


def answer_query(args) -> None:
    embeddings = ProjectEmbeddings(
        provider=args.embedding_provider,
        model=args.embedding_model,
        api_base=args.embedding_api_base,
        api_key=args.embedding_api_key,
        batch_size=args.batch_size,
    )
    store = _load_vector_store(args.store_dir, embeddings, args.index_name)
    records = load_records(args.records)
    documents = records_to_documents(records)
    doc_id_map = build_doc_id_map(documents)
    reranker = RerankerClient(
        provider=args.reranker_provider,
        model_name=args.reranker_model,
        device=args.reranker_device,
        fail_open=args.reranker_fail_open,
    )

    def retrieve(payload: dict) -> dict:
        q = payload["question"]
        if args.retrieval_mode == "dense":
            hits = dense_search(q, store, doc_id_map, args.top_k)
            debug = []
        else:
            hits, debug = hybrid_search(
                question=q,
                store=store,
                documents=documents,
                doc_id_map=doc_id_map,
                top_k=args.top_k,
                dense_top_k=args.dense_top_k,
                sparse_top_k=args.sparse_top_k,
                rrf_k=args.rrf_k,
                rerank_top_n=args.rerank_top_n,
                reranker=reranker,
                reranker_weight=args.reranker_weight,
            )
        return {"question": q, "hits": hits, "debug_ranked": debug, "context": build_context(hits, args.max_context_chars)}

    retrieval_result = RunnableLambda(retrieve).invoke({"question": args.question})
    hits = retrieval_result["hits"]
    debug_ranked = retrieval_result["debug_ranked"]
    context = retrieval_result["context"]
    if not hits:
        print("未检索到相关内容，请先执行 build 或更换问题。")
        return
    if args.show_context:
        print("【模型上下文】")
        print(context)
        print()

    llm_client = LLMClient(
        provider=args.llm_provider,
        model=args.llm_model,
        api_base=args.llm_api_base,
        api_key=args.llm_api_key,
        timeout_sec=args.llm_timeout_sec,
    )
    result = build_langchain_answer_chain(
        llm_client,
        args.temperature,
        no_thinking_prompt=args.no_thinking_prompt,
    ).invoke(retrieval_result)

    if args.show_retrieval_debug and debug_ranked:
        print("【LangChain Hybrid 检索重排 Top 候选】")
        for i, row in enumerate(debug_ranked[: min(10, len(debug_ranked))], start=1):
            title = records[row["doc_id"]].get("metadata", {}).get("title", "未知车系")
            print(
                f"{i}. {title} | final={row['final_score']:.4f} | rrf={row['rrf']:.4f} | "
                f"dense={row['dense_score']:.4f} | sparse={row['sparse_score']:.4f} | "
                f"cov={row['term_coverage']:.2f} | rerank={row.get('model_rerank_score', 0.0):.4f}"
            )

    storage = maybe_init_storage(args)
    if storage:
        query_id = str(uuid.uuid4())
        storage.log_query(
            query_id=query_id,
            question=args.question,
            retrieval_mode=f"langchain_{args.retrieval_mode}",
            top_k=args.top_k,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model or "",
            answer_text=result["answer"].strip(),
            hits=[
                {
                    "record_id": h.record_id,
                    "score": h.score,
                    "title": h.metadata.get("title", ""),
                    "series_id": h.metadata.get("series_id", ""),
                    "url": h.metadata.get("url", ""),
                }
                for h in hits
            ],
        )

    print("【LangChain RAG 回答】")
    print(result["answer"].strip())
    print("\n【参考来源】")
    for h in iter_unique_series(hits, max_series=5):
        print(f"- {h.metadata.get('title','未知车系')} | {h.metadata.get('url','N/A')} | score={h.score:.4f}")
