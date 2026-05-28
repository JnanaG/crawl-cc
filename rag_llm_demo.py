import argparse
import json
import os
import re
import time
import uuid
from collections import Counter
from typing import Iterable

from loguru import logger

from utils.embedding_client import EmbeddingClient
from utils.faiss_vector_store import FaissVectorStore, SearchHit
from utils.light_storage import LightRAGStorage
from utils.llm_client import LLMClient
from utils.reranker_client import RerankerClient


DEFAULT_INPUT = os.path.join("data", "processed", "dongchedi_training_data.jsonl")
DEFAULT_STORE_DIR = os.path.join("data", "vector_store", "faiss")
DEFAULT_INDEX = os.path.join(DEFAULT_STORE_DIR, "dongchedi.index")
DEFAULT_RECORDS = os.path.join(DEFAULT_STORE_DIR, "dongchedi_records.jsonl")
DEFAULT_META = os.path.join(DEFAULT_STORE_DIR, "dongchedi_meta.json")
DEFAULT_STORAGE_DB = os.path.join("data", "storage", "rag.duckdb")


def load_training_items(jsonl_path: str) -> list[dict]:
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_training_items_from_paths(paths: list[str]) -> list[dict]:
    all_items = []
    for path in paths:
        current_items = load_training_items(path)
        logger.info(f"加载输入语料: path={path}, rows={len(current_items)}")
        all_items.extend(current_items)
    return all_items


def batched(values: list, batch_size: int) -> Iterable[list]:
    for i in range(0, len(values), batch_size):
        yield values[i : i + batch_size]


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


class LightweightBM25:
    """
    轻量 BM25 实现（无额外依赖），用于 hybrid retrieval 的 sparse 召回。
    """

    def __init__(self, records: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tf = []
        self.doc_len = []
        self.avgdl = 0.0
        self.idf = {}

        df = Counter()
        for record in records:
            tokens = tokenize_zh_en(record.get("text", ""))
            tf = Counter(tokens)
            self.doc_tf.append(tf)
            self.doc_len.append(len(tokens))
            for term in tf:
                df[term] += 1

        n_docs = max(len(records), 1)
        self.avgdl = sum(self.doc_len) / n_docs if self.doc_len else 0.0
        for term, freq in df.items():
            # BM25 常见平滑形式
            self.idf[term] = max(0.0, float((n_docs - freq + 0.5) / (freq + 0.5)))

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        q_terms = tokenize_zh_en(query)
        if not q_terms or not self.doc_tf:
            return []

        scores = []
        for doc_id, tf in enumerate(self.doc_tf):
            dl = self.doc_len[doc_id]
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f <= 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-6)))
                score += idf * ((f * (self.k1 + 1.0)) / max(denom, 1e-6))
            if score > 0:
                scores.append((doc_id, float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def _minmax_norm(value: float, min_v: float, max_v: float) -> float:
    if max_v - min_v <= 1e-8:
        return 0.0
    return (value - min_v) / (max_v - min_v)


def hybrid_search(
    question: str,
    query_vec: list[float],
    store: FaissVectorStore,
    top_k: int,
    dense_top_k: int,
    sparse_top_k: int,
    rrf_k: int,
    rerank_top_n: int,
    reranker: RerankerClient | None = None,
    reranker_weight: float = 0.65,
) -> tuple[list[SearchHit], list[dict]]:
    dense_hits = store.search(query_vector=query_vec, top_k=max(dense_top_k, top_k))
    bm25 = LightweightBM25(store.records)
    sparse_hits = bm25.search(question, top_k=max(sparse_top_k, top_k))

    dense_rank = {}
    dense_score = {}
    for rank, hit in enumerate(dense_hits, start=1):
        if hit.record_id < 0:
            continue
        dense_rank[hit.record_id] = rank
        dense_score[hit.record_id] = hit.score

    sparse_rank = {}
    sparse_score = {}
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

        rrf = 0.0
        if d_rank is not None:
            rrf += 1.0 / (rrf_k + d_rank)
        if s_rank is not None:
            rrf += 1.0 / (rrf_k + s_rank)

        d_norm = _minmax_norm(d_score, min_dense, max_dense)
        s_norm = _minmax_norm(s_score, min_sparse, max_sparse)

        doc_text = store.records[doc_id].get("text", "")
        doc_tokens = set(tokenize_zh_en(doc_text))
        term_coverage = (len(q_tokens & doc_tokens) / len(q_tokens)) if q_tokens else 0.0

        # 第二阶段重排：融合分数 + 关键词覆盖，避免“高向量分但答非所问”
        final_score = 0.55 * rrf + 0.25 * d_norm + 0.15 * s_norm + 0.05 * term_coverage
        ranked.append(
            {
                "doc_id": doc_id,
                "final_score": float(final_score),
                "rrf": float(rrf),
                "dense_score": float(d_score),
                "sparse_score": float(s_score),
                "term_coverage": float(term_coverage),
                "dense_rank": d_rank,
                "sparse_rank": s_rank,
            }
        )

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    ranked = ranked[: max(rerank_top_n, top_k)]

    if reranker and reranker.enabled() and ranked:
        rerank_scores = reranker.score(
            question,
            [store.records[item["doc_id"]].get("text", "") for item in ranked],
        )
        if rerank_scores:
            min_r, max_r = min(rerank_scores), max(rerank_scores)
            alpha = max(0.0, min(1.0, reranker_weight))
            for i, item in enumerate(ranked):
                raw = rerank_scores[i]
                model_norm = _minmax_norm(raw, min_r, max_r)
                base = item["final_score"]
                fused = (1.0 - alpha) * base + alpha * model_norm
                item["base_score"] = float(base)
                item["model_rerank_score"] = float(raw)
                item["final_score"] = float(fused)
            ranked.sort(key=lambda x: x["final_score"], reverse=True)

    final_ranked = ranked[:top_k]

    hits = []
    for item in final_ranked:
        doc_id = item["doc_id"]
        record = store.records[doc_id]
        hits.append(
            SearchHit(
                score=item["final_score"],
                text=record.get("text", ""),
                metadata=record.get("metadata", {}),
                record_id=doc_id,
            )
        )
    return hits, final_ranked


def build_store(args) -> None:
    input_paths = [args.input, *(args.extra_inputs or [])]
    logger.info(f"加载训练数据: inputs={input_paths}")
    items = load_training_items_from_paths(input_paths)
    logger.info(f"训练样本条数: {len(items)}")

    records = []
    texts = []
    modality_counter = Counter()
    for item in items:
        text = item.get("text", "").strip()
        if not text:
            continue
        metadata = item.get("metadata", {})
        modality_counter[metadata.get("modality") or metadata.get("content_type") or "text"] += 1
        records.append({"text": text, "metadata": metadata})
        texts.append(text)

    emb_client = EmbeddingClient(
        provider=args.embedding_provider,
        model=args.embedding_model,
        api_base=args.embedding_api_base,
        api_key=args.embedding_api_key,
    )

    logger.info(
        f"开始生成向量: provider={args.embedding_provider}, model={emb_client.model_name}, "
        f"batch_size={args.batch_size}"
    )
    vectors = []
    total_texts = len(texts)
    total_batches = (total_texts + args.batch_size - 1) // max(1, args.batch_size)
    start_time = time.perf_counter()
    for batch_idx, chunk in enumerate(batched(texts, args.batch_size), start=1):
        vectors.extend(emb_client.embed_texts(chunk))
        done = min(batch_idx * args.batch_size, total_texts)
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        rate = done / elapsed
        remain = max(total_texts - done, 0)
        eta_sec = int(remain / max(rate, 1e-6))
        progress = (done / max(total_texts, 1)) * 100
        logger.info(
            f"[build进度] {done}/{total_texts} ({progress:.1f}%) | "
            f"batch={batch_idx}/{total_batches} | elapsed={elapsed:.1f}s | eta={eta_sec}s"
        )

    if not vectors:
        raise RuntimeError("未生成任何向量，请检查输入数据或 embedding 配置")

    dim = len(vectors[0])
    store = FaissVectorStore(dim=dim)
    store.build(vectors=vectors, records=records)
    store.save(index_path=args.index, records_path=args.records, meta_path=args.meta)
    storage = maybe_init_storage(args)
    if storage:
        run_id = str(uuid.uuid4())
        storage.refresh_chunks(records)
        storage.log_build_run(
            run_id=run_id,
            records_count=len(records),
            index_path=args.index,
            records_path=args.records,
            meta_path=args.meta,
            embedding_provider=args.embedding_provider,
            embedding_model=emb_client.model_name,
        )
        logger.info(f"轻量存储层写入完成: db={args.storage_db}, run_id={run_id}")
    logger.info(f"FAISS 构建完成: vectors={len(records)}, dim={dim}")
    logger.info(f"语料模态分布: {dict(modality_counter)}")
    logger.info(f"索引文件: {args.index}")
    logger.info(f"记录文件: {args.records}")
    logger.info(f"元信息文件: {args.meta}")


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


def build_context(hits: list[SearchHit], max_chars: int) -> str:
    contexts = []
    used = 0
    for i, hit in enumerate(hits, start=1):
        title = hit.metadata.get("title", "未知车系")
        url = hit.metadata.get("url", "N/A")
        sid = hit.metadata.get("series_id", "")
        modality = hit.metadata.get("modality") or hit.metadata.get("content_type") or "text"
        image_category = hit.metadata.get("image_category_name") or ""
        block = (
            f"[{i}] title={title}; series_id={sid}; modality={modality}; "
            f"image_category={image_category or 'N/A'}; score={hit.score:.4f}; url={url}\n"
            f"{hit.text}\n"
        )
        if used + len(block) > max_chars:
            break
        contexts.append(block)
        used += len(block)
    return "\n".join(contexts)


def answer_query(args) -> None:
    emb_client = EmbeddingClient(
        provider=args.embedding_provider,
        model=args.embedding_model,
        api_base=args.embedding_api_base,
        api_key=args.embedding_api_key,
    )
    store = FaissVectorStore(dim=1)
    store.load(index_path=args.index, records_path=args.records, meta_path=args.meta)

    query_vec = emb_client.embed_query(args.question)
    reranker = RerankerClient(
        provider=args.reranker_provider,
        model_name=args.reranker_model,
        device=args.reranker_device,
        fail_open=args.reranker_fail_open,
    )

    if args.retrieval_mode == "dense":
        hits = store.search(query_vector=query_vec, top_k=args.top_k)
        debug_ranked = []
    else:
        hits, debug_ranked = hybrid_search(
            question=args.question,
            query_vec=query_vec,
            store=store,
            top_k=args.top_k,
            dense_top_k=args.dense_top_k,
            sparse_top_k=args.sparse_top_k,
            rrf_k=args.rrf_k,
            rerank_top_n=args.rerank_top_n,
            reranker=reranker,
            reranker_weight=args.reranker_weight,
        )
    if not hits:
        print("未检索到相关内容，请先执行 build 或更换问题。")
        return

    if args.show_retrieval_debug and debug_ranked:
        print("【Hybrid 检索重排 Top 候选】")
        for i, row in enumerate(debug_ranked[: min(10, len(debug_ranked))], start=1):
            record = store.records[row["doc_id"]]
            title = record.get("metadata", {}).get("title", "未知车系")
            print(
                f"{i}. {title} | final={row['final_score']:.4f} | "
                f"rrf={row['rrf']:.4f} | dense={row['dense_score']:.4f} | "
                f"sparse={row['sparse_score']:.4f} | cov={row['term_coverage']:.2f} | "
                f"rerank={row.get('model_rerank_score', 0.0):.4f}"
            )

    context = build_context(hits, args.max_context_chars)
    if args.show_context:
        print("【模型上下文】")
        print(context)
        print()
    system_prompt = (
        "你是汽车知识RAG助手。"
        "必须仅基于给定上下文回答，不要编造。"
        "若上下文不足，明确说'根据当前检索内容无法确认'。"
        "回答后附上引用编号，例如[1][3]。"
    )
    user_prompt = (
        f"用户问题:\n{args.question}\n\n"
        f"检索上下文:\n{context}\n\n"
        "请给出简洁准确的中文答案，并保留引用编号。"
    )

    llm_client = LLMClient(
        provider=args.llm_provider,
        model=args.llm_model,
        api_base=args.llm_api_base,
        api_key=args.llm_api_key,
    )
    answer = llm_client.chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=args.temperature)

    storage = maybe_init_storage(args)
    if storage:
        query_id = str(uuid.uuid4())
        storage.log_query(
            query_id=query_id,
            question=args.question,
            retrieval_mode=args.retrieval_mode,
            top_k=args.top_k,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model or "",
            answer_text=answer.strip(),
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
        logger.info(f"查询日志已写入轻量存储层: db={args.storage_db}, query_id={query_id}")

    print("【RAG 标准版回答】")
    print(answer.strip())
    print("\n【参考来源】")
    for h in iter_unique_series(hits, max_series=5):
        print(
            f"- {h.metadata.get('title', '未知车系')} | "
            f"{h.metadata.get('url', 'N/A')} | score={h.score:.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="懂车帝标准RAG（真实Embedding + FAISS + LLM）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="构建 FAISS 向量库")
    p_build.add_argument("--input", default=DEFAULT_INPUT)
    p_build.add_argument("--extra-inputs", nargs="*", default=[])
    p_build.add_argument("--index", default=DEFAULT_INDEX)
    p_build.add_argument("--records", default=DEFAULT_RECORDS)
    p_build.add_argument("--meta", default=DEFAULT_META)
    p_build.add_argument(
        "--embedding-provider",
        default="openai_compatible",
        choices=["openai_compatible", "ollama", "fastembed", "sentence_transformers", "hash"],
    )
    p_build.add_argument("--embedding-model", default=None)
    p_build.add_argument("--embedding-api-base", default=None)
    p_build.add_argument("--embedding-api-key", default=None)
    p_build.add_argument("--batch-size", type=int, default=64)
    p_build.add_argument("--storage-db", default=DEFAULT_STORAGE_DB)
    p_build.add_argument("--disable-storage", action="store_true")

    p_query = sub.add_parser("query", help="执行 RAG 问答")
    p_query.add_argument("--question", required=True)
    p_query.add_argument("--index", default=DEFAULT_INDEX)
    p_query.add_argument("--records", default=DEFAULT_RECORDS)
    p_query.add_argument("--meta", default=DEFAULT_META)
    p_query.add_argument("--top-k", type=int, default=6)
    p_query.add_argument("--retrieval-mode", default="hybrid", choices=["hybrid", "dense"])
    p_query.add_argument("--dense-top-k", type=int, default=24)
    p_query.add_argument("--sparse-top-k", type=int, default=40)
    p_query.add_argument("--rrf-k", type=int, default=60)
    p_query.add_argument("--rerank-top-n", type=int, default=12)
    p_query.add_argument("--reranker-provider", default="none", choices=["none", "cross_encoder"])
    p_query.add_argument("--reranker-model", default=None)
    p_query.add_argument("--reranker-device", default=None)
    p_query.add_argument("--reranker-weight", type=float, default=0.65)
    p_query.add_argument("--reranker-fail-open", action="store_true", default=True)
    p_query.add_argument("--show-retrieval-debug", action="store_true")
    p_query.add_argument("--show-context", action="store_true")
    p_query.add_argument("--max-context-chars", type=int, default=3200)
    p_query.add_argument("--temperature", type=float, default=0.2)

    p_query.add_argument(
        "--embedding-provider",
        default="openai_compatible",
        choices=["openai_compatible", "ollama", "fastembed", "sentence_transformers", "hash"],
    )
    p_query.add_argument("--embedding-model", default=None)
    p_query.add_argument("--embedding-api-base", default=None)
    p_query.add_argument("--embedding-api-key", default=None)

    p_query.add_argument("--llm-provider", default="openai_compatible", choices=["openai_compatible", "ollama"])
    p_query.add_argument("--llm-model", default=None)
    p_query.add_argument("--llm-api-base", default=None)
    p_query.add_argument("--llm-api-key", default=None)
    p_query.add_argument("--storage-db", default=DEFAULT_STORAGE_DB)
    p_query.add_argument("--disable-storage", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build":
        build_store(args)
        return
    if args.command == "query":
        answer_query(args)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
