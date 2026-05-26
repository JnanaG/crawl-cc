import argparse
import json
import os
import re
import uuid
from datetime import datetime

from rag_llm_demo import build_context, hybrid_search
from utils.embedding_client import EmbeddingClient
from utils.faiss_vector_store import FaissVectorStore
from utils.llm_client import LLMClient


DEFAULT_STORE_DIR = os.path.join("data", "vector_store", "faiss")
DEFAULT_INDEX = os.path.join(DEFAULT_STORE_DIR, "dongchedi.index")
DEFAULT_RECORDS = os.path.join(DEFAULT_STORE_DIR, "dongchedi_records.jsonl")
DEFAULT_META = os.path.join(DEFAULT_STORE_DIR, "dongchedi_meta.json")
DEFAULT_EVAL_SET = os.path.join("data", "eval", "rag_eval_set.jsonl")
DEFAULT_REPORT = os.path.join("data", "eval", "rag_eval_report.json")
DEFAULT_DETAILS = os.path.join("data", "eval", "rag_eval_details.jsonl")


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def normalize_positive_ids(raw_value) -> set[str]:
    if raw_value is None:
        return set()
    if isinstance(raw_value, list):
        return {str(x) for x in raw_value if str(x).strip()}
    return {str(raw_value)}


def prepare_eval_set(args) -> None:
    records = load_jsonl(args.records)
    templates = [
        "{title}属于什么级别车型？",
        "{title}的官方价格大概多少？",
        "{title}适合什么人群？",
        "{title}有哪些车型版本？",
    ]

    by_series = {}
    for row in records:
        meta = row.get("metadata", {}) or {}
        sid = str(meta.get("series_id", "")).strip()
        title = str(meta.get("title", "")).strip()
        if not sid or not title:
            continue
        if sid not in by_series:
            by_series[sid] = {"title": title}

    eval_rows = []
    index = 0
    for sid, value in by_series.items():
        title = value["title"]
        q = templates[index % len(templates)].format(title=title)
        eval_rows.append(
            {
                "question": q,
                "positive_series_ids": [sid],
                "meta": {"title": title, "source": "auto_generated"},
            }
        )
        index += 1
        if len(eval_rows) >= args.max_samples:
            break

    write_jsonl(args.output, eval_rows)
    print(f"已生成评测集: {args.output} ({len(eval_rows)} 条)")


def evaluate(args) -> None:
    eval_items = load_jsonl(args.eval_set)
    if not eval_items:
        raise RuntimeError(f"评测集为空: {args.eval_set}")

    emb_client = EmbeddingClient(
        provider=args.embedding_provider,
        model=args.embedding_model,
        api_base=args.embedding_api_base,
        api_key=args.embedding_api_key,
    )
    store = FaissVectorStore(dim=1)
    store.load(index_path=args.index, records_path=args.records, meta_path=args.meta)

    llm_client = None
    if args.with_generation:
        llm_client = LLMClient(
            provider=args.llm_provider,
            model=args.llm_model,
            api_base=args.llm_api_base,
            api_key=args.llm_api_key,
        )

    hit_flags = []
    recall_values = []
    mrr_values = []
    citation_precision_values = []
    citation_hit_values = []
    answer_length_values = []
    answer_has_citation_values = []
    cannot_confirm_values = []
    details = []

    for item in eval_items:
        question = str(item.get("question", "")).strip()
        positive_ids = normalize_positive_ids(item.get("positive_series_ids"))
        if not question:
            continue

        query_vec = emb_client.embed_query(question)
        if args.retrieval_mode == "hybrid":
            hits, _ = hybrid_search(
                question=question,
                query_vec=query_vec,
                store=store,
                top_k=args.top_k,
                dense_top_k=args.dense_top_k,
                sparse_top_k=args.sparse_top_k,
                rrf_k=args.rrf_k,
                rerank_top_n=args.rerank_top_n,
            )
        else:
            hits = store.search(query_vector=query_vec, top_k=args.top_k)

        top_hits = hits[: args.top_k]
        matched_ids = set()
        first_relevant_rank = None
        for rank, hit in enumerate(top_hits, start=1):
            sid = str(hit.metadata.get("series_id", "")).strip()
            if sid in positive_ids:
                matched_ids.add(sid)
                if first_relevant_rank is None:
                    first_relevant_rank = rank

        hit_at_k = 1.0 if first_relevant_rank is not None else 0.0
        recall_at_k = (len(matched_ids) / max(len(positive_ids), 1)) if positive_ids else 0.0
        mrr = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0

        hit_flags.append(hit_at_k)
        recall_values.append(recall_at_k)
        mrr_values.append(mrr)

        row_detail = {
            "question": question,
            "positive_series_ids": sorted(list(positive_ids)),
            "hit_at_k": hit_at_k,
            "recall_at_k": recall_at_k,
            "mrr": mrr,
            "retrieved": [
                {
                    "rank": idx + 1,
                    "score": h.score,
                    "series_id": str(h.metadata.get("series_id", "")),
                    "title": h.metadata.get("title", ""),
                    "url": h.metadata.get("url", ""),
                }
                for idx, h in enumerate(top_hits)
            ],
        }

        if llm_client:
            context = build_context(top_hits, args.max_context_chars)
            system_prompt = (
                "你是汽车知识RAG助手。"
                "必须仅基于给定上下文回答，不要编造。"
                "若上下文不足，明确说'根据当前检索内容无法确认'。"
                "回答后附上引用编号，例如[1][3]。"
            )
            user_prompt = (
                f"用户问题:\n{question}\n\n"
                f"检索上下文:\n{context}\n\n"
                "请给出简洁准确的中文答案，并保留引用编号。"
            )
            answer = llm_client.chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=args.temperature)

            cited_idx = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
            valid_idx = [x for x in cited_idx if 1 <= x <= len(top_hits)]
            cited_series_ids = [
                str(top_hits[i - 1].metadata.get("series_id", "")).strip()
                for i in valid_idx
            ]
            if valid_idx:
                correct = sum(1 for sid in cited_series_ids if sid in positive_ids)
                citation_precision = float(correct / len(valid_idx))
                citation_hit = 1.0 if correct > 0 else 0.0
            else:
                citation_precision = 0.0
                citation_hit = 0.0

            citation_precision_values.append(citation_precision)
            citation_hit_values.append(citation_hit)
            answer_length_values.append(float(len(answer)))
            answer_has_citation_values.append(1.0 if cited_idx else 0.0)
            cannot_confirm_values.append(1.0 if "无法确认" in answer else 0.0)

            row_detail["answer"] = answer
            row_detail["citation_indexes"] = cited_idx
            row_detail["citation_precision"] = citation_precision
            row_detail["citation_hit"] = citation_hit
            row_detail["answer_has_citation"] = 1.0 if cited_idx else 0.0
            row_detail["contains_cannot_confirm"] = 1.0 if "无法确认" in answer else 0.0

        details.append(row_detail)

    summary = {
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(),
        "eval_set": args.eval_set,
        "total_questions": len(details),
        "retrieval_mode": args.retrieval_mode,
        "top_k": args.top_k,
        "metrics": {
            "hit_rate_at_k": safe_mean(hit_flags),
            "recall_at_k": safe_mean(recall_values),
            "mrr": safe_mean(mrr_values),
        },
    }

    if llm_client:
        summary["metrics"]["citation_precision"] = safe_mean(citation_precision_values)
        summary["metrics"]["citation_hit_rate"] = safe_mean(citation_hit_values)
        summary["metrics"]["answer_has_citation_rate"] = safe_mean(answer_has_citation_values)
        summary["metrics"]["cannot_confirm_rate"] = safe_mean(cannot_confirm_values)
        summary["metrics"]["avg_answer_length"] = safe_mean(answer_length_values)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_jsonl(args.details, details)

    print("评测完成。")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    print(f"summary: {args.report}")
    print(f"details: {args.details}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG 效果评估工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="根据现有 records 自动生成评测集")
    p_prepare.add_argument("--records", default=DEFAULT_RECORDS)
    p_prepare.add_argument("--output", default=DEFAULT_EVAL_SET)
    p_prepare.add_argument("--max-samples", type=int, default=60)

    p_run = sub.add_parser("run", help="执行检索/生成评测")
    p_run.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
    p_run.add_argument("--index", default=DEFAULT_INDEX)
    p_run.add_argument("--records", default=DEFAULT_RECORDS)
    p_run.add_argument("--meta", default=DEFAULT_META)
    p_run.add_argument("--report", default=DEFAULT_REPORT)
    p_run.add_argument("--details", default=DEFAULT_DETAILS)
    p_run.add_argument("--top-k", type=int, default=6)
    p_run.add_argument("--retrieval-mode", default="hybrid", choices=["hybrid", "dense"])
    p_run.add_argument("--dense-top-k", type=int, default=24)
    p_run.add_argument("--sparse-top-k", type=int, default=40)
    p_run.add_argument("--rrf-k", type=int, default=60)
    p_run.add_argument("--rerank-top-n", type=int, default=12)

    p_run.add_argument(
        "--embedding-provider",
        default="openai_compatible",
        choices=["openai_compatible", "ollama", "fastembed", "sentence_transformers"],
    )
    p_run.add_argument("--embedding-model", default=None)
    p_run.add_argument("--embedding-api-base", default=None)
    p_run.add_argument("--embedding-api-key", default=None)

    p_run.add_argument("--with-generation", action="store_true")
    p_run.add_argument("--max-context-chars", type=int, default=3200)
    p_run.add_argument("--temperature", type=float, default=0.2)
    p_run.add_argument("--llm-provider", default="openai_compatible", choices=["openai_compatible", "ollama"])
    p_run.add_argument("--llm-model", default=None)
    p_run.add_argument("--llm-api-base", default=None)
    p_run.add_argument("--llm-api-key", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_eval_set(args)
        return
    if args.command == "run":
        evaluate(args)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
