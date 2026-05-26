import argparse
import json
import os
from typing import Iterable

from loguru import logger

from utils.simple_vector_store import SimpleVectorStore


DEFAULT_INPUT = os.path.join("data", "processed", "dongchedi_training_data.jsonl")
DEFAULT_STORE_DIR = os.path.join("data", "vector_store")
DEFAULT_INDEX = os.path.join(DEFAULT_STORE_DIR, "dongchedi_hashvec.npz")
DEFAULT_RECORDS = os.path.join(DEFAULT_STORE_DIR, "dongchedi_records.jsonl")


def load_training_items(jsonl_path: str) -> list[dict]:
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def iter_unique_series(hits: Iterable, max_series: int = 3):
    seen = set()
    for hit in hits:
        sid = str(hit.metadata.get("series_id", ""))
        if sid in seen:
            continue
        seen.add(sid)
        yield hit
        if len(seen) >= max_series:
            break


def build_store(input_path: str, index_path: str, records_path: str, dim: int) -> None:
    logger.info(f"加载训练数据: {input_path}")
    items = load_training_items(input_path)
    logger.info(f"训练样本条数: {len(items)}")
    store = SimpleVectorStore(dim=dim)

    def on_progress(done: int, total: int, elapsed: float, eta_sec: int) -> None:
        if done % 100 == 0 or done == total:
            progress = (done / max(total, 1)) * 100
            logger.info(
                f"[build进度] {done}/{total} ({progress:.1f}%) | "
                f"elapsed={elapsed:.1f}s | eta={eta_sec}s"
            )

    store.build(items, progress_callback=on_progress)
    store.save(index_path=index_path, records_path=records_path)
    logger.info(
        f"向量库构建完成: vectors={store.embeddings.shape[0]}, dim={store.embeddings.shape[1]}"
    )
    logger.info(f"索引文件: {index_path}")
    logger.info(f"记录文件: {records_path}")


def answer_query(
    query: str,
    index_path: str,
    records_path: str,
    top_k: int,
    max_context_chars: int,
) -> None:
    store = SimpleVectorStore()
    store.load(index_path=index_path, records_path=records_path)
    hits = store.search(query=query, top_k=top_k)
    if not hits:
        print("未检索到相关内容，请先执行 build 或更换问题。")
        return

    contexts = []
    for h in hits:
        title = h.metadata.get("title", "未知车系")
        sid = h.metadata.get("series_id", "")
        snippet = h.text[: max_context_chars // max(1, top_k)]
        contexts.append(f"[{title} | series_id={sid} | score={h.score:.4f}] {snippet}")

    answer_lines = [
        "【RAG Demo 回答】",
        f"问题: {query}",
        "基于检索结果，相关内容如下:",
    ]
    answer_lines.extend([f"{i + 1}. {c}" for i, c in enumerate(contexts)])
    answer_lines.append("建议: 若用于生产，请接入真实Embedding模型和LLM生成器。")
    print("\n".join(answer_lines))

    print("\n【参考来源】")
    for h in iter_unique_series(hits, max_series=5):
        print(
            f"- {h.metadata.get('title', '未知车系')} | "
            f"{h.metadata.get('url', 'N/A')} | score={h.score:.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="懂车帝数据最小可用 RAG Demo")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="从训练数据构建本地向量库")
    p_build.add_argument("--input", default=DEFAULT_INPUT, help="输入训练JSONL路径")
    p_build.add_argument("--index", default=DEFAULT_INDEX, help="向量索引npz路径")
    p_build.add_argument("--records", default=DEFAULT_RECORDS, help="记录jsonl路径")
    p_build.add_argument("--dim", type=int, default=768, help="向量维度")

    p_query = sub.add_parser("query", help="基于本地向量库执行问答检索")
    p_query.add_argument("--question", required=True, help="用户问题")
    p_query.add_argument("--index", default=DEFAULT_INDEX, help="向量索引npz路径")
    p_query.add_argument("--records", default=DEFAULT_RECORDS, help="记录jsonl路径")
    p_query.add_argument("--top-k", type=int, default=5, help="返回候选数量")
    p_query.add_argument("--max-context-chars", type=int, default=1200, help="输出上下文字符预算")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build":
        build_store(
            input_path=args.input,
            index_path=args.index,
            records_path=args.records,
            dim=args.dim,
        )
        return

    if args.command == "query":
        answer_query(
            query=args.question,
            index_path=args.index,
            records_path=args.records,
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
