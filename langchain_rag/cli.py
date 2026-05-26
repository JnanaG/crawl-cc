import argparse

from langchain_rag.config import (
    DEFAULT_INDEX_NAME,
    DEFAULT_INPUT,
    DEFAULT_META,
    DEFAULT_RECORDS,
    DEFAULT_STORAGE_DB,
    DEFAULT_STORE_DIR,
)
from langchain_rag.core import answer_query, build_store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="懂车帝 LangChain RAG（并行版本）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="构建 LangChain FAISS 向量库")
    p_build.add_argument("--input", default=DEFAULT_INPUT)
    p_build.add_argument("--store-dir", default=DEFAULT_STORE_DIR)
    p_build.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    p_build.add_argument("--records", default=DEFAULT_RECORDS)
    p_build.add_argument("--meta", default=DEFAULT_META)
    p_build.add_argument(
        "--embedding-provider",
        default="openai_compatible",
        choices=["openai_compatible", "ollama", "fastembed", "sentence_transformers"],
    )
    p_build.add_argument("--embedding-model", default=None)
    p_build.add_argument("--embedding-api-base", default=None)
    p_build.add_argument("--embedding-api-key", default=None)
    p_build.add_argument("--batch-size", type=int, default=64)
    p_build.add_argument("--storage-db", default=DEFAULT_STORAGE_DB)
    p_build.add_argument("--disable-storage", action="store_true")

    p_query = sub.add_parser("query", help="执行 LangChain RAG 问答")
    p_query.add_argument("--question", required=True)
    p_query.add_argument("--store-dir", default=DEFAULT_STORE_DIR)
    p_query.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
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
    p_query.add_argument("--batch-size", type=int, default=64)
    p_query.add_argument(
        "--embedding-provider",
        default="openai_compatible",
        choices=["openai_compatible", "ollama", "fastembed", "sentence_transformers"],
    )
    p_query.add_argument("--embedding-model", default=None)
    p_query.add_argument("--embedding-api-base", default=None)
    p_query.add_argument("--embedding-api-key", default=None)
    p_query.add_argument("--llm-provider", default="openai_compatible", choices=["openai_compatible", "ollama"])
    p_query.add_argument("--llm-model", default=None)
    p_query.add_argument("--llm-api-base", default=None)
    p_query.add_argument("--llm-api-key", default=None)
    p_query.add_argument("--llm-timeout-sec", type=int, default=90)
    p_query.add_argument("--no-thinking-prompt", action="store_true")
    p_query.add_argument("--storage-db", default=DEFAULT_STORAGE_DB)
    p_query.add_argument("--disable-storage", action="store_true")
    return parser


def main() -> None:
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
