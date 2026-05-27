from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动多轮对话在线 API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--memory-db", default="data/conversation/chat_memory.sqlite3")
    parser.add_argument("--backend-mode", default="hash", choices=["hash", "faiss"])
    parser.add_argument("--records", default="data/vector_store/faiss/dongchedi_records.jsonl")
    parser.add_argument("--index", default="data/vector_store/faiss/dongchedi.index")
    parser.add_argument("--meta", default="data/vector_store/faiss/dongchedi_meta.json")
    parser.add_argument("--embedding-provider", default="fastembed")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-api-base", default=None)
    parser.add_argument("--embedding-api-key", default=None)
    parser.add_argument("--llm-provider", default="ollama", choices=["openai_compatible", "ollama"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-base", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--retrieval-mode", default="hybrid", choices=["hybrid", "dense"])
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--use-llm", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    os.environ["CHAT_MEMORY_DB"] = args.memory_db
    os.environ["CHAT_BACKEND_MODE"] = args.backend_mode
    os.environ["CHAT_RECORDS_PATH"] = args.records
    os.environ["CHAT_INDEX_PATH"] = args.index
    os.environ["CHAT_META_PATH"] = args.meta
    os.environ["CHAT_EMBEDDING_PROVIDER"] = args.embedding_provider
    os.environ["CHAT_RETRIEVAL_MODE"] = args.retrieval_mode
    os.environ["CHAT_TOP_K"] = str(args.top_k)
    os.environ["CHAT_USE_LLM"] = "true" if args.use_llm else "false"
    if args.embedding_model:
        os.environ["CHAT_EMBEDDING_MODEL"] = args.embedding_model
    if args.embedding_api_base:
        os.environ["CHAT_EMBEDDING_API_BASE"] = args.embedding_api_base
    if args.embedding_api_key:
        os.environ["CHAT_EMBEDDING_API_KEY"] = args.embedding_api_key
    if args.llm_model:
        os.environ["CHAT_LLM_MODEL"] = args.llm_model
    if args.llm_api_base:
        os.environ["CHAT_LLM_API_BASE"] = args.llm_api_base
    if args.llm_api_key:
        os.environ["CHAT_LLM_API_KEY"] = args.llm_api_key
    os.environ["CHAT_LLM_PROVIDER"] = args.llm_provider

    import uvicorn

    uvicorn.run(
        "conversation.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
