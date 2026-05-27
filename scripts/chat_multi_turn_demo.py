from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conversation import MultiTurnChatService
from conversation.rag_backend import FaissRAGBackend, HashRAGBackend
from memory import ConversationMemoryStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="多轮对话最小 Demo")
    parser.add_argument("--message", required=True, help="当前轮用户消息")
    parser.add_argument("--session-id", default=None, help="已有会话 ID；不传则自动创建")
    parser.add_argument("--user-id", default="demo-user")
    parser.add_argument("--memory-db", default="data/conversation/chat_memory.sqlite3")
    parser.add_argument("--backend-mode", default="hash", choices=["hash", "faiss"])
    parser.add_argument("--index", default="data/vector_store/faiss/dongchedi.index")
    parser.add_argument("--records", default="data/vector_store/faiss/dongchedi_records.jsonl")
    parser.add_argument("--meta", default="data/vector_store/faiss/dongchedi_meta.json")
    parser.add_argument(
        "--embedding-provider",
        default="fastembed",
        choices=["openai_compatible", "ollama", "fastembed", "sentence_transformers"],
    )
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-api-base", default=None)
    parser.add_argument("--embedding-api-key", default=None)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-provider", default="ollama", choices=["openai_compatible", "ollama"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-base", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--retrieval-mode", default="hybrid", choices=["hybrid", "dense"])
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--show-session", action="store_true", help="打印当前会话完整状态")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    store = ConversationMemoryStore(db_path=args.memory_db)
    if args.backend_mode == "faiss":
        rag_backend = FaissRAGBackend(
            index_path=args.index,
            records_path=args.records,
            meta_path=args.meta,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            embedding_api_base=args.embedding_api_base,
            embedding_api_key=args.embedding_api_key,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_api_base=args.llm_api_base,
            llm_api_key=args.llm_api_key,
            retrieval_mode=args.retrieval_mode,
            top_k=args.top_k,
            use_llm=args.use_llm,
        )
    else:
        rag_backend = HashRAGBackend(
            records_path=args.records,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_api_base=args.llm_api_base,
            llm_api_key=args.llm_api_key,
            top_k=args.top_k,
            use_llm=args.use_llm,
        )
    service = MultiTurnChatService(store=store, rag_backend=rag_backend)
    payload = service.chat(message=args.message, session_id=args.session_id, user_id=args.user_id)

    result = {
        "session_id": payload.session_id,
        "turn_id": payload.turn_id,
        "route": payload.route,
        "rewritten_query": payload.rewritten_query,
        "should_clarify": payload.should_clarify,
        "clarification_question": payload.clarification_question,
        "memory_summary": payload.memory_summary,
        "preferences": payload.preferences.model_dump(),
        "task_memory": payload.task_memory.model_dump(),
        "hits": [hit.model_dump() for hit in payload.hits[:4]],
        "answer": payload.answer,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.show_session:
        print("\n=== session_state ===")
        print(json.dumps(store.export_session(payload.session_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
