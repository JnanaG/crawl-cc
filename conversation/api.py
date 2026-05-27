from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from conversation.dashboard import get_assets_summary, get_evaluation_summary, get_feedback_summary, get_overview_payload, get_reports_summary, get_workflow_summary
from conversation.rag_backend import create_rag_backend
from conversation.service import MultiTurnChatService
from memory.store import ConversationMemoryStore

WEB_DIR = Path(__file__).resolve().parent / "web"
logger = logging.getLogger(__name__)


class CreateSessionRequest(BaseModel):
    user_id: str | None = None
    title: str = ""


class SessionSummary(BaseModel):
    session_id: str
    user_id: str = ""
    title: str = ""
    created_at: str
    updated_at: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    user_id: str | None = None
    runtime_config: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    session_id: str
    turn_id: str
    route: str
    rewritten_query: str
    should_clarify: bool
    clarification_question: str = ""
    memory_summary: str = ""
    preferences: dict[str, Any]
    task_memory: dict[str, Any]
    hits: list[dict[str, Any]]
    runtime_config_used: dict[str, Any] = Field(default_factory=dict)
    answer: str


class SessionDetail(BaseModel):
    session_id: str
    user_id: str = ""
    title: str = ""
    created_at: str
    updated_at: str
    messages: list[dict[str, Any]]
    preference: dict[str, Any]
    task_memory: dict[str, Any]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _get_default_runtime_config() -> dict[str, Any]:
    return {
        "backend_mode": os.getenv("CHAT_BACKEND_MODE", "hash"),
        "embedding_provider": os.getenv("CHAT_EMBEDDING_PROVIDER", "fastembed"),
        "embedding_model": os.getenv("CHAT_EMBEDDING_MODEL") or "",
        "embedding_api_base": os.getenv("CHAT_EMBEDDING_API_BASE") or "",
        "retrieval_mode": os.getenv("CHAT_RETRIEVAL_MODE", "hybrid"),
        "top_k": int(os.getenv("CHAT_TOP_K", "6")),
        "use_llm": _env_bool("CHAT_USE_LLM", False),
        "llm_provider": os.getenv("CHAT_LLM_PROVIDER", "ollama"),
        "llm_model": os.getenv("CHAT_LLM_MODEL") or "",
        "llm_api_base": os.getenv("CHAT_LLM_API_BASE") or "",
    }


def _sanitize_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(config)
    sanitized.pop("llm_api_key", None)
    sanitized.pop("embedding_api_key", None)
    return sanitized


def _build_service_from_env() -> MultiTurnChatService:
    store = ConversationMemoryStore(
        db_path=os.getenv("CHAT_MEMORY_DB", os.path.join("data", "conversation", "chat_memory.sqlite3"))
    )
    rag_backend = create_rag_backend(
        backend_mode=os.getenv("CHAT_BACKEND_MODE", "hash"),
        index_path=os.getenv("CHAT_INDEX_PATH", "data/vector_store/faiss/dongchedi.index"),
        records_path=os.getenv("CHAT_RECORDS_PATH", "data/vector_store/faiss/dongchedi_records.jsonl"),
        meta_path=os.getenv("CHAT_META_PATH", "data/vector_store/faiss/dongchedi_meta.json"),
        embedding_provider=os.getenv("CHAT_EMBEDDING_PROVIDER", "fastembed"),
        embedding_model=os.getenv("CHAT_EMBEDDING_MODEL") or None,
        embedding_api_base=os.getenv("CHAT_EMBEDDING_API_BASE") or None,
        embedding_api_key=os.getenv("CHAT_EMBEDDING_API_KEY") or None,
        llm_provider=os.getenv("CHAT_LLM_PROVIDER", "ollama"),
        llm_model=os.getenv("CHAT_LLM_MODEL") or None,
        llm_api_base=os.getenv("CHAT_LLM_API_BASE") or None,
        llm_api_key=os.getenv("CHAT_LLM_API_KEY") or None,
        retrieval_mode=os.getenv("CHAT_RETRIEVAL_MODE", "hybrid"),
        top_k=int(os.getenv("CHAT_TOP_K", "6")),
        use_llm=os.getenv("CHAT_USE_LLM", "false").lower() in {"1", "true", "yes", "on"},
    )
    return MultiTurnChatService(store=store, rag_backend=rag_backend)


def create_app(
    service: MultiTurnChatService | None = None,
    backend_factory: Any | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.chat_service = service or _build_service_from_env()
        app.state.default_runtime_config = _get_default_runtime_config()
        app.state.backend_factory = backend_factory
        yield

    app = FastAPI(
        title="Crawl_cc Multi-turn Chat API",
        version="0.1.0",
        description="多轮对话在线 API，支持会话创建、聊天、查会话和删会话。",
        lifespan=lifespan,
    )

    def get_service() -> MultiTurnChatService:
        return app.state.chat_service

    def get_runtime_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        config = dict(app.state.default_runtime_config)
        for key, value in (overrides or {}).items():
            if value is None:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            config[key] = value
        return config

    def build_chat_service_with_runtime(overrides: dict[str, Any] | None = None) -> MultiTurnChatService:
        base_service = get_service()
        runtime = get_runtime_config(overrides)
        if app.state.backend_factory is not None:
            rag_backend = app.state.backend_factory(runtime)
        else:
            rag_backend = create_rag_backend(
                backend_mode=str(runtime.get("backend_mode", "hash")),
                index_path=os.getenv("CHAT_INDEX_PATH", "data/vector_store/faiss/dongchedi.index"),
                records_path=os.getenv("CHAT_RECORDS_PATH", "data/vector_store/faiss/dongchedi_records.jsonl"),
                meta_path=os.getenv("CHAT_META_PATH", "data/vector_store/faiss/dongchedi_meta.json"),
                embedding_provider=str(runtime.get("embedding_provider", "fastembed")),
                embedding_model=(runtime.get("embedding_model") or None),
                embedding_api_base=(runtime.get("embedding_api_base") or None),
                embedding_api_key=os.getenv("CHAT_EMBEDDING_API_KEY") or None,
                llm_provider=str(runtime.get("llm_provider", "ollama")),
                llm_model=(runtime.get("llm_model") or None),
                llm_api_base=(runtime.get("llm_api_base") or None),
                llm_api_key=(runtime.get("llm_api_key") or os.getenv("CHAT_LLM_API_KEY") or None),
                retrieval_mode=str(runtime.get("retrieval_mode", "hybrid")),
                top_k=int(runtime.get("top_k", 6)),
                use_llm=bool(runtime.get("use_llm", False)),
            )
        return MultiTurnChatService(store=base_service.store, rag_backend=rag_backend)

    if WEB_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        if WEB_DIR.exists():
            return RedirectResponse(url="/ui/")
        return RedirectResponse(url="/docs")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "service": "multi_turn_chat_api"}

    @app.get("/api/v1/runtime-config")
    async def runtime_config() -> dict[str, Any]:
        return {"default_runtime_config": _sanitize_runtime_config(get_runtime_config())}

    @app.get("/app", include_in_schema=False)
    async def app_page() -> FileResponse:
        if not WEB_DIR.exists():
            raise HTTPException(status_code=404, detail="frontend not found")
        return FileResponse(WEB_DIR / "index.html")

    @app.post("/api/v1/sessions", response_model=SessionSummary)
    async def create_session(request: CreateSessionRequest) -> SessionSummary:
        chat_service = get_service()
        session_id = chat_service.start_session(user_id=request.user_id, title=request.title)
        session = chat_service.store.export_session(session_id)
        return SessionSummary.model_validate(session)

    @app.get("/api/v1/sessions", response_model=list[SessionSummary])
    async def list_sessions(limit: int = Query(default=20, ge=1, le=200)) -> list[SessionSummary]:
        chat_service = get_service()
        return [SessionSummary.model_validate(item) for item in chat_service.store.list_sessions(limit=limit)]

    @app.get("/api/v1/sessions/{session_id}", response_model=SessionDetail)
    async def get_session(session_id: str) -> SessionDetail:
        chat_service = get_service()
        try:
            session = chat_service.store.export_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SessionDetail.model_validate(session)

    @app.delete("/api/v1/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict[str, Any]:
        chat_service = get_service()
        try:
            chat_service.store.export_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        chat_service.store.delete_session(session_id)
        return {"ok": True, "deleted_session_id": session_id}

    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        started_at = time.perf_counter()
        trace_id = uuid.uuid4().hex[:12]
        stage = "prepare_runtime"
        runtime_config_used = get_runtime_config(request.runtime_config)
        sanitized_runtime_config = _sanitize_runtime_config(runtime_config_used)
        logger.info(
            "chat_request_started trace_id=%s session_id=%s user_id=%s runtime=%s message_preview=%r",
            trace_id,
            request.session_id or "<new>",
            request.user_id or "",
            sanitized_runtime_config,
            request.message[:80],
        )
        try:
            stage = "build_chat_service"
            logger.info(
                "chat_request_building_service trace_id=%s backend_mode=%s use_llm=%s retrieval_mode=%s",
                trace_id,
                sanitized_runtime_config.get("backend_mode", "hash"),
                sanitized_runtime_config.get("use_llm", False),
                sanitized_runtime_config.get("retrieval_mode", "hybrid"),
            )
            chat_service = build_chat_service_with_runtime(request.runtime_config)
            stage = "execute_chat"
            payload = chat_service.chat(
                message=request.message,
                session_id=request.session_id,
                user_id=request.user_id,
            )
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info(
                "chat_request_finished trace_id=%s session_id=%s turn_id=%s route=%s should_clarify=%s hits=%s latency_ms=%s runtime=%s",
                trace_id,
                payload.session_id,
                payload.turn_id,
                payload.route,
                payload.should_clarify,
                len(payload.hits),
                latency_ms,
                sanitized_runtime_config,
            )
            return ChatResponse(
                session_id=payload.session_id,
                turn_id=payload.turn_id,
                route=payload.route,
                rewritten_query=payload.rewritten_query,
                should_clarify=payload.should_clarify,
                clarification_question=payload.clarification_question,
                memory_summary=payload.memory_summary,
                preferences=payload.preferences.model_dump(),
                task_memory=payload.task_memory.model_dump(),
                hits=[item.model_dump() for item in payload.hits],
                runtime_config_used=sanitized_runtime_config,
                answer=payload.answer,
            )
        except HTTPException:
            raise
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            logger.exception(
                "chat_request_failed trace_id=%s stage=%s session_id=%s user_id=%s latency_ms=%s runtime=%s error=%s",
                trace_id,
                stage,
                request.session_id or "<new>",
                request.user_id or "",
                latency_ms,
                sanitized_runtime_config,
                exc,
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "message": str(exc) or type(exc).__name__,
                    "trace_id": trace_id,
                    "stage": stage,
                },
            ) from exc

    @app.get("/api/v1/overview")
    async def overview() -> dict[str, Any]:
        chat_service = get_service()
        session_count = len(chat_service.store.list_session_ids())
        return get_overview_payload(session_count=session_count)

    @app.get("/api/v1/modules/{module_name}")
    async def module_detail(module_name: str) -> dict[str, Any]:
        chat_service = get_service()
        mapping = {
            "reports": get_reports_summary,
            "assets": get_assets_summary,
            "evaluation": get_evaluation_summary,
            "feedback": get_feedback_summary,
            "workflow": get_workflow_summary,
            "sessions": lambda: {"sessions": chat_service.store.list_sessions(limit=100)},
        }
        loader = mapping.get(module_name)
        if loader is None:
            raise HTTPException(status_code=404, detail=f"unknown module: {module_name}")
        return loader()

    return app


app = create_app()
