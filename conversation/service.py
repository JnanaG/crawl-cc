from __future__ import annotations

import logging
import time
from typing import Any

from conversation.context_builder import build_memory_summary
from conversation.clarification import build_clarification_decision
from conversation.dialog_router import classify_dialog_route
from conversation.query_rewriter import merge_preferences, rewrite_query
from conversation.rag_backend import RAGBackend
from conversation.response_generator import render_dialog_answer
from conversation.types import Message, ResponsePayload, RetrievedHit, TaskMemory
from memory.store import ConversationMemoryStore

logger = logging.getLogger(__name__)


def _dedupe_titles(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        item = (value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


class MultiTurnChatService:
    def __init__(self, store: ConversationMemoryStore, rag_backend: RAGBackend):
        self.store = store
        self.rag_backend = rag_backend

    def start_session(self, user_id: str | None = None, title: str = "") -> str:
        return self.store.create_session(user_id=user_id, title=title)

    def chat(self, *, message: str, session_id: str | None = None, user_id: str | None = None) -> ResponsePayload:
        started_at = time.perf_counter()
        session_id = self.store.ensure_session(session_id=session_id, user_id=user_id)
        self.store.add_message(session_id, role="user", content=message, metadata={"user_id": user_id or ""})
        recent_messages = self.store.list_messages(session_id, limit=8)
        preference = merge_preferences(self.store.get_preference(session_id, user_id=user_id), message)
        self.store.upsert_preference(session_id, preference, user_id=user_id)
        task_memory = self.store.get_task_memory(session_id)

        route = classify_dialog_route(message, preference, task_memory)
        rewritten_query, rewrite_reasons = rewrite_query(message, recent_messages, preference, task_memory)
        memory_summary = build_memory_summary(recent_messages, preference, task_memory)
        backend_runtime = self._describe_backend_runtime()
        logger.info(
            "chat_pipeline_prepared session_id=%s route=%s backend=%s rewrite_reasons=%s preference=%s task_stage=%s",
            session_id,
            route,
            backend_runtime,
            rewrite_reasons,
            preference.model_dump(),
            task_memory.stage,
        )
        clarification = build_clarification_decision(
            route=route,
            preference=preference,
            task_memory=task_memory,
            message=message,
        )

        if clarification.should_clarify:
            logger.info(
                "chat_clarification_required session_id=%s route=%s reason=%s question=%r",
                session_id,
                route,
                clarification.reason,
                clarification.question,
            )
            answer = clarification.question
            hits: list[RetrievedHit] = []
            should_clarify = True
        else:
            logger.info(
                "chat_rag_started session_id=%s route=%s rewritten_query=%r backend=%s",
                session_id,
                route,
                rewritten_query,
                backend_runtime,
            )
            rag_result = self.rag_backend.ask(rewritten_query)
            hits = [
                RetrievedHit.model_validate(hit)
                for hit in rag_result.get("hits", [])
            ]
            logger.info(
                "chat_rag_finished session_id=%s route=%s hits=%s backend_answer_preview=%r",
                session_id,
                route,
                len(hits),
                (rag_result.get("answer", "") or "")[:120],
            )
            answer = render_dialog_answer(
                route=route,
                original_question=message,
                rewritten_query=rewritten_query,
                preference=preference,
                task_memory=task_memory,
                hits=hits,
                backend_answer=rag_result.get("answer", "").strip(),
            )
            should_clarify = False

        next_task_memory = self._update_task_memory(
            previous=task_memory,
            route=route,
            rewritten_query=rewritten_query,
            hits=hits,
            user_message=message,
        )
        self.store.upsert_task_memory(session_id, next_task_memory)

        assistant_message_id = self.store.add_message(
            session_id,
            role="assistant",
            content=answer,
            metadata={
                "route": route,
                "rewritten_query": rewritten_query,
                "rewrite_reasons": rewrite_reasons,
                "clarification": clarification.question,
                "clarification_reason": clarification.reason,
                "hit_titles": [hit.title for hit in hits[:4]],
            },
        )
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "chat_pipeline_finished session_id=%s turn_id=%s route=%s current_focus=%s candidates=%s latency_ms=%s",
            session_id,
            assistant_message_id,
            route,
            next_task_memory.current_focus_series,
            next_task_memory.candidate_series,
            latency_ms,
        )

        return ResponsePayload(
            session_id=session_id,
            turn_id=assistant_message_id,
            route=route,
            rewritten_query=rewritten_query,
            answer=answer,
            memory_summary=memory_summary,
            preferences=preference,
            task_memory=next_task_memory,
            hits=hits,
            should_clarify=should_clarify,
            clarification_question=clarification.question,
        )

    def _describe_backend_runtime(self) -> dict[str, Any]:
        describe = getattr(self.rag_backend, "describe_runtime", None)
        if callable(describe):
            try:
                return describe()
            except Exception:
                return {"backend": type(self.rag_backend).__name__, "describe_error": True}
        return {"backend": type(self.rag_backend).__name__}

    def _update_task_memory(
        self,
        *,
        previous: TaskMemory,
        route: str,
        rewritten_query: str,
        hits: list[RetrievedHit],
        user_message: str,
    ) -> TaskMemory:
        task_memory = previous.model_copy(deep=True)
        task_memory.task_type = route
        task_memory.last_rewritten_query = rewritten_query
        if route == "recommend":
            task_memory.stage = "collecting_preferences"
        elif route == "compare":
            task_memory.stage = "comparing_candidates"
        elif route == "follow_up":
            task_memory.stage = "follow_up_qa"
        elif route == "clarify":
            task_memory.stage = "clarifying_requirements"
        else:
            task_memory.stage = "fact_lookup"

        hit_titles = [hit.title for hit in hits if hit.title]
        merged = _dedupe_titles(task_memory.candidate_series + hit_titles)
        task_memory.candidate_series = merged[:5]
        if route == "follow_up" and previous.current_focus_series:
            task_memory.current_focus_series = previous.current_focus_series
        elif hit_titles:
            task_memory.current_focus_series = hit_titles[0]

        notes = list(task_memory.notes)
        user_note = f"latest_user_message={user_message[:80]}"
        if not notes or notes[-1] != user_note:
            notes.append(user_note)
        task_memory.notes = notes[-6:]
        return task_memory
