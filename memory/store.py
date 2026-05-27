from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable

from conversation.types import Message, TaskMemory, UserPreference


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationMemoryStore:
    def __init__(self, db_path: str = os.path.join("data", "conversation", "chat_memory.sqlite3")):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_created
                ON messages(session_id, created_at);

                CREATE TABLE IF NOT EXISTS preferences (
                    memory_key TEXT PRIMARY KEY,
                    user_id TEXT,
                    session_id TEXT,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS task_memory (
                    session_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def create_session(self, user_id: str | None = None, title: str = "") -> str:
        session_id = str(uuid.uuid4())
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, user_id or "", title, now, now),
            )
        return session_id

    def ensure_session(self, session_id: str | None = None, user_id: str | None = None, title: str = "") -> str:
        if session_id:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT session_id FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if row:
                    return session_id
        return self.create_session(user_id=user_id, title=title)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        message_id = str(uuid.uuid4())
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages(message_id, session_id, role, content, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id, session_id, role, content, now, json.dumps(metadata or {}, ensure_ascii=False)),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
        return message_id

    def list_messages(self, session_id: str, limit: int = 10) -> list[Message]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at, metadata_json
                FROM messages
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        messages = []
        for row in reversed(rows):
            messages.append(
                Message(
                    role=row["role"],
                    content=row["content"],
                    created_at=row["created_at"],
                    metadata=json.loads(row["metadata_json"] or "{}"),
                )
            )
        return messages

    def build_recent_summary(self, session_id: str, limit: int = 6, max_chars: int = 500) -> str:
        messages = self.list_messages(session_id, limit=limit)
        parts = []
        used = 0
        for msg in messages:
            prefix = "用户" if msg.role == "user" else "助手"
            line = f"{prefix}: {msg.content.strip()}"
            if used + len(line) > max_chars:
                break
            parts.append(line)
            used += len(line)
        return "\n".join(parts)

    def _memory_key(self, session_id: str, user_id: str | None = None) -> str:
        return f"user:{user_id}" if user_id else f"session:{session_id}"

    def get_preference(self, session_id: str, user_id: str | None = None) -> UserPreference:
        key = self._memory_key(session_id, user_id=user_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM preferences WHERE memory_key = ?",
                (key,),
            ).fetchone()
        if not row:
            return UserPreference()
        return UserPreference.model_validate(json.loads(row["payload_json"]))

    def upsert_preference(self, session_id: str, preference: UserPreference, user_id: str | None = None) -> None:
        key = self._memory_key(session_id, user_id=user_id)
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO preferences(memory_key, user_id, session_id, payload_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(memory_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at,
                    user_id = excluded.user_id,
                    session_id = excluded.session_id
                """,
                (
                    key,
                    user_id or "",
                    session_id,
                    json.dumps(preference.model_dump(), ensure_ascii=False),
                    now,
                ),
            )

    def get_task_memory(self, session_id: str) -> TaskMemory:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM task_memory WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return TaskMemory()
        return TaskMemory.model_validate(json.loads(row["payload_json"]))

    def upsert_task_memory(self, session_id: str, task_memory: TaskMemory) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO task_memory(session_id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (session_id, json.dumps(task_memory.model_dump(), ensure_ascii=False), now),
            )

    def export_session(self, session_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id, title, created_at, updated_at FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            raise KeyError(f"session not found: {session_id}")
        user_id = row["user_id"] if row and row["user_id"] else None
        return {
            "session_id": session_id,
            "user_id": user_id or "",
            "title": row["title"] or "",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "messages": [msg.model_dump() for msg in self.list_messages(session_id, limit=200)],
            "preference": self.get_preference(session_id, user_id=user_id).model_dump(),
            "task_memory": self.get_task_memory(session_id).model_dump(),
        }

    def delete_session(self, session_id: str) -> None:
        keys = [f"session:{session_id}"]
        with self._connect() as conn:
            user_rows = conn.execute(
                "SELECT DISTINCT user_id FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchall()
            for row in user_rows:
                if row["user_id"]:
                    keys.append(f"user:{row['user_id']}")
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM task_memory WHERE session_id = ?", (session_id,))
            conn.executemany("DELETE FROM preferences WHERE memory_key = ?", ((key,) for key in keys))

    def list_session_ids(self) -> Iterable[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT session_id FROM sessions ORDER BY updated_at DESC").fetchall()
        return [row["session_id"] for row in rows]

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, user_id, title, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "user_id": row["user_id"] or "",
                "title": row["title"] or "",
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
