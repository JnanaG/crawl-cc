from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


DialogRoute = Literal["fact_qa", "compare", "recommend", "follow_up", "clarify"]


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserPreference(BaseModel):
    budget_text: str = ""
    budget_min: float | None = None
    budget_max: float | None = None
    energy_type: str = ""
    car_type: str = ""
    brand_preference: list[str] = Field(default_factory=list)
    focus_points: list[str] = Field(default_factory=list)


class TaskMemory(BaseModel):
    task_type: str = ""
    stage: str = ""
    current_focus_series: str = ""
    candidate_series: list[str] = Field(default_factory=list)
    last_rewritten_query: str = ""
    notes: list[str] = Field(default_factory=list)


class RetrievedHit(BaseModel):
    title: str = ""
    url: str = ""
    series_id: str = ""
    text_snippet: str = ""
    score: float = 0.0


class RetrievedContext(BaseModel):
    original_query: str
    rewritten_query: str
    route: DialogRoute
    memory_summary: str = ""
    hits: list[RetrievedHit] = Field(default_factory=list)


class ResponsePayload(BaseModel):
    session_id: str
    turn_id: str
    route: DialogRoute
    rewritten_query: str
    answer: str
    memory_summary: str = ""
    preferences: UserPreference = Field(default_factory=UserPreference)
    task_memory: TaskMemory = Field(default_factory=TaskMemory)
    hits: list[RetrievedHit] = Field(default_factory=list)
    should_clarify: bool = False
    clarification_question: str = ""
