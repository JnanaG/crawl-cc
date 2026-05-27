from __future__ import annotations

from conversation.types import Message, RetrievedContext, RetrievedHit, TaskMemory, UserPreference


def build_memory_summary(messages: list[Message], preference: UserPreference, task_memory: TaskMemory, max_chars: int = 800) -> str:
    parts: list[str] = []
    if preference.budget_text or preference.energy_type or preference.car_type or preference.focus_points:
        pref_bits = []
        if preference.budget_text:
            pref_bits.append(f"预算={preference.budget_text}")
        if preference.energy_type:
            pref_bits.append(f"能源={preference.energy_type}")
        if preference.car_type:
            pref_bits.append(f"车型={preference.car_type}")
        if preference.focus_points:
            pref_bits.append(f"关注={','.join(preference.focus_points[:3])}")
        parts.append("用户偏好: " + " | ".join(pref_bits))

    if task_memory.current_focus_series or task_memory.candidate_series:
        task_bits = []
        if task_memory.current_focus_series:
            task_bits.append(f"当前焦点={task_memory.current_focus_series}")
        if task_memory.candidate_series:
            task_bits.append(f"候选={','.join(task_memory.candidate_series[:4])}")
        if task_memory.stage:
            task_bits.append(f"阶段={task_memory.stage}")
        parts.append("任务状态: " + " | ".join(task_bits))

    if messages:
        lines = []
        for message in messages[-4:]:
            role = "用户" if message.role == "user" else "助手"
            lines.append(f"{role}: {message.content.strip()}")
        parts.append("最近对话:\n" + "\n".join(lines))

    summary = "\n".join(parts)
    return summary[:max_chars]


def build_retrieved_context(
    *,
    original_query: str,
    rewritten_query: str,
    route: str,
    memory_summary: str,
    hits: list[RetrievedHit],
) -> RetrievedContext:
    return RetrievedContext(
        original_query=original_query,
        rewritten_query=rewritten_query,
        route=route,
        memory_summary=memory_summary,
        hits=hits,
    )
