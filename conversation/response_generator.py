from __future__ import annotations

from conversation.types import RetrievedHit, TaskMemory, UserPreference


def _format_hit_brief(hit: RetrievedHit) -> str:
    snippet = (hit.text_snippet or "").replace("\n", " ").strip()
    return f"{hit.title or '未知车系'}: {snippet[:90]}"


def _preference_summary(preference: UserPreference) -> str:
    parts = []
    if preference.budget_text:
        parts.append(f"预算 {preference.budget_text}")
    if preference.energy_type:
        parts.append(f"偏好 {preference.energy_type}")
    if preference.car_type:
        parts.append(f"车型 {preference.car_type}")
    if preference.focus_points:
        parts.append(f"重点关注 {'/'.join(preference.focus_points[:3])}")
    return "，".join(parts)


def render_dialog_answer(
    *,
    route: str,
    original_question: str,
    rewritten_query: str,
    preference: UserPreference,
    task_memory: TaskMemory,
    hits: list[RetrievedHit],
    backend_answer: str,
) -> str:
    if not hits:
        return backend_answer.strip() or "根据当前检索内容无法确认。"

    if route == "recommend":
        pref_text = _preference_summary(preference)
        lines = []
        if pref_text:
            lines.append(f"基于你当前给出的条件，我先按“{pref_text}”来筛选。")
        else:
            lines.append("基于当前检索结果，我先给你一版候选建议。")
        lines.append("优先可以先看这几款：")
        for idx, hit in enumerate(hits[:3], start=1):
            lines.append(f"{idx}. {_format_hit_brief(hit)}")
        if len(hits) >= 2:
            lines.append(f"如果你愿意，我下一步可以继续把“{hits[0].title}”和“{hits[1].title}”做针对性对比。")
        else:
            lines.append("如果你愿意，我下一步可以继续展开价格、配置和空间表现。")
        return "\n".join(lines)

    if route == "compare":
        left = task_memory.current_focus_series or (hits[0].title if hits else "第一款")
        right = hits[1].title if len(hits) > 1 else (hits[0].title if hits else "第二款")
        lines = [f"围绕“{rewritten_query}”，当前检索结果里最值得先比较的是：{left} 和 {right}。"]
        for idx, hit in enumerate(hits[:2], start=1):
            lines.append(f"{idx}. {_format_hit_brief(hit)}")
        lines.append("如果你告诉我更关注价格、空间、油耗、配置还是动力，我可以继续按维度细化。")
        return "\n".join(lines)

    if route == "follow_up":
        focus = task_memory.current_focus_series or hits[0].title
        lines = [f"你这轮是在继续追问“{focus}”，我按当前上下文接着回答。", backend_answer.strip()]
        return "\n".join(line for line in lines if line.strip())

    return backend_answer.strip() or "根据当前检索内容无法确认。"
