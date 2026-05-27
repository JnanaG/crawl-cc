from __future__ import annotations

from dataclasses import dataclass

from conversation.dialog_router import extract_compare_target
from conversation.types import DialogRoute, TaskMemory, UserPreference


@dataclass
class ClarificationDecision:
    should_clarify: bool
    question: str = ""
    reason: str = ""
    missing_slots: list[str] | None = None


def identify_missing_recommend_slots(preference: UserPreference) -> list[str]:
    missing = []
    if preference.budget_min is None and preference.budget_max is None:
        missing.append("预算")
    if not preference.energy_type:
        missing.append("能源类型")
    if not preference.car_type:
        missing.append("车型类别")
    return missing


def build_clarification_decision(
    *,
    route: DialogRoute,
    preference: UserPreference,
    task_memory: TaskMemory,
    message: str,
) -> ClarificationDecision:
    if route == "recommend":
        missing = identify_missing_recommend_slots(preference)
        if len(missing) >= 2:
            fields = "、".join(missing[:3])
            return ClarificationDecision(
                should_clarify=True,
                question=f"为了给你更准确地推荐，先补充一下{fields}，比如“20万内、插混、SUV”。",
                reason="recommend_slots_missing",
                missing_slots=missing,
            )
        return ClarificationDecision(False, missing_slots=missing)

    if route == "compare":
        target = extract_compare_target(message)
        has_focus = bool(task_memory.current_focus_series or task_memory.candidate_series)
        if not has_focus and not target:
            return ClarificationDecision(
                should_clarify=True,
                question="你想比较哪两款车？可以直接说“宋PLUS DM-i 和 银河L7 对比”。",
                reason="compare_subject_missing",
                missing_slots=["比较对象"],
            )
        return ClarificationDecision(False)

    if route == "follow_up" and not task_memory.current_focus_series:
        return ClarificationDecision(
            should_clarify=True,
            question="你现在是想继续问哪款车？直接说车型名，我再接着回答。",
            reason="follow_up_focus_missing",
            missing_slots=["当前焦点车型"],
        )

    return ClarificationDecision(False)
