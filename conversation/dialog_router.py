from __future__ import annotations

import re

from conversation.types import DialogRoute, TaskMemory, UserPreference


COMPARE_KEYWORDS = ("对比", "比较", "区别", "哪个好")
RECOMMEND_KEYWORDS = ("推荐", "怎么选", "选哪", "买哪", "同价位", "适合我")
FOLLOW_UP_MARKERS = ("它", "这款", "那款", "这台", "那台", "这个", "那个")


def classify_dialog_route(message: str, preference: UserPreference, task_memory: TaskMemory) -> DialogRoute:
    text = (message or "").strip()
    if not text:
        return "clarify"

    compare_like = (
        any(keyword in text for keyword in COMPARE_KEYWORDS)
        or bool(re.search(r"(?:和|跟)[^，。！？?]{1,20}?(?:比|对比|比较)", text))
    )
    if compare_like:
        return "compare"

    if any(keyword in text for keyword in RECOMMEND_KEYWORDS):
        return "recommend"

    if any(marker in text for marker in FOLLOW_UP_MARKERS) and (
        task_memory.current_focus_series or task_memory.candidate_series
    ):
        return "follow_up"

    return "fact_qa"


def build_clarification_question(route: DialogRoute, preference: UserPreference) -> str:
    if route != "recommend":
        return ""

    missing = []
    if preference.budget_min is None and preference.budget_max is None:
        missing.append("预算")
    if not preference.energy_type:
        missing.append("能源类型")
    if not preference.car_type:
        missing.append("车型类别")

    if not missing:
        return ""
    fields = "、".join(missing[:3])
    return f"为了给你更准确地推荐，先补充一下{fields}，比如“20万内、插混、SUV”。"


def extract_compare_target(message: str) -> str:
    text = (message or "").strip()
    patterns = [
        r"(?:和|跟)([^，。！？?]{1,20}?)(?:比|对比|比较)",
        r"([^，。！？?]{1,20}?)和[^，。！？?]{1,20}?(?:比|对比|比较)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ""
