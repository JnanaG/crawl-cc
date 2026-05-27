from __future__ import annotations

import re

from conversation.dialog_router import extract_compare_target
from conversation.types import Message, TaskMemory, UserPreference


PRONOUN_PATTERNS = [
    r"它的",
    r"它\b",
    r"这款车",
    r"那款车",
    r"这台车",
    r"那台车",
    r"这款",
    r"那款",
]
ENERGY_KEYWORDS = ("纯电", "插混", "增程", "燃油", "混动")
CAR_TYPE_KEYWORDS = ("SUV", "轿车", "MPV", "越野", "旅行车", "两厢", "三厢")
BRAND_CANDIDATES = (
    "比亚迪",
    "吉利",
    "银河",
    "特斯拉",
    "理想",
    "问界",
    "小鹏",
    "蔚来",
    "丰田",
    "本田",
    "大众",
)
FOCUS_POINT_MAP = {
    "空间": "空间",
    "油耗": "油耗",
    "续航": "续航",
    "智驾": "智驾",
    "配置": "配置",
    "舒适": "舒适性",
    "动力": "动力",
    "保值": "保值率",
}


def merge_preferences(previous: UserPreference, incoming_text: str) -> UserPreference:
    updated = previous.model_copy(deep=True)
    text = (incoming_text or "").strip()
    if not text:
        return updated

    budget_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|到|至)\s*(\d+(?:\.\d+)?)\s*万", text)
    if budget_match:
        low = float(budget_match.group(1))
        high = float(budget_match.group(2))
        updated.budget_min = min(low, high)
        updated.budget_max = max(low, high)
        updated.budget_text = f"{updated.budget_min:g}-{updated.budget_max:g}万"
    else:
        single_match = re.search(r"(\d+(?:\.\d+)?)\s*万(?:内|以内|以下)?", text)
        if single_match:
            value = float(single_match.group(1))
            updated.budget_min = 0.0
            updated.budget_max = value
            updated.budget_text = f"{value:g}万内"

    for keyword in ENERGY_KEYWORDS:
        if keyword in text:
            updated.energy_type = keyword
            break

    for keyword in CAR_TYPE_KEYWORDS:
        if keyword.lower() in text.lower():
            updated.car_type = keyword.upper() if keyword.lower() == "suv" else keyword
            break

    brands = list(updated.brand_preference)
    for candidate in BRAND_CANDIDATES:
        if candidate in text and candidate not in brands:
            brands.append(candidate)
    updated.brand_preference = brands[:5]

    focus_points = list(updated.focus_points)
    for raw, normalized in FOCUS_POINT_MAP.items():
        if raw in text and normalized not in focus_points:
            focus_points.append(normalized)
    updated.focus_points = focus_points[:6]
    return updated


def rewrite_query(
    message: str,
    recent_messages: list[Message],
    preference: UserPreference,
    task_memory: TaskMemory,
) -> tuple[str, list[str]]:
    rewritten = (message or "").strip()
    reasons: list[str] = []
    focus_series = task_memory.current_focus_series or (task_memory.candidate_series[0] if task_memory.candidate_series else "")

    if focus_series:
        for pattern in PRONOUN_PATTERNS:
            if re.search(pattern, rewritten):
                rewritten = re.sub(pattern, focus_series, rewritten, count=1)
                reasons.append("resolved_pronoun_to_focus_series")
                break

    compare_target = extract_compare_target(rewritten)
    if focus_series and compare_target and focus_series not in rewritten:
        rewritten = f"{focus_series} 和 {compare_target} 对比"
        reasons.append("filled_compare_subject_from_task_memory")

    if preference.budget_text and "万" not in rewritten and any(token in rewritten for token in ("推荐", "怎么选", "适合", "同价位")):
        rewritten = f"{rewritten}，预算{preference.budget_text}"
        reasons.append("appended_budget_preference")

    if preference.energy_type and not any(keyword in rewritten for keyword in ENERGY_KEYWORDS) and any(
        token in rewritten for token in ("推荐", "怎么选", "适合", "同价位")
    ):
        rewritten = f"{rewritten}，偏向{preference.energy_type}"
        reasons.append("appended_energy_preference")

    if preference.car_type and preference.car_type.lower() not in rewritten.lower() and any(
        token in rewritten for token in ("推荐", "怎么选", "适合", "同价位")
    ):
        rewritten = f"{rewritten}，车型偏好{preference.car_type}"
        reasons.append("appended_car_type_preference")

    if preference.focus_points and "重点关注" not in rewritten and any(token in rewritten for token in ("推荐", "怎么选", "适合")):
        rewritten = f"{rewritten}，重点关注{'/'.join(preference.focus_points[:3])}"
        reasons.append("appended_focus_points")

    if not reasons and recent_messages:
        reasons.append("kept_original_query")
    return rewritten, reasons
