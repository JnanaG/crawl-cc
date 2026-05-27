from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conversation import MultiTurnChatService
from memory import ConversationMemoryStore


class FakeRAGBackend:
    def ask(self, question: str) -> dict:
        if "宋PLUS DM-i 和 银河L7 对比" in question:
            return {
                "answer": "宋PLUS DM-i 更偏家用均衡，银河L7 更强调动力和配置。",
                "hits": [
                    {
                        "title": "宋PLUS DM-i",
                        "url": "https://example.com/song-plus-dmi",
                        "series_id": "song-plus-dmi",
                        "score": 0.92,
                        "text_snippet": "宋PLUS DM-i 主打家用、插混、空间均衡。",
                    },
                    {
                        "title": "银河L7",
                        "url": "https://example.com/galaxy-l7",
                        "series_id": "galaxy-l7",
                        "score": 0.88,
                        "text_snippet": "银河L7 动力响应较强，配置激进。",
                    },
                ],
            }
        if "宋PLUS DM-i" in question and "价格" in question:
            return {
                "answer": "宋PLUS DM-i 当前检索到的价格区间大致在 15-20 万附近。",
                "hits": [
                    {
                        "title": "宋PLUS DM-i",
                        "url": "https://example.com/song-plus-dmi",
                        "series_id": "song-plus-dmi",
                        "score": 0.95,
                        "text_snippet": "宋PLUS DM-i 经销商报价在 15-20 万附近。",
                    }
                ],
            }
        return {
            "answer": "20万内插混 SUV 可以先看宋PLUS DM-i、银河L7。",
            "hits": [
                {
                    "title": "宋PLUS DM-i",
                    "url": "https://example.com/song-plus-dmi",
                    "series_id": "song-plus-dmi",
                    "score": 0.97,
                    "text_snippet": "宋PLUS DM-i 空间、油耗和家用属性比较均衡。",
                },
                {
                    "title": "银河L7",
                    "url": "https://example.com/galaxy-l7",
                    "series_id": "galaxy-l7",
                    "score": 0.89,
                    "text_snippet": "银河L7 适合偏动力和配置取向的用户。",
                },
            ],
        }


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="multi-turn-test-"))
    try:
        memory_db = temp_dir / "chat_memory.sqlite3"
        store = ConversationMemoryStore(db_path=str(memory_db))
        service = MultiTurnChatService(store=store, rag_backend=FakeRAGBackend())

        clarify_turn = service.chat(message="推荐一款家用车", user_id="tester-clarify")
        assert_true(clarify_turn.route == "recommend", "推荐场景应命中 recommend 路由")
        assert_true(clarify_turn.should_clarify is True, "缺槽位时应主动澄清")
        assert_true("预算" in clarify_turn.answer, "澄清问题应提示预算信息")

        turn1 = service.chat(message="推荐20万内的插混SUV，重点看空间和油耗", user_id="tester")
        assert_true(turn1.route == "recommend", "首轮应命中 recommend 路由")
        assert_true(turn1.preferences.budget_max == 20.0, "预算提取失败")
        assert_true(turn1.preferences.energy_type == "插混", "能源类型提取失败")
        assert_true(turn1.preferences.car_type == "SUV", "车型偏好提取失败")
        assert_true("空间" in turn1.preferences.focus_points, "关注点提取失败")
        assert_true(turn1.task_memory.current_focus_series == "宋PLUS DM-i", "首轮应锁定当前焦点车系")
        assert_true("优先可以先看这几款" in turn1.answer, "推荐回答应走结构化输出")

        turn2 = service.chat(message="它的价格怎么样", session_id=turn1.session_id, user_id="tester")
        assert_true(turn2.route == "follow_up", "第二轮应命中 follow_up 路由")
        assert_true(turn2.rewritten_query.startswith("宋PLUS DM-i"), "代词改写未补齐焦点车系")
        assert_true("价格" in turn2.rewritten_query, "第二轮改写未保留价格意图")
        assert_true("继续追问" in turn2.answer, "追问场景应提示接续上下文")

        turn3 = service.chat(message="那和银河L7比呢", session_id=turn1.session_id, user_id="tester")
        assert_true(turn3.route == "compare", "第三轮应命中 compare 路由")
        assert_true(turn3.rewritten_query == "宋PLUS DM-i 和 银河L7 对比", "比较问题改写不符合预期")
        assert_true("银河L7" in turn3.task_memory.candidate_series, "候选车系未进入任务记忆")
        assert_true("最值得先比较的是" in turn3.answer, "比较回答应走结构化输出")

        exported = store.export_session(turn1.session_id)
        assert_true(len(exported["messages"]) == 6, "三轮对话应落 6 条消息")
        assert_true(exported["preference"]["energy_type"] == "插混", "偏好记忆未持久化")
        assert_true(exported["task_memory"]["current_focus_series"] == "宋PLUS DM-i", "任务记忆未持久化")

        print(json.dumps({"ok": True, "session_id": turn1.session_id}, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
