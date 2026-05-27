from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient

from conversation.api import create_app
from conversation.service import MultiTurnChatService
from memory.store import ConversationMemoryStore


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


class ConfigurableFakeRAGBackend(FakeRAGBackend):
    def __init__(self, runtime_config: dict):
        self.runtime_config = runtime_config

    def ask(self, question: str) -> dict:
        result = super().ask(question)
        mode = "llm_on" if self.runtime_config.get("use_llm") else "llm_off"
        result["answer"] = f"[{mode}] " + result["answer"]
        return result


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="multi-turn-api-test-"))
    try:
        memory_db = temp_dir / "chat_memory.sqlite3"
        captured_runtime_configs: list[dict] = []
        service = MultiTurnChatService(
            store=ConversationMemoryStore(db_path=str(memory_db)),
            rag_backend=FakeRAGBackend(),
        )
        app = create_app(
            service=service,
            backend_factory=lambda runtime: captured_runtime_configs.append(dict(runtime)) or ConfigurableFakeRAGBackend(runtime),
        )

        with TestClient(app) as client:
            health = client.get("/health")
            assert_true(health.status_code == 200, "health 接口失败")

            root = client.get("/", follow_redirects=False)
            assert_true(root.status_code in {302, 307}, "根路径应重定向到前端")

            overview = client.get("/api/v1/overview")
            assert_true(overview.status_code == 200, "overview 接口失败")
            overview_body = overview.json()
            assert_true("cards" in overview_body, "overview 缺少 cards")
            assert_true("modules" in overview_body, "overview 缺少 modules")

            runtime_resp = client.get("/api/v1/runtime-config")
            assert_true(runtime_resp.status_code == 200, "runtime-config 接口失败")
            assert_true("default_runtime_config" in runtime_resp.json(), "runtime-config 缺少默认配置")
            assert_true("llm_api_key" not in runtime_resp.json()["default_runtime_config"], "runtime-config 不应暴露 llm_api_key")

            create_resp = client.post("/api/v1/sessions", json={"user_id": "api-user", "title": "API测试"})
            assert_true(create_resp.status_code == 200, "创建会话失败")
            session_id = create_resp.json()["session_id"]

            chat1 = client.post(
                "/api/v1/chat",
                json={"session_id": session_id, "user_id": "api-user", "message": "推荐20万内的插混SUV，重点看空间和油耗"},
            )
            assert_true(chat1.status_code == 200, "首轮聊天失败")
            chat1_body = chat1.json()
            assert_true(chat1_body["route"] == "recommend", "首轮 route 不正确")
            assert_true(chat1_body["should_clarify"] is False, "首轮不应澄清")
            assert_true(chat1_body["runtime_config_used"]["use_llm"] is False, "默认应走 llm_off 分支")

            chat2 = client.post(
                "/api/v1/chat",
                json={
                    "session_id": session_id,
                    "user_id": "api-user",
                    "message": "它的价格怎么样",
                    "runtime_config": {
                        "use_llm": True,
                        "llm_provider": "ollama",
                        "llm_model": "qwen2.5:3b",
                        "llm_api_base": "http://localhost:11434",
                        "llm_api_key": "test-secret-key",
                    },
                },
            )
            assert_true(chat2.status_code == 200, "第二轮聊天失败")
            chat2_body = chat2.json()
            assert_true(chat2_body["route"] == "follow_up", "第二轮 route 不正确")
            assert_true(chat2_body["rewritten_query"].startswith("宋PLUS DM-i"), "第二轮改写失败")
            assert_true(chat2_body["runtime_config_used"]["use_llm"] is True, "运行时切换 LLM 未生效")
            assert_true(chat2_body["runtime_config_used"]["llm_model"] == "qwen2.5:3b", "LLM 模型透传失败")
            assert_true("llm_api_key" not in chat2_body["runtime_config_used"], "响应不应回显 llm_api_key")
            assert_true(any(item.get("llm_api_key") == "test-secret-key" for item in captured_runtime_configs), "llm_api_key 未传入运行时后端")

            session_resp = client.get(f"/api/v1/sessions/{session_id}")
            assert_true(session_resp.status_code == 200, "获取会话详情失败")
            session_body = session_resp.json()
            assert_true(len(session_body["messages"]) == 4, "两轮聊天后消息数应为 4")
            assert_true(session_body["preference"]["energy_type"] == "插混", "偏好记忆未保留")

            list_resp = client.get("/api/v1/sessions?limit=10")
            assert_true(list_resp.status_code == 200, "列会话失败")
            assert_true(any(item["session_id"] == session_id for item in list_resp.json()), "会话列表缺少新会话")

            sessions_module = client.get("/api/v1/modules/sessions")
            assert_true(sessions_module.status_code == 200, "sessions 模块接口失败")
            assert_true(any(item["session_id"] == session_id for item in sessions_module.json()["sessions"]), "sessions 模块缺少会话")

            delete_resp = client.delete(f"/api/v1/sessions/{session_id}")
            assert_true(delete_resp.status_code == 200, "删除会话失败")

            missing_resp = client.get(f"/api/v1/sessions/{session_id}")
            assert_true(missing_resp.status_code == 404, "删除后应返回 404")

        print('{"ok": true}')
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
