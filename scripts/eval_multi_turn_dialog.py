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


def load_cases(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_case(service: MultiTurnChatService, case: dict, user_id: str) -> tuple[bool, dict]:
    payload = None
    session_id = None
    for turn in case["turns"]:
        payload = service.chat(message=turn, session_id=session_id, user_id=user_id)
        session_id = payload.session_id

    assert payload is not None
    expect = case["expect"]
    failures = []
    if payload.route != expect.get("final_route"):
        failures.append(f"route={payload.route} expected={expect.get('final_route')}")
    if payload.should_clarify != expect.get("should_clarify"):
        failures.append(
            f"should_clarify={payload.should_clarify} expected={expect.get('should_clarify')}"
        )
    for fragment in expect.get("answer_contains", []):
        if fragment not in payload.answer:
            failures.append(f"answer_missing={fragment}")
    if "rewritten_query" in expect and payload.rewritten_query != expect["rewritten_query"]:
        failures.append("rewritten_query_mismatch")
    if "preference_energy_type" in expect and payload.preferences.energy_type != expect["preference_energy_type"]:
        failures.append("preference_energy_type_mismatch")
    if "preference_car_type" in expect and payload.preferences.car_type != expect["preference_car_type"]:
        failures.append("preference_car_type_mismatch")
    return (
        len(failures) == 0,
        {
            "case_id": case["case_id"],
            "session_id": payload.session_id,
            "route": payload.route,
            "should_clarify": payload.should_clarify,
            "rewritten_query": payload.rewritten_query,
            "answer": payload.answer,
            "failures": failures,
        },
    )


def main() -> None:
    cases_path = REPO_ROOT / "data" / "eval" / "multi_turn_dialog_eval.jsonl"
    cases = load_cases(cases_path)
    temp_dir = Path(tempfile.mkdtemp(prefix="multi-turn-eval-"))
    try:
        memory_db = temp_dir / "chat_memory.sqlite3"
        service = MultiTurnChatService(
            store=ConversationMemoryStore(db_path=str(memory_db)),
            rag_backend=FakeRAGBackend(),
        )
        results = []
        passed = 0
        for idx, case in enumerate(cases, start=1):
            ok, result = run_case(service, case, user_id=f"eval-user-{idx}")
            result["passed"] = ok
            results.append(result)
            if ok:
                passed += 1
        output = {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": (passed / len(results)) if results else 0.0,
            "results": results,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        if passed != len(results):
            raise SystemExit(1)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
