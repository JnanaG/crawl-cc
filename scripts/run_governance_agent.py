from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_pipeline import GovernanceOrchestrator
from cleaner.dcd_cleaner import DongchediCleaner


class DemoSemanticStore:
    def __init__(self):
        self.similarity_threshold = 0.94
        self.same_series_threshold = 0.88

    def find_candidates(self, clean_record: dict, markdown_text: str) -> dict:
        return {
            "query_text": markdown_text or "",
            "query_dim": 0,
            "hits": [],
        }

    def add_record(self, clean_record: dict, markdown_text: str, metadata: dict) -> None:
        return


def find_default_input() -> Path:
    cleaned_dir = REPO_ROOT / "data" / "cleaned" / "dongchedi" / "json"
    candidates = sorted(cleaned_dir.glob("series_*.json"))
    if not candidates:
        raise FileNotFoundError(f"未找到 cleaned JSON: {cleaned_dir}")
    return candidates[0]


def load_clean_record(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_parser(force_enable_llm: bool | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行单条治理 Agent，支持启用或关闭 LLM Agent")
    parser.add_argument("--input-clean-json", default=None, help="cleaned JSON 输入路径，默认取 data/cleaned/dongchedi/json 下第一条")
    parser.add_argument("--base-dir", default=str(REPO_ROOT / "data" / "agent_runs"), help="治理产物输出目录")
    parser.add_argument("--batch-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output", default=None, help="将结果写入指定 JSON 文件")
    if force_enable_llm is None:
        parser.add_argument("--enable-llm-agents", action="store_true", help="启用 LLM 驱动 route/quality agent")
    parser.add_argument("--llm-provider", default="ollama", choices=["openai_compatible", "ollama"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-base", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-fail-open", action="store_true", default=True, help="LLM 调用失败时回退规则结果")
    parser.add_argument("--semantic-store-mode", default="real", choices=["real", "stub"], help="单条脚本使用真实或 stub 语义去重存储")
    return parser


def run_cli(force_enable_llm: bool | None = None) -> int:
    args = build_parser(force_enable_llm=force_enable_llm).parse_args()
    input_path = Path(args.input_clean_json).resolve() if args.input_clean_json else find_default_input().resolve()
    clean_record = load_clean_record(input_path)
    cleaner = DongchediCleaner()
    markdown_text = cleaner.clean_record_to_markdown(clean_record)

    enable_llm = force_enable_llm if force_enable_llm is not None else bool(args.enable_llm_agents)
    semantic_store = DemoSemanticStore() if args.semantic_store_mode == "stub" else None
    orchestrator = GovernanceOrchestrator(
        base_dir=str(Path(args.base_dir).resolve()),
        enable_llm_agents=enable_llm,
        agent_llm_provider=args.llm_provider,
        agent_llm_model=args.llm_model,
        agent_llm_api_base=args.llm_api_base,
        agent_llm_api_key=args.llm_api_key,
        llm_fail_open=bool(args.llm_fail_open),
        semantic_store=semantic_store,
    )
    result = orchestrator.govern(clean_record=clean_record, markdown_text=markdown_text, batch_id=args.batch_id)
    payload = {
        "mode": "llm" if enable_llm else "rule_only",
        "input_clean_json": str(input_path),
        "series_id": result.series_id,
        "decision": result.decision_result.decision,
        "route": result.route_result.model_dump(),
        "quality": result.quality_result.model_dump(),
        "metadata": result.metadata,
        "audit_logs": result.audit_logs,
    }

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print("[PASS] 单条治理执行完成")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
