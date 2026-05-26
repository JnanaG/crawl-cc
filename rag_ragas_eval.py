"""
Ragas 评估脚本：对 RAG 系统进行多维度质量评估。

优先使用真实 ragas 评估；当当前 Python 环境中 ragas 因依赖问题无法导入时，
会自动降级到轻量评估逻辑，保证脚本能完整跑通并输出结果文件。
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import os
import re
import sys
import time
import types
from collections import Counter
from datetime import datetime
from typing import Any, Sequence

from loguru import logger

from utils.embedding_client import EmbeddingClient
from utils.faiss_vector_store import FaissVectorStore
from utils.llm_client import LLMClient
from utils.reranker_client import RerankerClient


DEFAULT_INDEX = os.path.join("data", "vector_store", "faiss", "dongchedi.index")
DEFAULT_RECORDS = os.path.join("data", "vector_store", "faiss", "dongchedi_records.jsonl")
DEFAULT_META = os.path.join("data", "vector_store", "faiss", "dongchedi_meta.json")

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


class LightweightEvaluationResult:
    """兼容 ragas 结果接口的轻量结果对象。"""

    def __init__(self, rows: list[dict[str, Any]], backend: str, error: str | None = None):
        self.rows = rows
        self.backend = backend
        self.error = error
        self._scores_dict = {
            metric: [float(row.get(metric, 0.0)) for row in rows]
            for metric in METRIC_NAMES
        }
        self._repr_dict = {
            metric: _safe_mean(values) for metric, values in self._scores_dict.items()
        }

    def __getitem__(self, key: str) -> list[float]:
        return self._scores_dict[key]

    def to_pandas(self):
        import pandas as pd

        return pd.DataFrame(self.rows)

    def __repr__(self) -> str:
        score_strs = [f"'{key}': {value:0.4f}" for key, value in self._repr_dict.items()]
        return "{" + ", ".join(score_strs) + "}"


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    valid = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(valid) / len(valid) if valid else 0.0


def tokenize_zh_en(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower())


def lexical_similarity(left: str, right: str) -> float:
    left_tokens = Counter(tokenize_zh_en(left))
    right_tokens = Counter(tokenize_zh_en(right))
    if not left_tokens or not right_tokens:
        return 0.0

    common = set(left_tokens) & set(right_tokens)
    numerator = sum(left_tokens[token] * right_tokens[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left_tokens.values()))
    right_norm = math.sqrt(sum(value * value for value in right_tokens.values()))
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        return 0.0
    return numerator / (left_norm * right_norm)


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"[。！？!?;\n]+", text)
    return [part.strip() for part in parts if len(part.strip()) >= 4]


def load_golden_set(path: str, max_cases: int | None = None) -> list[dict[str, Any]]:
    """加载黄金测试集。"""
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if max_cases is not None and len(items) >= max_cases:
                break
    logger.info(f"加载黄金集: {len(items)} 条测试用例")
    return items


def run_rag_query(
    question: str,
    store: FaissVectorStore,
    emb_client: EmbeddingClient,
    llm_client: LLMClient,
    reranker: RerankerClient | None,
    top_k: int,
    retrieval_mode: str,
    max_context_chars: int,
) -> tuple[str, list[str]]:
    """
    执行 RAG 查询，返回 (answer, retrieved_contexts)。
    """
    from rag_llm_demo import build_context, hybrid_search

    query_vec = emb_client.embed_query(question)
    if retrieval_mode == "dense":
        hits = store.search(query_vector=query_vec, top_k=top_k)
    else:
        hits, _ = hybrid_search(
            question=question,
            query_vec=query_vec,
            store=store,
            top_k=top_k,
            dense_top_k=24,
            sparse_top_k=40,
            rrf_k=60,
            rerank_top_n=12,
            reranker=reranker,
            reranker_weight=0.65,
        )

    if not hits:
        return "未检索到相关内容", []

    context = build_context(hits, max_context_chars)
    retrieved_contexts = [hit.text for hit in hits]
    system_prompt = (
        "你是汽车知识RAG助手。"
        "必须仅基于给定上下文回答，不要编造。"
        "若上下文不足，明确说'根据当前检索内容无法确认'。"
        "回答后附上引用编号，例如[1][3]。"
    )
    user_prompt = (
        f"用户问题:\n{question}\n\n"
        f"检索上下文:\n{context}\n\n"
        "请给出简洁准确的中文答案，并保留引用编号。"
    )
    answer = llm_client.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.2,
    )
    return answer.strip(), retrieved_contexts


def prepare_eval_rows(
    golden_items: list[dict[str, Any]],
    store: FaissVectorStore,
    emb_client: EmbeddingClient,
    llm_client: LLMClient,
    reranker: RerankerClient | None,
    args: Any,
) -> list[dict[str, Any]]:
    """
    对黄金集中的每个问题执行 RAG，构造评估输入行。
    """
    rows: list[dict[str, Any]] = []
    logger.info(f"开始对 {len(golden_items)} 条测试用例执行 RAG 查询...")

    for idx, item in enumerate(golden_items, start=1):
        question = item["question"]
        expected_answer = item.get("expected_answer", "")
        ground_truth_contexts = item.get("ground_truth_contexts") or []
        logger.info(f"[{idx}/{len(golden_items)}] 查询: {question}")

        row = {
            "question": question,
            "ground_truth": expected_answer,
            "reference_contexts": ground_truth_contexts,
            "metadata": item.get("metadata", {}),
        }

        try:
            answer, contexts = run_rag_query(
                question=question,
                store=store,
                emb_client=emb_client,
                llm_client=llm_client,
                reranker=reranker,
                top_k=args.top_k,
                retrieval_mode=args.retrieval_mode,
                max_context_chars=args.max_context_chars,
            )
            row["answer"] = answer
            row["contexts"] = contexts
            row["query_error"] = None
            logger.debug(f"  答案: {answer[:100]}...")
            logger.debug(f"  检索到 {len(contexts)} 个上下文")
        except Exception as exc:
            logger.exception(f"  查询失败: {exc}")
            row["answer"] = "查询失败"
            row["contexts"] = []
            row["query_error"] = str(exc)

        rows.append(row)
        time.sleep(max(0.0, args.sleep_sec))

    return rows


def validate_retrieval_compatibility(
    store: FaissVectorStore,
    emb_client: EmbeddingClient,
) -> None:
    """校验当前 embedding 配置是否与已加载索引兼容。"""
    probe_vector = emb_client.embed_query("维度检查")
    probe_dim = len(probe_vector)
    if probe_dim <= 0:
        raise RuntimeError("embedding 客户端未返回有效向量，无法继续评估")
    if store.dim != probe_dim:
        raise RuntimeError(
            "当前 embedding 配置与已加载索引不兼容："
            f"查询向量维度={probe_dim}，索引维度={store.dim}。"
            "请使用与构建索引时一致的 embedding provider/model，"
            "或先重建向量库后再运行评估。"
        )


def _install_ragas_import_shims() -> None:
    """
    为当前环境中的 ragas 依赖问题打补丁。

    当前用户环境存在以下问题：
    1. `transformers` 导入时会继续触发 `torch` DLL 错误；
    2. `langchain_community.chat_models.vertexai` 模块不存在。

    这些依赖仅用于 ragas 的可选分支或 token 计数，不影响本项目的评估主路径。
    """

    if "langchain_core.pydantic_v1" not in sys.modules:
        try:
            importlib.import_module("langchain_core.pydantic_v1")
        except Exception:
            fake_pydantic_v1 = types.ModuleType("langchain_core.pydantic_v1")
            from pydantic.v1 import BaseModel as PydanticV1BaseModel
            from pydantic.v1 import root_validator

            fake_pydantic_v1.BaseModel = PydanticV1BaseModel
            fake_pydantic_v1.root_validator = root_validator
            sys.modules["langchain_core.pydantic_v1"] = fake_pydantic_v1

    if "transformers" not in sys.modules:
        # 这里不要真实 import transformers。
        # 在当前环境中，transformers 会继续导入 torch，从而与 faiss 的 OpenMP runtime 冲突。
        fake_transformers = types.ModuleType("transformers")

        class GPT2TokenizerFast:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def encode(self, text: str, verbose: bool = False):
                tokens = tokenize_zh_en(text)
                return list(range(len(tokens) if tokens else len(text)))

        fake_transformers.GPT2TokenizerFast = GPT2TokenizerFast
        sys.modules["transformers"] = fake_transformers

    vertex_module_name = "langchain_community.chat_models.vertexai"
    if vertex_module_name not in sys.modules:
        try:
            importlib.import_module(vertex_module_name)
        except Exception:
            fake_vertex_module = types.ModuleType(vertex_module_name)

            class ChatVertexAI:
                pass

            fake_vertex_module.ChatVertexAI = ChatVertexAI
            sys.modules[vertex_module_name] = fake_vertex_module

    try:
        llms_module = importlib.import_module("langchain_community.llms")
        if not hasattr(llms_module, "VertexAI"):
            class VertexAI:
                pass

            llms_module.VertexAI = VertexAI
    except Exception:
        pass


def try_load_ragas_runtime() -> tuple[dict[str, Any] | None, Exception | None]:
    """尝试懒加载 ragas 运行时。"""
    _install_ragas_import_shims()
    try:
        runtime = {
            "evaluate": importlib.import_module("ragas").evaluate,
            "metrics": importlib.import_module("ragas.metrics"),
            "Dataset": importlib.import_module("datasets").Dataset,
            "Generation": importlib.import_module("langchain_core.outputs").Generation,
            "LLMResult": importlib.import_module("langchain_core.outputs").LLMResult,
            "RunConfig": importlib.import_module("ragas.run_config").RunConfig,
        }
        return runtime, None
    except Exception as exc:
        return None, exc


def build_ragas_dataset(rows: list[dict[str, Any]], dataset_cls: Any):
    payload = {
        "question": [row["question"] for row in rows],
        "answer": [row["answer"] for row in rows],
        "contexts": [row["contexts"] for row in rows],
        "ground_truth": [row["ground_truth"] for row in rows],
        "reference_contexts": [row.get("reference_contexts", []) for row in rows],
    }
    return dataset_cls.from_dict(payload)


def run_ragas_evaluation(
    rows: list[dict[str, Any]],
    llm_client: LLMClient,
    emb_client: EmbeddingClient,
    runtime: dict[str, Any],
):
    """运行真实 ragas 评估。"""
    Dataset = runtime["Dataset"]
    evaluate = runtime["evaluate"]
    metrics_mod = runtime["metrics"]
    Generation = runtime["Generation"]
    LLMResult = runtime["LLMResult"]
    RunConfig = runtime["RunConfig"]

    class ProjectRagasLLM:
        def __init__(self, client: LLMClient):
            self.client = client
            self.run_config = RunConfig()

        def set_run_config(self, run_config):
            self.run_config = run_config

        def get_temperature(self, n: int) -> float:
            return 0.3 if n > 1 else 1e-8

        def generate_text(
            self,
            prompt,
            n: int = 1,
            temperature: float = 1e-8,
            stop: list[str] | None = None,
            callbacks=None,
        ):
            prompt_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
            generations = []
            for _ in range(n):
                text = self.client.chat(
                    system_prompt=(
                        "You are a rigorous evaluation assistant. "
                        "Follow the instruction exactly and output valid JSON when requested."
                    ),
                    user_prompt=prompt_text,
                    temperature=temperature,
                )
                generations.append(Generation(text=text))
            return LLMResult(generations=[generations])

        async def agenerate_text(
            self,
            prompt,
            n: int = 1,
            temperature: float | None = None,
            stop: list[str] | None = None,
            callbacks=None,
        ):
            return await asyncio.to_thread(
                self.generate_text,
                prompt,
                n,
                self.get_temperature(n) if temperature is None else temperature,
                stop,
                callbacks,
            )

        async def generate(
            self,
            prompt,
            n: int = 1,
            temperature: float | None = None,
            stop: list[str] | None = None,
            callbacks=None,
            is_async: bool = True,
        ):
            if is_async:
                return await self.agenerate_text(prompt, n, temperature, stop, callbacks)
            return await asyncio.to_thread(
                self.generate_text,
                prompt,
                n,
                self.get_temperature(n) if temperature is None else temperature,
                stop,
                callbacks,
            )

    class ProjectRagasEmbeddings:
        def __init__(self, client: EmbeddingClient):
            self.client = client
            self.run_config = RunConfig()

        def set_run_config(self, run_config):
            self.run_config = run_config

        def embed_query(self, text: str) -> list[float]:
            return self.client.embed_query(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.client.embed_texts(texts)

        async def aembed_query(self, text: str) -> list[float]:
            return await asyncio.to_thread(self.embed_query, text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_documents, texts)

    dataset = build_ragas_dataset(rows, Dataset)
    metrics = [
        metrics_mod.faithfulness,
        metrics_mod.answer_relevancy,
        metrics_mod.context_precision,
        metrics_mod.context_recall,
    ]
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ProjectRagasLLM(llm_client),
        embeddings=ProjectRagasEmbeddings(emb_client),
        raise_exceptions=False,
    )
    logger.info(f"Ragas 评估完成: {result}")
    return result


def score_faithfulness(answer: str, contexts: list[str]) -> float:
    statements = split_sentences(answer)
    if not statements:
        return 0.0
    scores = []
    merged_context = "\n".join(contexts)
    for statement in statements:
        support = max(
            [lexical_similarity(statement, ctx) for ctx in contexts] + [lexical_similarity(statement, merged_context)]
        )
        scores.append(1.0 if support >= 0.35 else 0.0)
    return _safe_mean(scores)


def score_answer_relevancy(question: str, answer: str) -> float:
    return lexical_similarity(question, answer)


def score_context_precision(
    question: str,
    answer: str,
    contexts: list[str],
    reference: str,
    reference_contexts: list[str],
) -> float:
    if not contexts:
        return 0.0
    verdicts: list[int] = []
    for context in contexts:
        scores = [
            lexical_similarity(context, question),
            lexical_similarity(context, answer),
            lexical_similarity(context, reference),
        ]
        scores.extend(lexical_similarity(context, ref_ctx) for ref_ctx in reference_contexts)
        verdicts.append(1 if max(scores) >= 0.22 else 0)
    positives = sum(verdicts)
    if positives == 0:
        return 0.0
    numerator = sum(
        (sum(verdicts[: idx + 1]) / (idx + 1)) * verdicts[idx]
        for idx in range(len(verdicts))
    )
    return numerator / positives


def score_context_recall(reference: str, contexts: list[str], reference_contexts: list[str]) -> float:
    if reference_contexts:
        matched = 0
        for ref_ctx in reference_contexts:
            best = max((lexical_similarity(ref_ctx, ctx) for ctx in contexts), default=0.0)
            if best >= 0.2:
                matched += 1
        return matched / len(reference_contexts)

    statements = split_sentences(reference)
    if not statements:
        return 0.0
    matched = 0
    for statement in statements:
        best = max((lexical_similarity(statement, ctx) for ctx in contexts), default=0.0)
        if best >= 0.25:
            matched += 1
    return matched / len(statements)


def run_lightweight_evaluation(
    rows: list[dict[str, Any]],
    backend: str,
    error: str | None = None,
) -> LightweightEvaluationResult:
    """使用轻量启发式指标保证评估链路可运行。"""
    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        answer = row.get("answer", "") or ""
        contexts = row.get("contexts", []) or []
        reference = row.get("ground_truth", "") or ""
        question = row.get("question", "") or ""
        reference_contexts = row.get("reference_contexts", []) or []

        scored_row = dict(row)
        scored_row["faithfulness"] = score_faithfulness(answer, contexts)
        scored_row["answer_relevancy"] = score_answer_relevancy(question, answer)
        scored_row["context_precision"] = score_context_precision(
            question=question,
            answer=answer,
            contexts=contexts,
            reference=reference,
            reference_contexts=reference_contexts,
        )
        scored_row["context_recall"] = score_context_recall(
            reference=reference,
            contexts=contexts,
            reference_contexts=reference_contexts,
        )
        scored_rows.append(scored_row)

    result = LightweightEvaluationResult(scored_rows, backend=backend, error=error)
    logger.warning(f"使用轻量评估后端完成评估: {result}")
    return result


def extract_metric_summary(result: Any) -> dict[str, float]:
    summary: dict[str, float] = {}
    repr_dict = getattr(result, "_repr_dict", None)
    if isinstance(repr_dict, dict):
        for metric in METRIC_NAMES:
            summary[metric] = float(repr_dict.get(metric, 0.0))
        return summary

    for metric in METRIC_NAMES:
        values = None
        try:
            values = result[metric]
        except Exception:
            values = None
        if isinstance(values, list):
            summary[metric] = _safe_mean(values)
        elif isinstance(result, dict):
            summary[metric] = float(result.get(metric, 0.0))
        else:
            summary[metric] = 0.0
    return summary


def save_results(
    result: Any,
    output_path: str,
    golden_items: list[dict[str, Any]],
    backend: str,
    eval_error: str | None = None,
) -> None:
    """保存评估结果到 JSON 文件。"""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    if hasattr(result, "to_pandas"):
        try:
            raw_rows = result.to_pandas().to_dict(orient="records")
        except Exception as exc:
            logger.warning(f"结果转 DataFrame 失败，跳过明细导出: {exc}")

    output = {
        "timestamp": datetime.now().isoformat(),
        "backend": backend,
        "metrics": extract_metric_summary(result),
        "test_cases_count": len(golden_items),
        "raw_result": raw_rows,
    }
    if eval_error:
        output["eval_error"] = eval_error

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"评估结果已保存: {output_path}")
    print("\n" + "=" * 60)
    print(f"评估结果摘要（backend={backend}）")
    print("=" * 60)
    for metric, score in output["metrics"].items():
        print(f"{metric:20s}: {score:.4f}")
    if eval_error:
        print("-" * 60)
        print(f"降级原因: {eval_error}")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ragas 评估脚本")
    parser.add_argument(
        "--golden-set",
        required=True,
        help="黄金测试集路径（JSONL 格式）",
    )
    parser.add_argument(
        "--output",
        default="data/evaluation/ragas_results.json",
        help="评估结果输出路径",
    )
    parser.add_argument(
        "--eval-backend",
        default="auto",
        choices=["auto", "ragas", "lightweight"],
        help="评估后端：auto 优先 ragas，失败时自动降级到 lightweight",
    )
    parser.add_argument("--max-cases", type=int, default=None, help="仅评估前 N 条样本，便于本地调试")
    parser.add_argument("--sleep-sec", type=float, default=0.5, help="每条查询之间的延迟秒数")

    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--meta", default=DEFAULT_META)

    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--retrieval-mode", default="hybrid", choices=["hybrid", "dense"])
    parser.add_argument("--max-context-chars", type=int, default=3200)

    parser.add_argument(
        "--embedding-provider",
        default="openai_compatible",
        choices=["openai_compatible", "ollama", "fastembed", "sentence_transformers"],
    )
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-api-base", default=None)
    parser.add_argument("--embedding-api-key", default=None)

    parser.add_argument(
        "--llm-provider",
        default="openai_compatible",
        choices=["openai_compatible", "ollama"],
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-base", default=None)
    parser.add_argument("--llm-api-key", default=None)

    parser.add_argument(
        "--reranker-provider",
        default="none",
        choices=["none", "cross_encoder"],
    )
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--reranker-device", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    golden_items = load_golden_set(args.golden_set, max_cases=args.max_cases)
    emb_client = EmbeddingClient(
        provider=args.embedding_provider,
        model=args.embedding_model,
        api_base=args.embedding_api_base,
        api_key=args.embedding_api_key,
    )
    llm_client = LLMClient(
        provider=args.llm_provider,
        model=args.llm_model,
        api_base=args.llm_api_base,
        api_key=args.llm_api_key,
    )
    reranker = RerankerClient(
        provider=args.reranker_provider,
        model_name=args.reranker_model,
        device=args.reranker_device,
        fail_open=True,
    )

    store = FaissVectorStore(dim=1)
    store.load(
        index_path=args.index,
        records_path=args.records,
        meta_path=args.meta,
    )
    logger.info(f"向量库加载完成: {len(store.records)} 条记录")
    validate_retrieval_compatibility(store, emb_client)

    rows = prepare_eval_rows(
        golden_items=golden_items,
        store=store,
        emb_client=emb_client,
        llm_client=llm_client,
        reranker=reranker,
        args=args,
    )

    backend = args.eval_backend
    eval_error: str | None = None
    if backend == "lightweight":
        result = run_lightweight_evaluation(rows, backend="lightweight")
        save_results(result, args.output, golden_items, backend="lightweight")
        return

    runtime, ragas_error = try_load_ragas_runtime()
    if runtime is not None:
        try:
            result = run_ragas_evaluation(rows, llm_client, emb_client, runtime)
            save_results(result, args.output, golden_items, backend="ragas")
            return
        except Exception as exc:
            ragas_error = exc

    eval_error = str(ragas_error) if ragas_error else "未知 ragas 错误"
    if backend == "ragas":
        raise RuntimeError(f"ragas 后端执行失败: {eval_error}") from ragas_error

    logger.warning(f"ragas 后端不可用，自动降级到 lightweight: {eval_error}")
    result = run_lightweight_evaluation(rows, backend="lightweight_fallback", error=eval_error)
    save_results(
        result,
        args.output,
        golden_items,
        backend="lightweight_fallback",
        eval_error=eval_error,
    )


if __name__ == "__main__":
    main()
