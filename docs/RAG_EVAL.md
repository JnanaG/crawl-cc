# RAG 效果评估说明

## 目标

提供一个可直接运行的轻量评估流程，覆盖三类指标：

- 检索命中：`HitRate@K`、`Recall@K`、`MRR`
- 引用准确性（可选生成评估）：`citation_precision`、`citation_hit_rate`
- 响应长度代理指标：`avg_answer_length`

评估脚本：`rag_eval.py`

## 1) 生成评测集

先基于当前向量库 `records` 自动生成问题集（按车系标题构造问题）：

```bash
python rag_eval.py prepare --records data/vector_store/faiss/dongchedi_records.jsonl --output data/eval/rag_eval_set.jsonl --max-samples 80
```

输出文件为 JSONL，每行格式如下：

```json
{
  "question": "宝马3系属于什么级别车型？",
  "positive_series_ids": ["12345"],
  "meta": {"title": "宝马3系", "source": "auto_generated"}
}
```

## 2) 仅检索评估（推荐先跑）

```bash
python rag_eval.py run ^
  --eval-set data/eval/rag_eval_set.jsonl ^
  --retrieval-mode hybrid ^
  --embedding-provider ollama ^
  --embedding-model nomic-embed-text ^
  --top-k 6
```

输出：

- 汇总报告：`data/eval/rag_eval_report.json`
- 明细结果：`data/eval/rag_eval_details.jsonl`

## 3) 检索 + 生成 + 引用准确率评估（可选）

```bash
python rag_eval.py run ^
  --eval-set data/eval/rag_eval_set.jsonl ^
  --retrieval-mode hybrid ^
  --embedding-provider ollama ^
  --embedding-model nomic-embed-text ^
  --with-generation ^
  --llm-provider ollama ^
  --llm-model qwen2.5:7b
```

新增指标：

- `citation_precision`: 引用编号对应命中结果中，相关引用占比
- `citation_hit_rate`: 至少出现 1 条正确引用的比例
- `answer_has_citation_rate`: 回答包含引用编号的比例
- `cannot_confirm_rate`: 回答中出现“无法确认”的比例（用于观测保守性）
- `avg_answer_length`: 平均回答长度（用于观测回答过短/过长）

## 4) 指标解释

- `HitRate@K`: Top-K 是否至少命中一个正确车系
- `Recall@K`: Top-K 命中的正确车系列表覆盖率
- `MRR`: 第一个正确结果的倒数排名均值，越高越好

## 5) 建议阈值（学习/求职项目可用）

- `HitRate@6 >= 0.75`
- `MRR >= 0.45`
- `citation_precision >= 0.60`（开启生成评测时）

达到上述范围，基本可以支撑你在面试中展示“可评估、可优化”的工程闭环。
