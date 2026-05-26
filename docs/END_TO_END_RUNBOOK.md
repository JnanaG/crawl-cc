# 端到端运行手册

本文档说明当前项目如何从数据生产、向量构建、RAG 评测到评测回流完成一条端到端链路，并给出验收方法。

## 环境

- Windows
- Python 虚拟环境：`D:\workplace\Crawl\.venv`

激活方式：

```powershell
& D:\workplace\Crawl\.venv\Scripts\Activate.ps1
```

## 1. 主流程

### 1.1 执行采集 + 清洗 + 治理 + 分块

```powershell
python main.py
```

主要产物：

- `data/cleaned/dongchedi/json/`
- `data/cleaned/dongchedi/markdown/`
- `data/processed/dongchedi_training_data.jsonl`
- `data/processed/dongchedi_training_data.parquet`
- `data/reports/task_summary.json`
- `data/reports/quality_report.json`

### 1.2 将 processed 分流为多目标资产

```powershell
python scripts/split_processed_assets.py
```

主要产物：

- `data/assets/rag/rag_corpus.jsonl`
- `data/assets/training/training_corpus.jsonl`
- `data/assets/evaluation/eval_candidates.jsonl`
- `data/assets/asset_split_summary.json`

说明：

- `rag_corpus.jsonl` 可直接作为建索引输入
- `training_corpus.jsonl` 更偏向训练友好的高质量样本
- `eval_candidates.jsonl` 是后续人工或 LLM 扩写黄金集的候选池

### 1.3 构建向量库

```powershell
python rag_llm_demo.py build --input data/assets/rag/rag_corpus.jsonl --embedding-provider fastembed
```

主要产物：

- `data/vector_store/faiss/dongchedi.index`
- `data/vector_store/faiss/dongchedi_records.jsonl`
- `data/vector_store/faiss/dongchedi_meta.json`

### 1.4 执行 RAG 评测

```powershell
python rag_ragas_eval.py ^
  --golden-set data/evaluation/golden_set.jsonl ^
  --output data/evaluation/ragas_results.json ^
  --embedding-provider fastembed ^
  --llm-provider ollama ^
  --llm-model qwen2.5:3b
```

主要产物：

- `data/evaluation/ragas_results.json`

### 1.5 回流低分样本

```powershell
python scripts/sync_ragas_feedback.py --results data/evaluation/ragas_results.json
```

主要产物：

- `data/feedback/agent_pipeline/ragas_feedback.jsonl`
- `data/feedback/agent_pipeline/ragas_feedback_summary.json`
- `data/review/agent_pipeline/review_queue.jsonl`
- `data/repair/agent_pipeline/repair_queue.jsonl`

### 1.6 评测守门

```powershell
python scripts/ci_eval_gate.py --results data/evaluation/ragas_results.json
```

说明：

- 返回码 `0`：指标通过
- 返回码 `1`：指标未达标
- 这一步失败不代表脚本坏了，通常表示当前评测质量没有达到门槛

## 2. 一键工作流

### 2.1 先看命令，不真正执行

```powershell
python scripts/run_pipeline_workflow.py --dry-run --run-ci-gate --llm-provider ollama --llm-model qwen2.5:3b
```

### 2.2 执行完整工作流

```powershell
python scripts/run_pipeline_workflow.py --run-ci-gate --llm-provider ollama --llm-model qwen2.5:3b
```

### 2.3 复用已有数据，只做后半段

```powershell
python scripts/run_pipeline_workflow.py --skip-main --run-ci-gate --llm-provider ollama --llm-model qwen2.5:3b
```

### 2.4 只做回流和守门

```powershell
python scripts/run_pipeline_workflow.py --skip-main --skip-build --skip-eval --run-ci-gate
```

### 2.5 失败后断点续跑

```powershell
python scripts/run_pipeline_workflow.py ^
  --run-id my-batch-001 ^
  --resume ^
  --step-retries 2 ^
  --retry-delay-sec 1 ^
  --run-ci-gate
```

说明：

- `--step-retries` 控制单步失败后的自动重试次数
- `--resume` 会复用同一 `run-id` 下已经成功的步骤，只从失败步骤继续
- 如果上一次已经全部成功，再次 `--resume` 会直接结束，不重复执行

工作流状态文件默认落盘到：

- `data/state/workflow_runs/workflow_<run_id>.json`

状态文件包含：

- 每一步命令
- 环境变量覆盖
- 开始/结束时间
- 耗时
- 返回码
- 成功或失败状态
- 重试次数与逐次 attempt 记录
- resume 历史

告警产物默认落盘到：

- `data/alerts/workflow/workflow_alerts.jsonl`
- `data/alerts/workflow/latest_alert.json`

状态目录还会追加：

- `data/state/workflow_runs/workflow_history.jsonl`
- `data/state/workflow_runs/workflow_quality_baseline.json`

## 3. 自动化测试

### 3.1 治理链路测试

```powershell
python scripts/test_governance_pipeline.py
```

### 3.2 评测回流测试

```powershell
python scripts/test_ragas_feedback.py
```

### 3.3 调度编排测试

```powershell
python scripts/test_pipeline_workflow.py
```

### 3.4 多目标资产分流测试

```powershell
python scripts/test_split_processed_assets.py
```

### 3.5 工作流韧性与告警测试

```powershell
python scripts/test_workflow_resilience.py
```

### 3.6 LLM Agent 接入测试

```powershell
python scripts/test_governance_llm_agents.py
```

## 4. 端到端验收

当你已经跑过主流程、build、eval、feedback 后，可以用下面的脚本检查关键产物是否齐全且结构合理：

```powershell
python scripts/verify_end_to_end.py
```

验收脚本会检查：

- `processed` 是否存在且非空
- `vector_store/faiss` 是否存在且非空
- `ragas_results.json` 是否存在且有 `metrics`
- `ragas_feedback_summary.json` 是否存在且与明细数量一致
- `task_summary.json` / `quality_report.json` 是否存在
- 最近一次 `workflow state` 是否存在
- 会先检查当前 `scripts/ci_eval_gate.py` 本身没有语法错误
- 如果最近一次 workflow 中的 `ci_gate` 记录了旧 `SyntaxError`，但当前脚本更新时间更新，则视为历史错误记录，不再误报当前脚本损坏

## 5. 如何理解验收结果

### 5.1 `verify_end_to_end.py` 通过

表示：

- 端到端产物链已经形成
- 关键脚本都成功产出预期文件
- 当前系统至少在工程链路上是闭合的

### 5.2 `ci_eval_gate.py` 返回 1

表示：

- 当前评测指标没达到门槛
- 不是脚本异常
- 应优先查看：
  - `data/evaluation/ragas_results.json`
  - `data/feedback/agent_pipeline/ragas_feedback_summary.json`
- 如果 `verify_end_to_end.py` 仍通过，说明工程链路闭合，问题属于质量门槛未达标而非脚本损坏

## 6. 当前建议的运行顺序

开发调试时建议按这个顺序：

1. `python scripts/test_governance_pipeline.py`
2. `python scripts/test_ragas_feedback.py`
3. `python scripts/test_pipeline_workflow.py`
4. `python scripts/test_split_processed_assets.py`
5. `python scripts/test_workflow_resilience.py`
6. `python scripts/run_pipeline_workflow.py --dry-run ...`
7. `python scripts/run_pipeline_workflow.py ...`
8. `python scripts/split_processed_assets.py`
9. `python scripts/verify_end_to_end.py`

## 5. LLM Agent 运行

### 5.1 单条样本关闭 LLM Agent

```powershell
python scripts/run_governance_without_llm.py --input-clean-json data\cleaned\dongchedi\json\series_*.json
```

或者使用通用入口：

```powershell
python scripts/run_governance_agent.py --input-clean-json <clean_json_path>
```

如果只想先验证脚本链路、不依赖真实 embedding 模型，可加：

```powershell
python scripts/run_governance_without_llm.py --semantic-store-mode stub
```

### 5.2 单条样本启用 LLM Agent

Ollama:

```powershell
python scripts/run_governance_with_llm.py ^
  --input-clean-json <clean_json_path> ^
  --semantic-store-mode stub ^
  --llm-provider ollama ^
  --llm-model qwen2.5:3b ^
  --llm-api-base http://localhost:11434
```

OpenAI Compatible:

```powershell
python scripts/run_governance_with_llm.py ^
  --input-clean-json <clean_json_path> ^
  --semantic-store-mode stub ^
  --llm-provider openai_compatible ^
  --llm-model gpt-4o-mini ^
  --llm-api-base https://your-endpoint/v1 ^
  --llm-api-key <your_key>
```

说明：

- `run_governance_with_llm.py` 和 `run_governance_without_llm.py` 都会输出 route、quality、metadata 和 audit
- 开启 LLM 后，当前只让 `route` 和 `quality` 两个 agent 调用模型
- 如果开启了 `llm_fail_open`，模型调用失败时会自动回退到规则/启发式结果

### 5.3 主流程启用 LLM Agent

`main.py` 已支持通过环境变量启用：

```powershell
$env:AGENT_ENABLE_LLM="true"
$env:AGENT_LLM_PROVIDER="ollama"
$env:AGENT_LLM_MODEL="qwen2.5:3b"
$env:AGENT_LLM_API_BASE="http://localhost:11434"
$env:AGENT_LLM_FAIL_OPEN="true"
python main.py
```

关闭时不设置这些变量，或显式：

```powershell
$env:AGENT_ENABLE_LLM="false"
python main.py
```

治理审计里会记录：

- `metadata.agent_llm_enabled`
- `metadata.route_llm_used`
- `metadata.quality_llm_used`
- `audit_logs[].llm_agent`

## 7. 当前已知现象

- `run_pipeline_workflow.py --skip-main` 依然会执行 `build/eval/feedback/ci_gate`
- 评测最慢的通常是 `eval`，因为需要对黄金集逐条执行完整 RAG 查询
- `ci_gate` 返回 `1` 的常见原因是指标不达标，而不是脚本崩溃
- `run_pipeline_workflow.py` 现在支持步级重试、resume 和质量波动告警，但 baseline 需要至少一次成功运行后才会建立
- LLM Agent 当前只接入 `route` 和 `quality` 两个节点，`decision` 仍由确定性规则守门
