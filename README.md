# Crawl_cc

懂车帝汽车数据采集、治理、RAG 检索与评测一体化项目。

项目当前已经覆盖从数据采集到治理、分流、建索引、评测、回流、工作流编排的一整条链路，并支持可选的 LLM Agent 增强。

## 项目能力

- 数据采集：抓取懂车帝车系页面，提取原始 HTML 与 `__NEXT_DATA__` JSON
- 数据清洗：生成结构化 JSON 和适合 LLM / RAG 的 Markdown 文本
- 智能治理：`route -> rule_dedup -> semantic_dedup -> quality -> decision` 五节点 Agent 编排
- 三层去重：精确去重、规范化去重、语义去重
- 质量诊断：质量分、问题标签、`rag_readiness`、`training_readiness`
- 多目标资产分流：RAG 语料、训练集、评测候选集
- RAG 能力：FAISS、混合检索、可选 reranker、Ollama / OpenAI Compatible / 本地 embedding
- RAG 评测：`ragas` 优先，失败时可回退轻量评测
- 评测回流：低分样本回流到 `review / repair / feedback`
- 工作流编排：支持 dry-run、断点续跑、重试、告警、质量波动检测
- 可观测性：trace_id、batch_id、audit_logs、workflow state、失败审计

## 当前架构

```text
采集 -> 清洗 -> 治理 -> 分块 -> 导出 processed
                          -> 分流 assets
processed/assets -> build vector store -> query -> eval -> feedback -> ci gate
```

治理层当前采用 LangGraph 风格的 5 节点主链路：

```text
route -> rule_dedup -> semantic_dedup -> quality -> decision
```

- 接入 LLM 的节点：`route`、`quality`
- 不接 LLM 的节点：`rule_dedup`、`semantic_dedup`、`decision`

拓扑图见：

- [docs/current_agent_topology.png](docs/current_agent_topology.png)

## 目录结构

```text
.
├── main.py                         # 主流程：采集、清洗、治理、分块、导出
├── agent_pipeline/                 # 智能治理层
├── cleaner/                        # 数据清洗
├── quality/                        # 质量检查
├── scraper/                        # 懂车帝抓取
├── utils/                          # embedding / llm / vector store / state 等工具
├── scripts/                        # workflow、测试、验收、分流、LLM Agent 运行脚本
├── docs/                           # 运行手册、RAG 文档、评审材料、拓扑图
├── langchain_rag/                  # LangChain 版 RAG 实现
├── rag_demo.py                     # 最小 RAG demo
├── rag_llm_demo.py                 # 标准 RAG demo
├── rag_langchain_demo.py           # LangChain RAG demo
├── rag_eval.py                     # 传统 RAG 检索评测
└── rag_ragas_eval.py               # ragas / fallback 评测
```

## 环境要求

- Windows
- Python 3.11 / 3.12
- 推荐虚拟环境：`D:\workplace\Crawl\.venv`

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 快速开始

### 1. 运行主流程

```powershell
python main.py
```

主要产物：

- `data/cleaned/dongchedi/json/`
- `data/cleaned/dongchedi/markdown/`
- `data/processed/dongchedi_training_data.jsonl`
- `data/processed/dongchedi_training_data.parquet`

### 2. 分流多目标资产

```powershell
python scripts/split_processed_assets.py
```

产物：

- `data/assets/rag/rag_corpus.jsonl`
- `data/assets/training/training_corpus.jsonl`
- `data/assets/evaluation/eval_candidates.jsonl`

### 3. 构建向量库

```powershell
python rag_llm_demo.py build --input data/assets/rag/rag_corpus.jsonl --embedding-provider fastembed
```

### 4. 执行 RAG 评测

```powershell
python rag_ragas_eval.py ^
  --golden-set data/evaluation/golden_set.jsonl ^
  --output data/evaluation/ragas_results.json ^
  --embedding-provider fastembed ^
  --llm-provider ollama ^
  --llm-model qwen2.5:3b
```

### 5. 回流与守门

```powershell
python scripts/sync_ragas_feedback.py --results data/evaluation/ragas_results.json
python scripts/ci_eval_gate.py --results data/evaluation/ragas_results.json
```

## 一键工作流

查看命令而不执行：

```powershell
python scripts/run_pipeline_workflow.py --dry-run --run-ci-gate --llm-provider ollama --llm-model qwen2.5:3b
```

执行工作流：

```powershell
python scripts/run_pipeline_workflow.py --run-ci-gate --llm-provider ollama --llm-model qwen2.5:3b
```

断点续跑：

```powershell
python scripts/run_pipeline_workflow.py ^
  --run-id my-batch-001 ^
  --resume ^
  --step-retries 2 ^
  --retry-delay-sec 1 ^
  --run-ci-gate
```

## LLM Agent 模式

### 单条样本关闭 LLM Agent

```powershell
python scripts/run_governance_without_llm.py --semantic-store-mode stub
```

### 单条样本启用 LLM Agent

```powershell
python scripts/run_governance_with_llm.py ^
  --semantic-store-mode stub ^
  --llm-provider ollama ^
  --llm-model qwen2.5:3b ^
  --llm-api-base http://localhost:11434
```

### 主流程启用 LLM Agent

```powershell
$env:AGENT_ENABLE_LLM="true"
$env:AGENT_LLM_PROVIDER="ollama"
$env:AGENT_LLM_MODEL="qwen2.5:3b"
$env:AGENT_LLM_API_BASE="http://localhost:11434"
$env:AGENT_LLM_FAIL_OPEN="true"
python main.py
```

## 测试与验收

治理链路：

```powershell
python scripts/test_governance_pipeline.py
python scripts/test_governance_llm_agents.py
```

工作流与资产分流：

```powershell
python scripts/test_pipeline_workflow.py
python scripts/test_workflow_resilience.py
python scripts/test_split_processed_assets.py
```

评测回流与端到端验收：

```powershell
python scripts/test_ragas_feedback.py
python scripts/test_verify_end_to_end.py
python scripts/verify_end_to_end.py
```

## 重要文档

- [docs/END_TO_END_RUNBOOK.md](docs/END_TO_END_RUNBOOK.md)
- [docs/RAG_DEMO.md](docs/RAG_DEMO.md)
- [docs/RAG_EVAL.md](docs/RAG_EVAL.md)
- [docs/QUALITY_RULES.md](docs/QUALITY_RULES.md)
- [项目说明文档.md](项目说明文档.md)

## 上传说明

仓库已通过 `.gitignore` 排除以下内容，不会上传：

- `.venv/`
- `venv/`
- `env/`
- `data/`
- 本地日志与 IDE 文件

## 当前状态

当前代码基线已经具备：

- 智能治理 Agent 编排
- 可选 LLM Agent 接入
- 端到端验证
- 数据分流
- 工作流编排与告警
- RAG 评测回流

后续规划方向包括：

- 多轮对话
- 记忆机制
- 更强的推荐 / 比较问答
- 知识图谱（暂未接入）
