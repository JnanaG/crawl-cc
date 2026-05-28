# 数据治理与智能 Pipeline 全链路说明

## 1. 文档定位

本文档补充说明当前项目中更完整的智能数据链路，重点不是“怎么执行命令”，而是：

- 数据如何从爬取进入治理
- 多 Agent 如何协作完成数据决策
- 哪些样本会进入 `processed`
- 评测结果如何回流到 `review/repair`
- 工作流如何把整条链路串成闭环

如果你想看具体命令和操作步骤，请结合阅读：

- [新三模块爬取与全流程说明.md](file:///D:/workplace/Crawl_cc/docs/%E6%96%B0%E4%B8%89%E6%A8%A1%E5%9D%97%E7%88%AC%E5%8F%96%E4%B8%8E%E5%85%A8%E6%B5%81%E7%A8%8B%E8%AF%B4%E6%98%8E.md)

## 2. 全链路总览

当前项目的完整链路可以概括为：

```text
新三模块爬取
-> 原始数据落盘 raw
-> 清洗标准化 cleaned
-> 规则质检
-> 多 Agent 治理编排
-> accept / repair / review / drop 决策
-> processed 训练与检索样本
-> 资产分流
-> RAG 建库与检索
-> 离线评测
-> 评测回流到 review / repair
-> workflow 基线对比、告警、验收
```

这条链路的核心特点有 3 个：

- 不是“抓到就入库”，而是先治理再决定是否进入下游
- 不是“只做 RAG Demo”，而是把评测结果真正回流到数据侧
- 不是“单脚本串联”，而是有状态、可审计、可续跑的 workflow

## 3. 数据分层

### 3.1 Raw 层

入口实现：

- [dcd_scraper.py](file:///D:/workplace/Crawl_cc/scraper/dcd_scraper.py)
- [refetch_series_modules.py](file:///D:/workplace/Crawl_cc/scripts/refetch_series_modules.py)

当前 raw 层固定采集 3 个模块：

- 主页 `series/<id>`
- 外观模块 `series/<id>/images-wg`
- 内饰模块 `series/<id>/images-ns`

raw 层产物包括：

- 原始 HTML
- 原始 JSON
- 原始图片文件
- 图片 manifest

对应目录：

- `data/raw/dongchedi/`
- `data/raw/dongchedi/images/series_<id>/`

### 3.2 Cleaned 层

入口实现：

- [dcd_cleaner.py](file:///D:/workplace/Crawl_cc/cleaner/dcd_cleaner.py)

cleaned 层将 raw JSON 规范化为统一 schema，核心目标是：

- 把页面结构转成稳定字段
- 生成后续治理和 chunk 可消费的 Markdown
- 保持字段版本和结构可追踪

典型字段包括：

- `series`
- `pricing`
- `scores`
- `dimensions`
- `models`
- `images`
- `stats`

对应目录：

- `data/cleaned/dongchedi/json/`
- `data/cleaned/dongchedi/markdown/`

### 3.3 Processed 层

入口实现：

- [main.py](file:///D:/workplace/Crawl_cc/main.py)

processed 层不是简单从 cleaned 直接导出，而是要先经过治理决策。只有满足治理条件的样本，才会进入：

- `data/processed/dongchedi_training_data.jsonl`
- `data/processed/dongchedi_training_data.parquet`

### 3.4 Assets 层

入口实现：

- [split_processed_assets.py](file:///D:/workplace/Crawl_cc/scripts/split_processed_assets.py)
- [dataset_splitter.py](file:///D:/workplace/Crawl_cc/agent_pipeline/dataset_splitter.py)

processed 层会进一步拆成：

- `rag_corpus`
- `training_corpus`
- `eval_candidates`

这意味着项目把“下游如何消费数据”显式建模了出来，而不是只产出一份通用语料。

## 4. 多 Agent 治理链路

核心入口：

- [orchestrator.py](file:///D:/workplace/Crawl_cc/agent_pipeline/orchestrator.py)

治理编排器 `GovernanceOrchestrator` 基于 `langgraph` 风格的状态图，顺序执行 5 个节点：

```text
route -> rule_dedup -> semantic_dedup -> quality -> decision
```

每一步都会写入 `audit_logs`，并保存治理结果或失败信息，因此整条链路天然支持：

- 审计
- 问题复盘
- 指标追踪
- 失败样本回查

### 4.1 Route 节点

实现位置：

- [orchestrator.py](file:///D:/workplace/Crawl_cc/agent_pipeline/orchestrator.py)
- [agents.py](file:///D:/workplace/Crawl_cc/agent_pipeline/agents.py)
- [llm_agents.py](file:///D:/workplace/Crawl_cc/agent_pipeline/llm_agents.py)

职责：

- 根据样本字段判断应走哪个路由渠道和模板
- 先执行规则识别
- 再执行启发式增强
- 在必要时再用 LLM 微调

实现逻辑：

- `detect_route_by_rules()` 先给出基础判断
- `route_agent_refine()` 在规则置信度不足时提升或修正结果
- `GovernanceLLMBridge.route_refine()` 只在基础路由不够确定时调用 LLM

这一层的设计思想是：

- 规则优先
- LLM 只做低置信度样本的增强
- 如果 LLM 失败则 `fail_open` 回退到原结果

### 4.2 Rule Dedup 节点

实现位置：

- [orchestrator.py](file:///D:/workplace/Crawl_cc/agent_pipeline/orchestrator.py)
- `agent_pipeline/rules_bridge.py`

职责：

- 做确定性去重
- 生成 `content_hash` 和 `normalized_hash`
- 命中 manifest 时快速识别重复内容

这一层适合处理：

- 完全重复
- 规范化后重复
- 明确可确定的重复样本

### 4.3 Semantic Dedup 节点

实现位置：

- [orchestrator.py](file:///D:/workplace/Crawl_cc/agent_pipeline/orchestrator.py)
- [agents.py](file:///D:/workplace/Crawl_cc/agent_pipeline/agents.py)
- `agent_pipeline/semantic_dedup_store.py`

职责：

- 补足规则去重覆盖不到的“语义近似样本”
- 根据相似度阈值判断是否重复
- 同车系与跨车系使用不同阈值

典型判断逻辑：

- 如果文本低信息密度过高，则跳过语义去重
- 如果同车系候选相似度超过 `same_series_threshold`，则判为语义重复
- 否则使用一般 `similarity_threshold`

### 4.4 Quality 节点

实现位置：

- [orchestrator.py](file:///D:/workplace/Crawl_cc/agent_pipeline/orchestrator.py)
- [agents.py](file:///D:/workplace/Crawl_cc/agent_pipeline/agents.py)
- [llm_agents.py](file:///D:/workplace/Crawl_cc/agent_pipeline/llm_agents.py)
- [data_quality.py](file:///D:/workplace/Crawl_cc/quality/data_quality.py)

职责：

- 对 cleaned 样本进行规则质检
- 给出多维质量评分
- 评估样本是否适合训练和检索

质量输出包括：

- `quality_score`
- `quality_tier`
- `rag_readiness`
- `training_readiness`
- `issues`
- `issue_groups`
- `repair_suggestion`

质量判断不是只看一个总分，而是从多个维度衡量：

- 完整性
- 结构性
- 内容密度
- 检索可切分性
- 时效性

LLM 在这里也是增强角色，不会完全覆盖规则分数，而是做有限幅度修正。

### 4.5 Decision 节点

实现位置：

- [agents.py](file:///D:/workplace/Crawl_cc/agent_pipeline/agents.py)

最终决策有 4 类：

- `accept`
- `repair`
- `review`
- `drop`

主要规则：

- 如果样本是重复内容，则直接 `drop`
- 如果路由置信度过低，则进入 `review`
- 如果质量分过低或 `rag_readiness` 太低，则进入 `review`
- 如果样本中等质量但仍可用，则进入 `repair`
- 只有高质量样本进入 `accept`

这一步的本质是把“是否进入下游”的标准显式化，而不是把所有 cleaned 数据一股脑写入训练和检索资产。

## 5. 治理结果如何影响下游输出

关键实现：

- [main.py](file:///D:/workplace/Crawl_cc/main.py)

`main.py` 中会在清洗完成后调用治理编排器：

- 对每个车系执行 `governance_orchestrator.govern(...)`
- 统计 `accept / repair / review / drop` 数量
- 只有 `accept` 和 `repair` 会继续进入后续 chunk 和 processed 输出

进入 processed 时会把治理元信息一并写到 metadata 中，例如：

- `governance_decision`
- `governance_reason`
- `quality_score`
- `quality_tier`
- `rag_readiness`
- `training_readiness`
- `route_confidence`
- `dedup_duplicate_type`
- `semantic_evidence`

这意味着后续所有资产分流、向量建库、评测，都不是盲目的，而是带着治理上下文继续往下走。

## 6. 资产分流逻辑

关键实现：

- [dataset_splitter.py](file:///D:/workplace/Crawl_cc/agent_pipeline/dataset_splitter.py)

分流时会进一步根据治理与可用性信息筛选：

- 只有 `governance_decision` 为 `accept` 或 `repair` 的样本才有资格进入资产层
- 路由置信度过低、去重命中、正文为空的样本会被直接排除
- `rag_corpus` 和 `training_corpus` 会使用不同的 readiness 阈值

这种设计体现了“同源数据多目标供给”的思想：

- `rag_corpus` 更关心检索可用性
- `training_corpus` 更关心训练稳定性和高质量程度
- `eval_candidates` 则服务评测问题构造

## 7. 多模态链路如何接入治理体系

当前新增的多模态链路包括：

- 原始图片保存
- 图片资产构建
- 多模态 caption 生成
- caption 语料文本化后并入 RAG

对应实现：

- [image_pipeline.py](file:///D:/workplace/Crawl_cc/multimodal/image_pipeline.py)
- [caption_builder.py](file:///D:/workplace/Crawl_cc/multimodal/caption_builder.py)
- [build_image_caption_corpus.py](file:///D:/workplace/Crawl_cc/scripts/build_image_caption_corpus.py)

当前多模态部分与治理主链路的关系是：

- 爬取与清洗阶段负责把原始图片稳定落盘
- 图片资产层负责把图片转成统一 manifest
- caption 层把视觉信息文本化
- caption 文本作为新的语料源进入 RAG 建库

换句话说，当前多模态链路是“并入检索资产层”的，而不是直接并入治理主图。这个设计的优点是改造成本低、复用现有 embedding 和检索流程更容易。

## 8. RAG 建库与检索

核心入口：

- [rag_llm_demo.py](file:///D:/workplace/Crawl_cc/rag_llm_demo.py)

当前项目支持：

- 文本 processed 语料建库
- 图片 caption 语料建库
- 二者混合建库

检索层支持：

- 稠密检索
- hybrid 检索
- RRF 融合
- 可选 rerank

这意味着数据治理的结果并不会停留在报告里，而是实质影响后续检索质量。

## 9. 离线评测体系

核心入口：

- [rag_eval.py](file:///D:/workplace/Crawl_cc/rag_eval.py)
- [rag_ragas_eval.py](file:///D:/workplace/Crawl_cc/rag_ragas_eval.py)

评测目标不是只看“模型答得像不像”，而是把系统拆成多个指标来衡量：

- `faithfulness`
- `answer_relevancy`
- `context_precision`
- `context_recall`

评测脚本支持两种模式：

- 优先使用真实 `ragas`
- 如果环境不满足，则退回 lightweight 评测逻辑

这样做的好处是评测链路不会因为环境问题完全中断。

## 10. 评测回流机制

核心入口：

- [sync_ragas_feedback.py](file:///D:/workplace/Crawl_cc/scripts/sync_ragas_feedback.py)
- [feedback_bridge.py](file:///D:/workplace/Crawl_cc/agent_pipeline/feedback_bridge.py)

评测结果不会只停留在一份报表中，而是会被分类成可执行的反馈项。

典型分类标签包括：

- `query_failure`
- `empty_context`
- `coverage_gap`
- `retrieval_noise`
- `grounding_weak`
- `answer_mismatch`

每条反馈会被分配到：

- `review` 队列
- 或 `repair` 队列

并带上：

- `severity`
- `failed_metrics`
- `issue_groups`
- `suggested_actions`

这一步把“评测发现问题”变成“治理侧可继续处理的问题单”，是整个项目闭环能力的关键。

## 11. Workflow、告警与验收

核心入口：

- [run_pipeline_workflow.py](file:///D:/workplace/Crawl_cc/scripts/run_pipeline_workflow.py)
- [verify_end_to_end.py](file:///D:/workplace/Crawl_cc/scripts/verify_end_to_end.py)

workflow 层负责把若干离散脚本串成一个批处理任务，包括：

- main 主流程
- build 向量建库
- eval 评测
- feedback 回流
- ci gate

workflow 提供的工程能力包括：

- `dry-run`
- `resume`
- 单步骤重试
- 状态落盘
- 质量基线对比
- 指标回退告警

它还会比较当前运行结果和 baseline 的差异，例如：

- 任务成功率是否下降
- 有效车系占比是否下降
- 训练样本量是否明显减少
- 覆盖车系数是否明显减少
- 评测指标是否显著回退

这让项目从“可以运行”进一步提升为“可以持续治理和持续交付”。

## 12. 当前项目最值得强调的工程价值

如果从 AI 数据开发或 AI 数据治理角度看，这个项目最有价值的不是单独某个脚本，而是下面这些能力被接成了闭环：

- 新三模块定向采集
- 规则 + 启发式 + LLM 的协同治理
- 去重、质检、决策的显式标准化
- 治理结果对 processed 和资产分流的直接约束
- 多模态 caption 文本化接入既有 RAG
- 离线评测与低分样本回流
- workflow 基线对比、告警和验收

## 13. 推荐阅读顺序

如果你要快速理解项目，建议按下面顺序阅读：

1. [新三模块爬取与全流程说明.md](file:///D:/workplace/Crawl_cc/docs/%E6%96%B0%E4%B8%89%E6%A8%A1%E5%9D%97%E7%88%AC%E5%8F%96%E4%B8%8E%E5%85%A8%E6%B5%81%E7%A8%8B%E8%AF%B4%E6%98%8E.md)
2. [数据治理与智能Pipeline全链路说明.md](file:///D:/workplace/Crawl_cc/docs/%E6%95%B0%E6%8D%AE%E6%B2%BB%E7%90%86%E4%B8%8E%E6%99%BA%E8%83%BDPipeline%E5%85%A8%E9%93%BE%E8%B7%AF%E8%AF%B4%E6%98%8E.md)
3. [项目说明文档.md](file:///D:/workplace/Crawl_cc/%E9%A1%B9%E7%9B%AE%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.md)

## 14. 一句话总结

当前项目不只是一个三模块爬虫或多模态 RAG Demo，而是一条面向汽车领域数据生产、治理、评测和回流的智能数据 Pipeline。
