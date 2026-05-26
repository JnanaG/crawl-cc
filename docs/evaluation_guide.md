# RAG 评估指南

本文档说明如何使用 Ragas 框架对 RAG 系统进行生产级评估，以及如何维护黄金测试集。

## 快速开始

### 1. 安装依赖

```bash
pip install ragas datasets
```

### 2. 运行评估

```bash
# 使用 ollama 本地模型
python rag_ragas_eval.py \
  --golden-set data/evaluation/golden_set.jsonl \
  --output data/evaluation/ragas_results.json \
  --embedding-provider ollama \
  --embedding-model nomic-embed-text \
  --llm-provider ollama \
  --llm-model qwen2.5:3b \
  --retrieval-mode hybrid

# 使用 OpenAI API（需要设置 OPENAI_API_KEY）
python rag_ragas_eval.py \
  --golden-set data/evaluation/golden_set.jsonl \
  --output data/evaluation/ragas_results.json \
  --embedding-provider openai_compatible \
  --embedding-model text-embedding-3-small \
  --llm-provider openai_compatible \
  --llm-model gpt-4o-mini
```

### 3. 检查阈值

```bash
python scripts/ci_eval_gate.py \
  --results data/evaluation/ragas_results.json
```

## 评估指标说明

### Faithfulness（忠实度）

**定义**：答案中的断言有多少比例能在检索到的上下文中找到支撑证据。

**计算方式**：
1. LLM 从答案中提取所有断言（claims）
2. 对每个断言，判断是否能从上下文中推断出来
3. `faithfulness = 支持的断言数 / 总断言数`

**阈值**：≥ 0.70（70% 的断言有证据支撑）

**示例**：
- 问题："比亚迪汉DM-i的油耗是多少"
- 答案："比亚迪汉DM-i的NEDC综合油耗为4.2L/100km，纯电续航121km"
- 上下文：包含"NEDC综合油耗4.2L/100km"和"纯电续航121km"
- 结果：2/2 = 1.0（完全忠实）

**低分原因**：
- 答案包含上下文中没有的信息（幻觉）
- 答案过度推断或编造细节

---

### Answer Relevancy（答案相关性）

**定义**：答案与问题的相关程度。

**计算方式**：
1. LLM 根据答案反向生成可能的问题
2. 计算生成问题与原问题的语义相似度
3. `answer_relevancy = avg(cosine_similarity(generated_q, original_q))`

**阈值**：≥ 0.65

**示例**：
- 问题："预算30万左右推荐哪些SUV"
- 好答案："30万左右的SUV推荐理想L7、问界M7、蔚来ES6..."（直接回答）
- 差答案："SUV是运动型多用途汽车，具有越野能力..."（答非所问）

**低分原因**：
- 答案偏离问题主题
- 答案包含大量无关信息
- 答案过于笼统或模糊
---

### Context Precision（上下文精确度）

**定义**：检索到的上下文中，相关 chunk 的排名是否靠前。

**计算方式**：
1. LLM 判断每个检索到的 chunk 是否与问题相关
2. 计算相关 chunk 在排序中的位置
3. 使用类似 Precision@K 的加权公式

**阈值**：≥ 0.60

**示例**：
- 问题："特斯拉Model 3的加速时间"
- 检索结果：
  1. "Model 3高性能版0-100km/h加速3.3秒"（相关）
  2. "Model 3续航里程606km"（不相关）
  3. "Model 3配备双电机四驱"（相关）
- 结果：相关 chunk 排在第1、3位，precision 较高

**低分原因**：
- 相关 chunk 排名靠后
- 检索到大量无关 chunk
- 检索策略（dense/sparse/hybrid）不合理

---

### Context Recall（上下文召回率）

**定义**：ground truth 中的关键信息有多少被检索到。

**计算方式**：
1. 从 ground truth 答案中提取关键断言
2. 判断每个断言是否能从检索上下文中推断
3. `context_recall = 被检索到的断言数 / 总断言数`

**阈值**：≥ 0.55

**示例**：
- 问题："理想L9和理想L8有什么区别"
- Ground truth："尺寸不同、座椅布局不同、价格不同、配置不同"
- 检索上下文：包含尺寸、价格、配置信息，但缺少座椅布局
- 结果：3/4 = 0.75

**低分原因**：
- top_k 设置过小，遗漏关键信息
- 检索策略偏向某类信息（如只检索到价格，遗漏配置）
- 数据源本身缺失 ground truth 中的信息

---

## 黄金测试集维护

### 文件格式

`data/evaluation/golden_set.jsonl`，每行一个 JSON 对象：

```json
{
  "question": "预算30万左右推荐哪些SUV",
  "expected_answer": "30万左右的SUV推荐包括：理想L7（31.98-37.98万）、问界M7（24.98-32.98万）...",
  "ground_truth_contexts": ["理想L7", "问界M7", "蔚来ES6"],
  "metadata": {
    "query_type": "推荐类",
    "constraints": ["价格区间"],
    "difficulty": "medium"
  }
}
```

### 字段说明

- **question**（必填）：用户问题
- **expected_answer**（必填）：期望的答案，用于 context_recall 计算
- **ground_truth_contexts**（可选）：期望检索到的车系/实体列表
- **metadata**（可选）：
  - `query_type`：推荐类/事实查询/对比类/场景适配/反向查询/配置查询
  - `constraints`：约束条件（价格区间、车身类型、动力类型等）
  - `difficulty`：easy/medium/hard

### 测试用例覆盖

当前黄金集包含 30 条测试用例，覆盖：

| 查询类型 | 数量 | 示例 |
|---------|------|----|
| 推荐类 | 8 | "预算30万左右推荐哪些SUV" |
| 事实查询 | 10 | "比亚迪汉DM-i的油耗是多少" |
| 对比类 | 6 | "特斯拉Model 3和小鹏P7哪个加速更快" |
| 反向查询 | 3 | "哪些车有CDC主动悬架" |
| 场景适配 | 1 | "坦克300适合越野吗" |
| 配置查询 | 2 | "红旗H9有哪些配置" |

### 添加新测试用例

1. 识别 RAG 系统的薄弱环节（如某类查询失败率高）
2. 编写测试用例，确保 `expected_answer` 准确且可验证
3. 追加到 `golden_set.jsonl`
4. 重新运行评估，观察指标变化

**建议**：
- 每次发现线上 bad case，转化为测试用例
- 定期从 `light_storage` 的查询日志中抽样，补充边缘 case
- 保持测试集多样性，避免过拟合某类查询

---

## CI 守门配置

### 阈值调整

编辑 `scripts/ci_eval_gate.py` 中的 `THRESHOLDS` 字典：

```python
THRESHOLDS = {
    "faithfulness": 0.70,
    "answer_relevancy": 0.65,
    "context_precision": 0.60,
    "context_recall": 0.55,
}
```

**调整原则**：
- 初期可放宽阈值（如 0.50-0.60），避免频繁阻塞
- 随着系统优化，逐步提高阈值
- 不同指标权重不同：faithfulness 最重要（防幻觉），其次 answer_relevancy

### GitHub Actions 配置

`.github/workflows/rag-eval.yml` 在以下情况触发：

- PR 修改了 RAG 相关代码（`rag_*.py`、`langchain_rag/`、`utils/`）
- PR 修改了黄金测试集（`data/evaluation/golden_set.jsonl`）
- 手动触发（workflow_dispatch）

**注意事项**：
1. 需要在 GitHub Secrets 中配置 `OPENAI_API_KEY` 和 `OPENAI_API_BASE`（如果使用 OpenAI）
2. 如果使用 ollama，需要在 runner 上安装 ollama 服务
3. 向量库构建耗时较长，建议缓存 `data/vector_store/` 目录

---

## 本地调试

### 单个测试用例调试

从黄金集中提取一条测试用例，手动执行 RAG 查询：

```bash
python rag_llm_demo.py query \
  --question "预算30万左右推荐哪些SUV" \
  --retrieval-mode hybrid \
  --embedding-provider ollama \
  --embedding-model nomic-embed-text \
  --llm-provider ollama \
  --llm-model qwen2.5:3b \
  --show-retrieval-debug \
  --show-context
```

观察：
- 检索到的 chunk 是否相关
- 答案是否有幻觉
- 引用编号是否正确

### 指标异常排查

| 指标低 | 可能原因 | 排查方向 |
|--------|---------|---------|
| faithfulness | 答案幻觉 | 检查 system prompt 是否强调"仅基于上下文"；LLM 是否倾向编造 |
| answer_relevancy | 答案偏题 | 检查 prompt 是否引导 LLM 直接回答；上下文是否包含无关信息 |
| context_precision | 检索排序差 | 调整 hybrid_search 权重；增加 reranker；检查 BM25 分词 |
| context_recall | 检索遗漏 | 增大 top_k；检查 chunking 是否过碎；检查数据源完整性 |

---

## 进阶：自定义评估指标

Ragas 支持自定义指标。示例：添加"引用准确性"指标（检查答案中的 [1][2] 是否对应正确的 chunk）。

编辑 `rag_ragas_eval.py`，在 `run_evaluation` 中添加：

```python
from ragas.metrics import Metric

class CitationAccuracy(Metric):
    def __call__(self, row):
        answer = row["answer"]
        contexts = row["contexts"]
        # 提取答案中的引用编号 [1], [2] 等
      # 检查编号是否在 contexts 范围内
        # 返回准确率
      pass

metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    CitationAccuracy(),  # 自定义指标
]
```

---

## 常见问题

### Q: Ragas 评估很慢怎么办？

A: Ragas 内部会多次调用 LLM（用于提取断言、生成问题等），每个测试用例可能调用 5-10 次。优化方法：
1. 使用更快的 LLM（如 gpt-4o-mini 而非 gpt-4）
2. 减少黄金集规模（CI 用 30 条，本地调试用 5-10 条）
3. 并行评估（修改 `prepare_ragas_dataset` 使用多线程）

### Q: 如何处理 Ragas 与 ollama 的兼容性问题？

A: Ragas 默认使用 OpenAI API。如果要用 ollama，需要：
1. 设置环境变量 `OPENAI_API_BASE=http://localhost:11434/v1`
2. 设置 `OPENAI_API_KEY=ollama`（任意非空值）
3. 确保 ollama 模型支持 OpenAI 兼容接口

### Q: 阈值设置多少合理？

A: 参考业界经验：
- faithfulness: 0.70-0.80（防幻觉是底线）
- answer_relevancy: 0.60-0.75
- context_precision: 0.50-0.70
- context_recall: 0.50-0.65

初期可设置较低阈值（0.50-0.60），随着优化逐步提高。

### Q: 黄金集需要多少条测试用例？

A: 
- 最小可行：20-30 条（覆盖主要查询类型）
- 推荐：100-200 条（覆盖边缘 case）
- 生产级：500+ 条（持续从线上日志补充）

---

## 参考资料

- [Ragas 官方文档](https://docs.ragas.io/)
- [Ragas GitHub](https://github.com/explodinggradients/ragas)
- [RAG 评估最佳实践](https://www.anthropic.com/research/contextual-retrieval)
