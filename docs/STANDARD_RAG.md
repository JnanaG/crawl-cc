# 标准 RAG 升级说明（真实 Embedding + FAISS + LLM）

## 组件

- 主脚本：`rag_llm_demo.py`
- Embedding：`utils/embedding_client.py`
- 向量库：`utils/faiss_vector_store.py`
- 生成器：`utils/llm_client.py`

## 安装依赖

```bash
pip install -r requirements.txt
```

新增关键依赖：

- `faiss-cpu`
- `fastembed`
- `sentence-transformers`
- `duckdb`

## 1) 构建向量库

推荐使用 OpenAI 兼容 Embedding API（稳定）：

```bash
set OPENAI_BASE_URL=https://your-openai-compatible-endpoint
set OPENAI_API_KEY=your_key
python rag_llm_demo.py build --embedding-provider openai_compatible --embedding-model text-embedding-3-small
```

若使用 Ollama Embedding：

```bash
set OLLAMA_BASE_URL=http://localhost:11434
python rag_llm_demo.py build --embedding-provider ollama --embedding-model nomic-embed-text
```

若使用本地 Python Embedding（可选）：

```bash
python rag_llm_demo.py build --embedding-provider fastembed --embedding-model BAAI/bge-small-zh-v1.5
```

说明：

- 默认会将 chunk 元数据和构建日志写入 `data/storage/rag.duckdb`。
- 可通过 `--storage-db` 指定路径，或 `--disable-storage` 关闭该能力。

## 2) 执行问答（检索 + LLM 生成）

默认采用 `hybrid` 检索链路：`FAISS 稠密召回 + BM25 稀疏召回 + RRF 融合重排`。

### OpenAI 兼容 LLM

```bash
set OPENAI_BASE_URL=https://your-openai-compatible-endpoint
set OPENAI_API_KEY=your_key
python rag_llm_demo.py query --question "预算30万左右，推荐哪些SUV？" --llm-provider openai_compatible --llm-model gpt-4o-mini
```

### Ollama 本地 LLM

```bash
set OLLAMA_BASE_URL=http://localhost:11434
python rag_llm_demo.py query --question "保时捷718适合什么人群？" --llm-provider ollama --llm-model qwen2.5:7b
```

查询时默认记录 `query_logs`（问题、命中结果、回答文本）到 DuckDB，便于后续评测与回溯。

可选检索参数（用于调优）：

```bash
python rag_llm_demo.py query ^
  --question "预算30万左右，推荐哪些SUV？" ^
  --retrieval-mode hybrid ^
  --dense-top-k 24 ^
  --sparse-top-k 40 ^
  --rerank-top-n 12 ^
  --show-retrieval-debug
```

可选模型级重排（Cross-Encoder）：

```bash
python rag_llm_demo.py query ^
  --question "预算30万左右，推荐哪些SUV？" ^
  --retrieval-mode hybrid ^
  --reranker-provider cross_encoder ^
  --reranker-model BAAI/bge-reranker-base ^
  --reranker-weight 0.65 ^
  --show-retrieval-debug
```

说明：

- `--reranker-provider none`（默认）只使用轻量融合重排。
- `--reranker-provider cross_encoder` 会在候选集上做模型级重排，精度更高但更慢。
- 若本机 `torch` / DLL 环境异常，脚本会自动降级到轻量重排（fail-open），避免任务中断。

## 输出

- 回答：由 LLM 基于 Top-K 检索上下文生成
- 参考来源：输出命中车系与链接，便于追溯

## 说明

- 脚本默认启用“仅基于上下文回答”的提示词约束，降低幻觉概率。
- 若要提升效果，建议增加重排模型（reranker）和评测集（Recall@K、引用正确率、人工打分）。
- 评估脚本与完整说明见：`docs/RAG_EVAL.md`（`rag_eval.py`）。
