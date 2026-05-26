# 懂车帝数据 RAG Demo（最小可用）

## 目标

基于你已生成的训练数据 `data/processed/dongchedi_training_data.jsonl`，
快速打通一条可运行的 RAG 链路：

1. 训练样本入本地向量库
2. 接收问题并检索 Top-K 相关片段
3. 输出带来源的检索增强回答

## 实现说明

- 脚本：`rag_demo.py`
- 向量库：`utils/simple_vector_store.py`
- 向量化方式：哈希向量（离线、无外部 API 依赖）
- 索引文件：
  - `data/vector_store/dongchedi_hashvec.npz`
  - `data/vector_store/dongchedi_records.jsonl`

## 使用方式

### 1) 构建向量库

```bash
python rag_demo.py build
```

可选参数：

```bash
python rag_demo.py build --input data/processed/dongchedi_training_data.jsonl --dim 768
```

### 2) 执行问答检索

```bash
python rag_demo.py query --question "预算30万左右，推荐哪些SUV？"
```

可选参数：

```bash
python rag_demo.py query --question "保时捷718的价格和定位是什么？" --top-k 6
```

## 输出格式

- 回答区：展示检索增强后的摘要性回答
- 参考来源：展示命中的车系标题、URL、相似度分数

## 注意事项

- 当前为最小可用版本，回答生成采用规则化拼接，重点是打通数据链路。
- 若进入生产阶段，建议替换为：
  - Embedding：`bge-m3` / `text-embedding-3-large` 等
  - 向量库：FAISS / Milvus / PGVector / Chroma
  - 生成器：接入真实 LLM（并使用 prompt+引用约束）
