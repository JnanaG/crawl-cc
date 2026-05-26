# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 爬虫获取数据集
$env:TARGET_TRAINING_RECORDS=5000
$env:TARGET_SERIES_POOL=1200
$env:MAX_SERIES_EXPAND_REQUESTS=120
$env:MAX_CHUNK_TOKENS=100
.\.venv\Scripts\python.exe main.py


# rag 演示程序
# 构建索引
.\.venv\Scripts\python.exe rag_demo.py build

# 查询
.\.venv\Scripts\python.exe rag_demo.py query --question "预算30万左右推荐哪些SUV"
  
# rag_llm 演示程序
# 构建索引
.\.venv\Scripts\python.exe rag_llm_demo.py build --embedding-provider ollama --embedding-model nomic-embed-text

# 查询
.\.venv\Scripts\python.exe rag_llm_demo.py query --question "预算30万左右推荐哪些SUV" --retrieval-mode hybrid --embedding-provider ollama --embedding-model nomic-embed-text --llm-provider ollama --llm-model qwen2.5:3b



# langchain rag 演示程序
# 构建索引
.\.venv\Scripts\python.exe rag_langchain_demo.py build --embedding-provider ollama --embedding-model nomic-embed-text

# 查询
.\.venv\Scripts\python.exe rag_langchain_demo.py query --question "预算30万左右推荐哪些SUV" --retrieval-mode hybrid --embedding-provider ollama --embedding-model nomic-embed-text --llm-provider ollama --llm-model qwen2.5:3b

.\.venv\Scripts\python.exe rag_llm_demo.py query --question "预算20万左右推荐哪些自动挡车" --embedding-provider ollama --embedding-model nomic-embed-text --llm-provider ollama --llm-model qwen2.5:3b --show-context



# 查询
# 1) 先 build（用 sentence_transformers 做 embedding）
.\.venv\Scripts\python.exe rag_llm_demo.py build `
  --embedding-provider sentence_transformers `
  --embedding-model D:\models\bge-small-zh-v1.5 `
  --batch-size 128

# 2) 再 query（混合检索 + cross_encoder 重排）
.\.venv\Scripts\python.exe rag_llm_demo.py query `
  --question "预算30万左右推荐哪些SUV" `
  --retrieval-mode hybrid `
  --embedding-provider sentence_transformers `
  --embedding-model D:\models\bge-small-zh-v1.5 `
  --reranker-provider cross_encoder `
  --reranker-model D:\models\bge-reranker-base `

  --show-retrieval-debug `
  --llm-provider ollama `
  --llm-model qwen2.5:3b

.\.venv\Scripts\python.exe rag_llm_demo.py query `
  --question "预算30万左右推荐哪些SUV" `
  --retrieval-mode hybrid `
  --embedding-provider sentence_transformers `
  --embedding-model D:\models\bge-small-zh-v1.5 `
  --show-retrieval-debug `
  --llm-provider ollama `
  --llm-model qwen2.5:3b
