# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Chinese-language data pipeline that scrapes car-series data from Dongchedi (懂车帝), cleans it into structured records, chunks the result into LLM training samples, and provides three RAG demo apps over the resulting corpus (a from-scratch FAISS pipeline, an LLM-augmented variant with hybrid retrieval + reranking, and a LangChain-based variant).

## Common Commands

The project uses a Windows venv at `.venv`. The canonical command examples are in `run.sh` (PowerShell, despite the `.sh` name).

```powershell
# Activate venv (PowerShell)
.\.venv\Scripts\Activate.ps1

# Run the full scrape+clean+chunk pipeline (writes data/processed/dongchedi_training_data.jsonl)
$env:TARGET_TRAINING_RECORDS=5000
$env:TARGET_SERIES_POOL=1200
$env:MAX_SERIES_EXPAND_REQUESTS=120
$env:MAX_CHUNK_TOKENS=100
python main.py

# Standard RAG demo (FAISS only, no LLM)
python rag_demo.py build
python rag_demo.py query --question "预算30万左右推荐哪些SUV"

# RAG + LLM demo (hybrid retrieval, optional reranker)
python rag_llm_demo.py build --embedding-provider ollama --embedding-model nomic-embed-text
python rag_llm_demo.py query --question "..." --retrieval-mode hybrid \
  --embedding-provider ollama --embedding-model nomic-embed-text \
  --llm-provider ollama --llm-model qwen2.5:3b

# LangChain RAG variant
python rag_langchain_demo.py build --embedding-provider ollama --embedding-model nomic-embed-text
python rag_langchain_demo.py query --question "..." --retrieval-mode hybrid \
  --embedding-provider ollama --embedding-model nomic-embed-text \
  --llm-provider ollama --llm-model qwen2.5:3b

# Evaluation harness (legacy)
python rag_eval.py

# Ragas evaluation (production-grade)
python rag_ragas_eval.py --golden-set data/evaluation/golden_set.jsonl \
  --embedding-provider ollama --embedding-model nomic-embed-text \
  --llm-provider ollama --llm-model qwen2.5:3b

# CI gate check (local)
python scripts/ci_eval_gate.py --results data/evaluation/ragas_results.json
```

No test suite, lint config, or build system is present — execute scripts directly with the venv's Python.

## Architecture

The pipeline has two distinct phases that share `data/` as a contract:

### Phase 1 — Ingestion (`main.py`)

`main.py` orchestrates the end-to-end ingestion. The flow is:

1. `scraper/dcd_scraper.py::DongchediScraper.collect_series_ids` expands a pool of target series IDs.
2. For each series: `fetch_series_data` (rate-limited, retried) → `cleaner/dcd_cleaner.py::DongchediCleaner` produces both a structured JSON record and a Markdown rendering → `utils/llm_processor.py::LLMDataProcessor.chunk_text` splits Markdown into token-bounded chunks (default 100 tokens, controlled by `MAX_CHUNK_TOKENS`).
3. Every chunk is validated by `quality/data_quality.py::DataQualityEngine.validate_training_item`; rejected chunks are dropped and counted.
4. `utils/job_state.py::JobStateStore` persists per-series status in `data/state/job_state.json` to support resumable runs — on rerun, previously-successful series are skipped but their cleaned JSON is replayed back into the training set.
5. Outputs: JSONL + Parquet training file in `data/processed/`, cleaned JSON/Markdown per series in `data/cleaned/dongchedi/`, and quality + task-stats reports in `data/reports/`.

Key env vars (read in `_env_int`): `TARGET_TRAINING_RECORDS`, `TARGET_SERIES_POOL`, `MAX_CHUNK_TOKENS`, `MAX_SERIES_EXPAND_REQUESTS`.

### Phase 2 — RAG (three parallel implementations)

All three read `data/processed/dongchedi_training_data.jsonl` and write into `data/vector_store/`.

- `rag_demo.py` — minimal FAISS-only retrieval demo, no LLM.
- `rag_llm_demo.py` — production-style pipeline. Implements **hybrid retrieval**: dense FAISS search + an in-file `LightweightBM25` sparse retriever, fused via RRF + min-max-normalized scores + query-term coverage, optionally rescored by a cross-encoder reranker. See `hybrid_search()` for the full scoring formula. Calls an LLM (`utils/llm_client.py`) with a strict citation-required system prompt.
- `langchain_rag/` — LangChain-backed variant exposed via `rag_langchain_demo.py` (which just delegates to `langchain_rag.cli.main`). Module layout: `config.py` (path defaults), `core.py` (build/query implementations), `cli.py` (argparse).

The two LLM-using variants share the same provider abstractions in `utils/`:
- `embedding_client.py` — supports `openai_compatible`, `ollama`, `fastembed`, `sentence_transformers`.
- `llm_client.py` — supports `openai_compatible`, `ollama`.
- `reranker_client.py` — `none` or `cross_encoder` (HF model path).
- `faiss_vector_store.py` — the standalone FAISS wrapper used by `rag_demo`/`rag_llm_demo`.
- `light_storage.py::LightRAGStorage` — DuckDB-backed audit log (`data/storage/rag.duckdb`) recording build runs and queries; disabled with `--disable-storage`.

### Data layout (canonical)

```
data/
  raw/dongchedi/              # raw HTML + JSON from scraper
  cleaned/dongchedi/json/     # one cleaned JSON per series
  cleaned/dongchedi/markdown/ # one Markdown per series
  processed/                  # JSONL + Parquet training samples
  vector_store/faiss/         # FAISS indexes (standard pipeline)
  vector_store/langchain/     # FAISS indexes (LangChain pipeline)
  state/job_state.json        # resumable scrape state
  storage/rag.duckdb          # query/build audit log
  reports/                    # quality + run stats
  logs/                       # loguru per-run logs
```

## Notes for editing

- Code comments, log messages, docstrings, and the resume-from-state contract are in Chinese — preserve the language when editing.
- Resumability: `JobStateStore` distinguishes `running`/`success`/`failed`/`skipped`. When changing the cleaner or chunker, be aware that previously-successful series will be re-loaded from `data/cleaned/.../json/` rather than re-scraped; delete `job_state.json` (or the relevant entries) to force a clean re-run.
- `rag_llm_demo.py` and `langchain_rag/core.py` contain parallel implementations of hybrid search; changes to scoring logic generally need to be made in both.
- Reference docs (in Chinese) live at repo root: `开发计划.md`, `项目说明文档.md`, `简历项目-精炼版.md`, `简历项目描述.md`.
