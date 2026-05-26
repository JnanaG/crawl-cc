import os

DEFAULT_INPUT = os.path.join("data", "processed", "dongchedi_training_data.jsonl")
DEFAULT_STORE_DIR = os.path.join("data", "vector_store", "langchain")
DEFAULT_INDEX_NAME = "dongchedi_langchain"
DEFAULT_RECORDS = os.path.join(DEFAULT_STORE_DIR, "dongchedi_records.jsonl")
DEFAULT_META = os.path.join(DEFAULT_STORE_DIR, "dongchedi_meta.json")
DEFAULT_STORAGE_DB = os.path.join("data", "storage", "rag.duckdb")
