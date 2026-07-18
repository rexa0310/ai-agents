from functools import lru_cache
from os import getenv
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


load_dotenv()

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class Settings(BaseModel):
    # --- Loaded from config.yaml (non-secret) ---
    local_model_id: str = "google/flan-t5-base"
    collection_name: str = "default"
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 3
    local_llm_max_new_tokens: int = 256
    local_llm_temperature: float = 0.2
    embedding_backend: str = "openrouter"
    openrouter_embedding_model: str = "openai/text-embedding-3-small"
    embedding_dimension: int = 1536
    vector_db_backend: str = "qdrant"
    qdrant_path: str = "./qdrant_data"
    qdrant_url: str = ""
    chroma_path: str = "./chroma_data"
    chroma_host: str = ""
    chroma_port: int = 8001
    faiss_path: str = "./faiss_data"
    mongo_db: str = "rag"
    mongo_collection: str = "ingested_documents"
    # --- Loaded from .env (secrets) ---
    openrouter_api_key: str = ""
    mongo_uri: str = ""


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data = _load_yaml(CONFIG_PATH)
    # Secrets come only from the environment (.env), never from config.yaml.
    data["openrouter_api_key"] = getenv("OPENROUTER_API_KEY", "")
    data["mongo_uri"] = getenv("MONGO_URI", "")
    return Settings(**data)
