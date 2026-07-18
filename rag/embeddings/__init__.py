from rag.embeddings.base import Embedding
from rag.embeddings.factory import available_embeddings, create_embedding

from rag.embeddings import openrouter  # noqa: F401 (registers backends)

__all__ = ["Embedding", "available_embeddings", "create_embedding"]
