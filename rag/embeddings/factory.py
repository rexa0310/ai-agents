from typing import Callable

from rag.config import Settings
from rag.embeddings.base import Embedding

_BUILDERS: dict[str, Callable[[Settings], Embedding]] = {}


def register(backend_name: str) -> Callable:
    def decorator(builder: Callable[[Settings], Embedding]) -> Callable[[Settings], Embedding]:
        _BUILDERS[backend_name] = builder
        return builder

    return decorator


def create_embedding(settings: Settings) -> Embedding:
    builder = _BUILDERS.get(settings.embedding_backend)
    if builder is None:
        available = ", ".join(sorted(_BUILDERS)) or "none registered"
        raise ValueError(
            f"Unknown embedding_backend '{settings.embedding_backend}' (config.yaml). Available backends: {available}"
        )
    return builder(settings)


def available_embeddings() -> list[str]:
    return sorted(_BUILDERS)
