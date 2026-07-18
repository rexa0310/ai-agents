from typing import Callable

from rag.config import Settings
from rag.vector_stores.base import VectorStore

_BUILDERS: dict[str, Callable[[Settings], VectorStore]] = {}


def register(backend_name: str) -> Callable:
    def decorator(builder: Callable[[Settings], VectorStore]) -> Callable[[Settings], VectorStore]:
        _BUILDERS[backend_name] = builder
        return builder

    return decorator


def create_vector_store(settings: Settings) -> VectorStore:
    builder = _BUILDERS.get(settings.vector_db_backend)
    if builder is None:
        available = ", ".join(sorted(_BUILDERS)) or "none registered"
        raise ValueError(
            f"Unknown vector_db_backend '{settings.vector_db_backend}' (config.yaml). Available backends: {available}"
        )
    return builder(settings)


def available_backends() -> list[str]:
    return sorted(_BUILDERS)
