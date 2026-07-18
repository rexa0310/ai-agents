from rag.config import Settings, get_settings
from rag.embeddings import create_embedding
from rag.routers.ingestion_router import IngestionRouter
from rag.routers.llm_call_router import LLMCallRouter
from rag.routers.query_router import QueryRouter
from rag.services.document_registry import build_registry
from rag.vector_stores import create_vector_store


class AppContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vector_store = create_vector_store(settings)
        self.embedding = create_embedding(settings)
        self.registry = build_registry(settings)
        self.llm_router = LLMCallRouter(settings=settings)
        self.ingestion_router = IngestionRouter(
            settings=settings,
            vector_store=self.vector_store,
            embedding=self.embedding,
            registry=self.registry,
        )
        self.query_router = QueryRouter(
            settings=settings,
            vector_store=self.vector_store,
            embedding=self.embedding,
            llm_router=self.llm_router,
        )


def build_container() -> AppContainer:
    return AppContainer(settings=get_settings())
