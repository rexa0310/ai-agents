from rag.config import Settings
from rag.embeddings import Embedding
from rag.services.document_registry import DocumentRegistry
from rag.services.ingestion_service import IngestionService
from rag.vector_stores import VectorStore


class IngestionRouter:
    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        embedding: Embedding,
        registry: DocumentRegistry | None = None,
    ) -> None:
        self.service = IngestionService(
            vector_store=vector_store,
            embedding=embedding,
            collection_name=settings.collection_name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            registry=registry,
        )

    def ingest_text(self, source_id: str, raw_text: str):
        return self.service.ingest_text(source_id=source_id, raw_text=raw_text)

    def ingest_file(
        self,
        filename: str,
        file_bytes: bytes,
        source_id: str | None = None,
    ):
        return self.service.ingest_file(
            filename=filename,
            file_bytes=file_bytes,
            source_id=source_id,
        )
