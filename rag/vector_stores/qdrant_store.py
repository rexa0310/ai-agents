from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.config import Settings
from rag.schemas import DocumentChunk, RetrievalResult
from rag.vector_stores.base import VectorStore
from rag.vector_stores.factory import register


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        embedding_dimension: int,
        storage_path: str | None = None,
        url: str | None = None,
    ) -> None:
        self.embedding_dimension = embedding_dimension
        if url:
            # Server mode: connect to a running Qdrant (e.g. via docker compose).
            self.client = QdrantClient(url=url)
        else:
            # Embedded mode: single-process, persisted to a local folder.
            storage_dir = Path(storage_path)
            storage_dir.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(storage_dir))

    def _ensure_collection(self, collection_name: str) -> None:
        collections = self.client.get_collections().collections
        existing_names = {collection.name for collection in collections}

        if collection_name in existing_names:
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dimension,
                distance=Distance.COSINE,
            ),
        )

    def upsert(self, collection_name: str, chunks: list[DocumentChunk]) -> None:
        self._ensure_collection(collection_name)
        points = [
            PointStruct(
                id=str(uuid4()),
                vector=chunk.embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                },
            )
            for chunk in chunks
        ]
        self.client.upsert(collection_name=collection_name, points=points)

    def similarity_search(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        self._ensure_collection(collection_name)
        hits = self.client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,
        ).points

        return [
            RetrievalResult(
                chunk_id=hit.payload["chunk_id"],
                source_id=hit.payload["source_id"],
                text=hit.payload["text"],
                score=float(hit.score or 0.0),
                metadata=hit.payload.get("metadata", {}),
            )
            for hit in hits
        ]


@register("qdrant")
def _build(settings: Settings) -> QdrantVectorStore:
    return QdrantVectorStore(
        embedding_dimension=settings.embedding_dimension,
        storage_path=settings.qdrant_path,
        url=settings.qdrant_url or None,
    )
