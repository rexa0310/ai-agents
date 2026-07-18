import chromadb

from rag.config import Settings
from rag.schemas import DocumentChunk, RetrievalResult
from rag.vector_stores.base import VectorStore
from rag.vector_stores.factory import register


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        storage_path: str | None = None,
        host: str | None = None,
        port: int = 8001,
    ) -> None:
        if host:
            # Server mode: connect to a running Chroma (e.g. via docker compose).
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            # Embedded mode: single-process, persisted to a local folder.
            self.client = chromadb.PersistentClient(path=storage_path)

    def _collection(self, collection_name: str):
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, collection_name: str, chunks: list[DocumentChunk]) -> None:
        self._collection(collection_name).upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[{"source_id": chunk.source_id, **chunk.metadata} for chunk in chunks],
        )

    def similarity_search(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        collection = self._collection(collection_name)
        if collection.count() == 0:
            return []

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
        )

        return [
            RetrievalResult(
                chunk_id=chunk_id,
                source_id=metadata["source_id"],
                text=document,
                score=1.0 - distance,
                metadata={key: value for key, value in metadata.items() if key != "source_id"},
            )
            for chunk_id, document, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]


@register("chroma")
def _build(settings: Settings) -> ChromaVectorStore:
    return ChromaVectorStore(
        storage_path=settings.chroma_path,
        host=settings.chroma_host or None,
        port=settings.chroma_port,
    )
