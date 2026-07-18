from abc import ABC, abstractmethod

from rag.schemas import DocumentChunk, RetrievalResult


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, collection_name: str, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def similarity_search(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        raise NotImplementedError
