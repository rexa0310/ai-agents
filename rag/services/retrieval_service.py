from rag.embeddings import Embedding
from rag.schemas import RetrievalResult
from rag.vector_stores import VectorStore


class RetrievalService:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding: Embedding,
        collection_name: str,
        top_k: int,
    ) -> None:
        self.vector_store = vector_store
        self.embedding = embedding
        self.collection_name = collection_name
        self.top_k = top_k

    def retrieve(self, question: str) -> list[RetrievalResult]:
        query_embedding = self.embedding.embed(question)
        return self.vector_store.similarity_search(
            collection_name=self.collection_name,
            query_embedding=query_embedding,
            top_k=self.top_k,
        )
