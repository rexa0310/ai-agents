from rag.config import Settings
from rag.embeddings import Embedding
from rag.routers.llm_call_router import LLMCallRouter
from rag.schemas import QueryResponse
from rag.services.prompt_builder import build_grounded_prompt
from rag.services.retrieval_service import RetrievalService
from rag.vector_stores import VectorStore


class QueryRouter:
    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        embedding: Embedding,
        llm_router: LLMCallRouter,
    ) -> None:
        self.settings = settings
        self.embedding = embedding
        self.llm_router = llm_router
        self.retrieval_service = RetrievalService(
            vector_store=vector_store,
            embedding=embedding,
            collection_name=settings.collection_name,
            top_k=settings.retrieval_top_k,
        )

    def answer(self, question: str) -> QueryResponse:
        retrieved_chunks = self.retrieval_service.retrieve(question=question)
        prompt = build_grounded_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )
        answer, provider_used = self.llm_router.generate(prompt=prompt)

        return QueryResponse(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            provider_used=provider_used,
            embedding_model=self.embedding.model_id,
        )
