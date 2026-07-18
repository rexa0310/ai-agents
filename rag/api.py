from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from rag.bootstrap import AppContainer, build_container
from rag.schemas import (
    HealthResponse,
    IngestionResult,
    QueryRequest,
    QueryResponse,
    TextIngestRequest,
)


def create_app() -> FastAPI:
    container = None
    app = FastAPI(
        title="RAG Boilerplate API",
        version="1.0.0",
        description="FastAPI-based RAG starter with text ingestion, file ingestion, Qdrant storage, and local LLM generation.",
    )

    @app.on_event("startup")
    async def startup() -> None:
        nonlocal container
        container = build_container()

    def get_container() -> "AppContainer":
        if container is None:
            raise RuntimeError("Application container is not initialized")
        return container

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        container = get_container()
        return HealthResponse(
            status="ok",
            collection_name=container.settings.collection_name,
            local_model_id=container.settings.local_model_id,
            vector_db_backend=container.settings.vector_db_backend,
        )

    @app.post("/ingest/text", response_model=IngestionResult)
    def ingest_text(payload: TextIngestRequest) -> IngestionResult:
        container = get_container()
        return container.ingestion_router.ingest_text(
            source_id=payload.source_id,
            raw_text=payload.raw_text,
        )

    @app.post("/ingest/file", response_model=IngestionResult)
    async def ingest_file(
        file: UploadFile = File(...),
        source_id: str | None = Form(default=None),
    ) -> IngestionResult:
        container = get_container()
        try:
            file_bytes = await file.read()
            return container.ingestion_router.ingest_file(
                filename=file.filename or "uploaded-file",
                file_bytes=file_bytes,
                source_id=source_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest) -> QueryResponse:
        container = get_container()
        return container.query_router.answer(question=payload.question)

    return app
