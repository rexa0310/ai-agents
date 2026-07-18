from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    source_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunk_id: str
    source_id: str
    text: str
    score: float
    metadata: dict[str, str] = Field(default_factory=dict)


class TextIngestRequest(BaseModel):
    source_id: str
    raw_text: str


class IngestionResult(BaseModel):
    source_id: str
    collection_name: str
    total_chunks: int
    parser_used: str
    embedding_model: str
    doc_hash: str
    skipped: bool = False


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: list[RetrievalResult]
    provider_used: str
    embedding_model: str


class HealthResponse(BaseModel):
    status: str
    collection_name: str
    local_model_id: str
    vector_db_backend: str
