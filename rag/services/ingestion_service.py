from hashlib import sha256
from pathlib import Path

from rag.chunking import chunk_text
from rag.embeddings import Embedding
from rag.schemas import DocumentChunk, IngestionResult
from rag.services.document_parser import DocumentParserService
from rag.services.document_registry import DocumentRegistry
from rag.vector_stores import VectorStore


class IngestionService:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding: Embedding,
        collection_name: str,
        chunk_size: int,
        chunk_overlap: int,
        registry: DocumentRegistry | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding = embedding
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.registry = registry
        self.document_parser = DocumentParserService()

    def ingest_text(self, source_id: str, raw_text: str) -> IngestionResult:
        data = raw_text.encode("utf-8")
        return self._ingest(
            source_id=source_id,
            raw_text=raw_text,
            parser_used="raw-text",
            file_name=source_id,
            file_type="text",
            size_bytes=len(data),
            doc_hash=sha256(data).hexdigest(),
        )

    def ingest_file(
        self,
        filename: str,
        file_bytes: bytes,
        source_id: str | None = None,
    ) -> IngestionResult:
        extracted_text, parser_used = self.document_parser.parse(
            filename=filename,
            file_bytes=file_bytes,
        )
        resolved_source_id = source_id or Path(filename).stem
        return self._ingest(
            source_id=resolved_source_id,
            raw_text=extracted_text,
            parser_used=parser_used,
            file_name=filename,
            file_type=Path(filename).suffix.lower().lstrip(".") or "unknown",
            size_bytes=len(file_bytes),
            doc_hash=sha256(file_bytes).hexdigest(),
        )

    def _ingest(
        self,
        *,
        source_id: str,
        raw_text: str,
        parser_used: str,
        file_name: str,
        file_type: str,
        size_bytes: int,
        doc_hash: str,
    ) -> IngestionResult:
        embedding_model = self.embedding.model_id

        if self.registry is not None:
            existing = self.registry.find(
                doc_hash=doc_hash,
                embedding_model=embedding_model,
                collection_name=self.collection_name,
            )
            if existing is not None:
                # Same document + same embedding model already ingested — skip re-embedding.
                return IngestionResult(
                    source_id=existing["source_id"],
                    collection_name=self.collection_name,
                    total_chunks=existing["total_chunks"],
                    parser_used=parser_used,
                    embedding_model=embedding_model,
                    doc_hash=doc_hash,
                    skipped=True,
                )

        chunks = chunk_text(
            text=raw_text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        embeddings = self.embedding.embed_batch(chunks) if chunks else []

        document_chunks = [
            DocumentChunk(
                chunk_id=f"{source_id}-{index}",
                source_id=source_id,
                text=chunk,
                embedding=embedding,
                metadata={
                    "position": str(index),
                    "parser_used": parser_used,
                    "embedding_model": embedding_model,
                },
            )
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings), start=1)
        ]

        if document_chunks:
            self.vector_store.upsert(self.collection_name, document_chunks)

        if self.registry is not None:
            self.registry.record(
                doc_hash=doc_hash,
                embedding_model=embedding_model,
                collection_name=self.collection_name,
                source_id=source_id,
                file_name=file_name,
                file_type=file_type,
                size_bytes=size_bytes,
                total_chunks=len(document_chunks),
            )

        return IngestionResult(
            source_id=source_id,
            collection_name=self.collection_name,
            total_chunks=len(document_chunks),
            parser_used=parser_used,
            embedding_model=embedding_model,
            doc_hash=doc_hash,
            skipped=False,
        )
