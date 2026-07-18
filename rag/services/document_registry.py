from datetime import datetime, timezone

from pymongo import ASCENDING, MongoClient

from rag.config import Settings


class DocumentRegistry:
    """Tracks ingested documents in MongoDB to dedupe re-ingestion and record metadata.

    Uniqueness is per (doc_hash, embedding_model, collection_name): the same file
    re-uploaded under the same embedding model is skipped, but switching models
    re-ingests it (the vectors differ and must be recomputed).
    """

    def __init__(self, uri: str, db_name: str, collection_name: str) -> None:
        self._client = MongoClient(uri)
        self._collection = self._client[db_name][collection_name]
        self._collection.create_index(
            [
                ("doc_hash", ASCENDING),
                ("embedding_model", ASCENDING),
                ("collection_name", ASCENDING),
            ],
            unique=True,
        )

    def find(self, *, doc_hash: str, embedding_model: str, collection_name: str) -> dict | None:
        return self._collection.find_one(
            {
                "doc_hash": doc_hash,
                "embedding_model": embedding_model,
                "collection_name": collection_name,
            },
            {"_id": 0},
        )

    def record(
        self,
        *,
        doc_hash: str,
        embedding_model: str,
        collection_name: str,
        source_id: str,
        file_name: str,
        file_type: str,
        size_bytes: int,
        total_chunks: int,
    ) -> None:
        key = {
            "doc_hash": doc_hash,
            "embedding_model": embedding_model,
            "collection_name": collection_name,
        }
        document = {
            **key,
            "source_id": source_id,
            "file_name": file_name,
            "file_type": file_type,
            "size_bytes": size_bytes,
            "total_chunks": total_chunks,
            "ingested_at": datetime.now(timezone.utc),
        }
        self._collection.update_one(key, {"$set": document}, upsert=True)

    def list_documents(self) -> list[dict]:
        return list(self._collection.find({}, {"_id": 0}).sort("ingested_at", -1))


def build_registry(settings: Settings) -> "DocumentRegistry | None":
    if not settings.mongo_uri:
        return None
    return DocumentRegistry(
        uri=settings.mongo_uri,
        db_name=settings.mongo_db,
        collection_name=settings.mongo_collection,
    )
