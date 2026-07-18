import json
from hashlib import sha1
from pathlib import Path

import faiss
import numpy as np

from rag.config import Settings
from rag.schemas import DocumentChunk, RetrievalResult
from rag.vector_stores.base import VectorStore
from rag.vector_stores.factory import register


def _stable_id(chunk_id: str) -> int:
    return int(sha1(chunk_id.encode("utf-8")).hexdigest()[:15], 16)


class FaissVectorStore(VectorStore):
    """Flat, exact inner-product index. Vectors must already be normalized for cosine similarity."""

    def __init__(self, storage_path: str, embedding_dimension: int) -> None:
        self.storage_dir = Path(storage_path)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dimension = embedding_dimension
        self._indexes: dict[str, "faiss.IndexIDMap"] = {}
        self._payloads: dict[str, dict[int, dict]] = {}

    def _index_path(self, collection_name: str) -> Path:
        return self.storage_dir / f"{collection_name}.index"

    def _payload_path(self, collection_name: str) -> Path:
        return self.storage_dir / f"{collection_name}.json"

    def _load(self, collection_name: str) -> None:
        if collection_name in self._indexes:
            return

        index_path = self._index_path(collection_name)
        payload_path = self._payload_path(collection_name)

        if index_path.exists() and payload_path.exists():
            index = faiss.read_index(str(index_path))
            payloads = {int(key): value for key, value in json.loads(payload_path.read_text()).items()}
        else:
            index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dimension))
            payloads = {}

        self._indexes[collection_name] = index
        self._payloads[collection_name] = payloads

    def _save(self, collection_name: str) -> None:
        faiss.write_index(self._indexes[collection_name], str(self._index_path(collection_name)))
        self._payload_path(collection_name).write_text(json.dumps(self._payloads[collection_name]))

    def upsert(self, collection_name: str, chunks: list[DocumentChunk]) -> None:
        self._load(collection_name)
        index = self._indexes[collection_name]
        payloads = self._payloads[collection_name]

        ids = np.array([_stable_id(chunk.chunk_id) for chunk in chunks], dtype="int64")
        index.remove_ids(ids)

        vectors = np.array([chunk.embedding for chunk in chunks], dtype="float32")
        index.add_with_ids(vectors, ids)

        for stable_id, chunk in zip(ids.tolist(), chunks):
            payloads[stable_id] = {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }

        self._save(collection_name)

    def similarity_search(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        self._load(collection_name)
        index = self._indexes[collection_name]
        payloads = self._payloads[collection_name]

        if index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype="float32")
        scores, ids = index.search(query, min(top_k, index.ntotal))

        results = []
        for score, stable_id in zip(scores[0], ids[0]):
            if stable_id == -1:
                continue
            payload = payloads[int(stable_id)]
            results.append(
                RetrievalResult(
                    chunk_id=payload["chunk_id"],
                    source_id=payload["source_id"],
                    text=payload["text"],
                    score=float(score),
                    metadata=payload.get("metadata", {}),
                )
            )
        return results


@register("faiss")
def _build(settings: Settings) -> FaissVectorStore:
    return FaissVectorStore(
        storage_path=settings.faiss_path,
        embedding_dimension=settings.embedding_dimension,
    )
