import requests

from rag.config import Settings
from rag.embeddings.base import Embedding
from rag.embeddings.factory import register

_ENDPOINT = "https://openrouter.ai/api/v1/embeddings"


class OpenRouterEmbedding(Embedding):
    """Hosted embeddings via OpenRouter's OpenAI-compatible endpoint. No local model."""

    def __init__(self, api_key: str, model: str, dimension: int, timeout: float = 30.0) -> None:
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required for the 'openrouter' embedding backend."
            )
        self._api_key = api_key
        self._model = model
        self._dimension = dimension
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def _post(self, model_input) -> list[dict]:
        response = requests.post(
            _ENDPOINT,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self._model, "input": model_input},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()["data"]

    def _validate_dimension(self, vector: list[float]) -> None:
        if len(vector) != self._dimension:
            raise ValueError(
                f"Embedding model '{self._model}' returned dimension {len(vector)}, but "
                f"embedding_dimension is {self._dimension} in config.yaml. Set "
                f"embedding_dimension: {len(vector)} and re-ingest into a fresh collection."
            )

    def embed(self, text: str) -> list[float]:
        vector = self._post(text)[0]["embedding"]
        self._validate_dimension(vector)
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        items = sorted(self._post(texts), key=lambda item: item["index"])
        vectors = [item["embedding"] for item in items]
        self._validate_dimension(vectors[0])
        return vectors


@register("openrouter")
def _build(settings: Settings) -> OpenRouterEmbedding:
    return OpenRouterEmbedding(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_embedding_model,
        dimension=settings.embedding_dimension,
    )
