from abc import ABC, abstractmethod


class Embedding(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable identifier of the embedding model, stored per document for tracking."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]
