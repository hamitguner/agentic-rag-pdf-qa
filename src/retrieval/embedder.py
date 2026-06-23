"""OpenAI embedding wrapper for dense retrieval."""

from __future__ import annotations

import openai

from src.log import get_logger

logger = get_logger(__name__)


class EmbeddingError(Exception):
    """Raised when the embedding API call fails."""


class Embedder:
    """Thin wrapper around OpenAI's text-embedding endpoint."""

    def __init__(self, model: str, api_key: str) -> None:
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        logger.debug("Embedder initialized: model=%s", model)

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            EmbeddingError: If the API call fails.

        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts in a single API call.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors in the same order as the input.

        Raises:
            EmbeddingError: If the API call fails.

        """
        if not texts:
            return []
        logger.debug("Embedding batch: %d texts", len(texts))
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=texts,
            )
        except openai.OpenAIError as exc:
            logger.error("Embedding API error: %s", exc)
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

        vectors = [item.embedding for item in response.data]
        logger.debug("Received %d embeddings (dim=%d)", len(vectors), len(vectors[0]))
        return vectors
