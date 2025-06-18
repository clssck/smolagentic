"""Custom embedding implementation for DeepInfra API.

This module provides a custom embedding class that uses DeepInfra's
OpenAI-compatible API for generating text embeddings.
"""

import logging
from typing import Any

from llama_index.core.embeddings import BaseEmbedding
from openai import OpenAI

logger = logging.getLogger(__name__)

class DeepInfraEmbedding(BaseEmbedding):
    """Custom embedding class for DeepInfra using OpenAI-compatible API"""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Embedding-8B",
        api_key: str | None = None,
        api_base: str = "https://api.deepinfra.com/v1/openai",
        embed_batch_size: int = 10,
        **kwargs: Any,
    ):
        """Initialize DeepInfra embedding client.

        Args:
            model: Name of the embedding model to use.
            api_key: API key for DeepInfra access.
            api_base: Base URL for the DeepInfra API.
            embed_batch_size: Number of texts to process in each batch.
            **kwargs: Additional arguments passed to BaseEmbedding.
        """
        # Set attributes after calling super init to avoid Pydantic issues
        super().__init__(
            embed_batch_size=embed_batch_size,
            **kwargs,
        )

        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "client", OpenAI(
            api_key=api_key,
            base_url=api_base,
        ))

    @classmethod
    def class_name(cls) -> str:
        """Return the class name for this embedding model.

        Returns:
            The class name as a string.
        """
        return "DeepInfraEmbedding"

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a single query"""
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text string.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the text embedding.

        Raises:
            Exception: If the API call fails.
        """
        """Get embedding for a single text"""
        try:
            response = self.client.embeddings.create(
                model=self._model,
                input=text,
                encoding_format="float",
            )
            return response.data[0].embedding
        except Exception as e:
            logger.exception("Error getting embedding: %s", e)
            raise

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts"""
        try:
            response = self.client.embeddings.create(
                model=self._model,
                input=texts,
                encoding_format="float",
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.exception("Error getting embeddings: %s", e)
            raise

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Async version of get_query_embedding"""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Async version of get_text_embedding"""
        return self._get_text_embedding(text)
