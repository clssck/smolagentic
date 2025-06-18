"""Model factory for creating and managing LLM and embedding models.

This module provides a factory pattern for initializing language models
and embedding models from various providers with consistent interfaces.
"""

import logging
import os
from typing import Any

import litellm
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.llms.litellm import LiteLLM
from utils.config_loader import ModelType, get_config_loader

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory for creating and managing language models and embedding models.

    This class provides a centralized way to create, cache, and manage
    different types of models from various providers.
    """
    def __init__(self):
        """Initialize the model factory with configuration and empty caches."""
        self.config = get_config_loader()
        self._chat_models: dict[str, LLM] = {}
        self._embedding_models: dict[str, BaseEmbedding] = {}
        self._setup_litellm()

    def _setup_litellm(self):
        litellm.set_verbose = False

        # Set API keys for providers that are actually configured
        configured_providers = list(self.config.config.get("providers", {}).keys())
        for provider_name in configured_providers:
            try:
                api_key = self.config.get_api_key(provider_name)
                if api_key:
                    env_var = self.config.get_provider_config(provider_name)["api_key_env"]
                    os.environ[env_var] = api_key
            except ValueError:
                logger.warning("API key not found for provider: %s", provider_name)

    def get_chat_model(self, model_name: str) -> LLM:
        """Get or create a chat model instance.

        Args:
            model_name: Name of the chat model to retrieve.

        Returns:
            The LLM instance for the specified model.
        """
        if model_name in self._chat_models:
            return self._chat_models[model_name]

        model_config = self.config.get_model_config(model_name, ModelType.CHAT)
        provider_config = self.config.get_provider_config(model_config["provider"])

        # Create LiteLLM model
        litellm_model = LiteLLM(
            model=model_config["model_id"],
            temperature=model_config.get("temperature", 0.7),
            max_tokens=model_config.get("max_tokens", 4000),
            api_key=self.config.get_api_key(model_config["provider"]),
            api_base=provider_config["base_url"],
            timeout=30,
            context_window=model_config.get("context_window", 4000),
        )

        self._chat_models[model_name] = litellm_model
        logger.info("Loaded chat model: %s", model_name)
        return litellm_model

    def get_embedding_model(self, model_name: str) -> BaseEmbedding:
        """Get or create an embedding model instance.

        Args:
            model_name: Name of the embedding model to retrieve.

        Returns:
            The embedding model instance for the specified model.
        """
        if model_name in self._embedding_models:
            return self._embedding_models[model_name]

        model_config = self.config.get_model_config(model_name, ModelType.EMBEDDING)

        # Handle DeepInfra separately since LiteLLM doesn't support it for embeddings
        if model_config["provider"] == "deepinfra":
            from .deepinfra_embedding import DeepInfraEmbedding
            # Use custom DeepInfra embedding client
            embedding_model = DeepInfraEmbedding(
                model=model_config["model_id"],
                api_key=self.config.get_api_key(model_config["provider"]),
                embed_batch_size=10,
            )
        else:
            # Use LiteLLM for other providers
            embedding_model = LiteLLMEmbedding(
                model_name=model_config["model_id"],
                api_key=self.config.get_api_key(model_config["provider"]),
                embed_batch_size=10,
            )

        self._embedding_models[model_name] = embedding_model
        logger.info("Loaded embedding model: %s", model_name)
        return embedding_model

    def list_available_models(self, model_type: ModelType) -> list[str]:
        """List all available models of a specific type.

        Args:
            model_type: The type of models to list (CHAT or EMBEDDING).

        Returns:
            List of available model names.
        """
        return self.config.list_models(model_type)

    def get_model_info(self, model_name: str, model_type: ModelType) -> dict[str, Any]:
        """Get configuration information for a specific model.

        Args:
            model_name: Name of the model.
            model_type: Type of the model (CHAT or EMBEDDING).

        Returns:
            Dictionary containing model configuration details.
        """
        return self.config.get_model_config(model_name, model_type)

    def clear_cache(self):
        """Clear all cached model instances to free memory."""
        self._chat_models.clear()
        self._embedding_models.clear()
        logger.info("Model cache cleared")

# Global factory instance
_model_factory = None

def get_model_factory() -> ModelFactory:
    """Get or create a singleton ModelFactory instance.

    Returns:
        The ModelFactory singleton instance.
    """
    global _model_factory
    if _model_factory is None:
        _model_factory = ModelFactory()
    return _model_factory
