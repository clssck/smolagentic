import os
import logging
from typing import Dict, Any, Optional, List
from litellm import completion, embedding
import litellm
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.litellm import LiteLLM
from llama_index.embeddings.litellm import LiteLLMEmbedding

from utils.config_loader import get_config_loader, ModelType

logger = logging.getLogger(__name__)

class ModelFactory:
    def __init__(self):
        self.config = get_config_loader()
        self._chat_models: Dict[str, LLM] = {}
        self._embedding_models: Dict[str, BaseEmbedding] = {}
        self._setup_litellm()
    
    def _setup_litellm(self):
        litellm.set_verbose = False
        
        # Set API keys for providers if available
        for provider_name in ["openrouter", "deepinfra", "openai"]:
            try:
                api_key = self.config.get_api_key(provider_name)
                if api_key:
                    env_var = self.config.get_provider_config(provider_name)["api_key_env"]
                    os.environ[env_var] = api_key
            except ValueError:
                logger.warning(f"API key not found for provider: {provider_name}")
    
    def get_chat_model(self, model_name: str) -> LLM:
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
            timeout=30
        )
        
        self._chat_models[model_name] = litellm_model
        logger.info(f"Loaded chat model: {model_name}")
        return litellm_model
    
    def get_embedding_model(self, model_name: str) -> BaseEmbedding:
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
                embed_batch_size=10
            )
        else:
            # Use LiteLLM for other providers
            embedding_model = LiteLLMEmbedding(
                model_name=model_config["model_id"],
                api_key=self.config.get_api_key(model_config["provider"]),
                embed_batch_size=10
            )
        
        self._embedding_models[model_name] = embedding_model
        logger.info(f"Loaded embedding model: {model_name}")
        return embedding_model
    
    def list_available_models(self, model_type: ModelType) -> List[str]:
        return self.config.list_models(model_type)
    
    def get_model_info(self, model_name: str, model_type: ModelType) -> Dict[str, Any]:
        return self.config.get_model_config(model_name, model_type)
    
    def clear_cache(self):
        self._chat_models.clear()
        self._embedding_models.clear()
        logger.info("Model cache cleared")

# Global factory instance
_model_factory = None

def get_model_factory() -> ModelFactory:
    global _model_factory
    if _model_factory is None:
        _model_factory = ModelFactory()
    return _model_factory