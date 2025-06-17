import logging
from typing import List, Optional, Any
from openai import OpenAI
from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)

class DeepInfraEmbedding(BaseEmbedding):
    """Custom embedding class for DeepInfra using OpenAI-compatible API"""
    
    def __init__(
        self,
        model: str = "Qwen/Qwen3-Embedding-8B",
        api_key: Optional[str] = None,
        api_base: str = "https://api.deepinfra.com/v1/openai",
        embed_batch_size: int = 10,
        **kwargs: Any,
    ):
        # Set attributes after calling super init to avoid Pydantic issues
        super().__init__(
            embed_batch_size=embed_batch_size,
            **kwargs,
        )
        
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, 'client', OpenAI(
            api_key=api_key,
            base_url=api_base
        ))
    
    @classmethod
    def class_name(cls) -> str:
        return "DeepInfraEmbedding"
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a single query"""
        return self._get_text_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            response = self.client.embeddings.create(
                model=self._model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        try:
            response = self.client.embeddings.create(
                model=self._model,
                input=texts,
                encoding_format="float"
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of get_query_embedding"""
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of get_text_embedding"""
        return self._get_text_embedding(text)