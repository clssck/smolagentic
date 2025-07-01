"""
Embedding Service for Agentic RAG System

Provides text embedding functionality using DeepInfra Qwen3-Embedding-8B model.
"""

import asyncio
import hashlib
import logging
import requests
import time
from typing import List, Union, Dict, Any
from .config import Config

try:
    import aiohttp
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using DeepInfra Qwen3 model."""
    
    def __init__(self, config: Config = None, cache_size: int = 1000):
        """
        Initialize embedding service.
        
        Args:
            config: Configuration object with DeepInfra settings
            cache_size: Maximum number of embeddings to cache
        """
        self.config = config or Config()
        
        if not self.config.DEEPINFRA_TOKEN:
            raise ValueError("DEEPINFRA_TOKEN not configured")
        
        self.api_url = "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Embedding-8B"
        self.headers = {
            'Authorization': f'Bearer {self.config.DEEPINFRA_TOKEN}',
            'Content-Type': 'application/json'
        }
        
        # Enhanced caching with connection pooling
        self.embedding_cache: Dict[str, List[float]] = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Optimized connection pooling for better performance
        self.connector = None
        self.connector_config = {
            'limit': 100,              # Total connection limit
            'limit_per_host': 10,      # Per-host limit  
            'keepalive_timeout': 300,  # Keep connections alive longer
            'enable_cleanup_closed': True,
            'use_dns_cache': True,     # DNS caching
            'ttl_dns_cache': 300       # DNS cache TTL
        } if ASYNC_AVAILABLE else None
        
        # Performance tracking
        self.total_requests = 0
        self.total_time = 0.0
        
        logger.info(f"Initialized embedding service with model: {self.config.DEEPINFRA_EMBEDDING_MODEL}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _manage_cache_size(self):
        """Remove oldest entries if cache exceeds size limit."""
        if len(self.embedding_cache) > self.cache_size:
            # Remove 20% of oldest entries
            keys_to_remove = list(self.embedding_cache.keys())[:int(self.cache_size * 0.2)]
            for key in keys_to_remove:
                del self.embedding_cache[key]

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            List of float values representing the embedding
        """
        # Check cache first
        cache_key = self._get_cache_key(text)
        
        if cache_key in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[cache_key]
        
        # Cache miss - generate embedding
        self.cache_misses += 1
        embedding = self.embed_texts([text])[0]
        
        # Store in cache
        self.embedding_cache[cache_key] = embedding
        self._manage_cache_size()
        
        return embedding
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with retry logic.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        retries = 3
        for attempt in range(retries):
            try:
                start_time = time.time()
                payload = {'inputs': texts}
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=45  # Increased timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embeddings = result.get('embeddings', [])
                    
                    if len(embeddings) != len(texts):
                        raise ValueError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
                    
                    # Track performance
                    self.total_requests += 1
                    self.total_time += time.time() - start_time
                    
                    logger.debug(f"Generated {len(embeddings)} embeddings in {time.time() - start_time:.2f}s")
                    return embeddings
                else:
                    error_msg = f"DeepInfra API error {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
            except Exception as e:
                wait_time = (2 ** attempt) * 1  # Exponential backoff
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                
                if attempt < retries - 1:
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {retries} embedding attempts failed")
                    raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query with preprocessing.
        
        Args:
            query: Query text to embed
            
        Returns:
            Query embedding as list of floats
        """
        # Preprocess query for better embedding
        processed_query = self._preprocess_query(query)
        return self.embed_text(processed_query)
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better embedding quality."""
        query = query.lower().strip()
        
        # Remove common search prefixes
        prefixes = [
            'search for', 'find', 'look up', 'get information about',
            'tell me about', 'what is', 'explain', 'show me'
        ]
        
        for prefix in prefixes:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()
                break
        
        # Add context for very short queries
        if len(query.split()) < 3:
            query = f"information about {query}"
        
        return query
    
    async def embed_text_async(self, text: str) -> List[float]:
        """
        Generate embedding asynchronously with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Check cache first
        cache_key = self._get_cache_key(text)
        
        if cache_key in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[cache_key]
        
        # Cache miss - generate embedding
        self.cache_misses += 1
        embedding = await self._generate_embedding_async(text)
        
        # Store in cache
        self.embedding_cache[cache_key] = embedding
        self._manage_cache_size()
        
        return embedding
    
    async def _generate_embedding_async(self, text: str, retries: int = 3) -> List[float]:
        """Generate embedding asynchronously with retry logic."""
        if not ASYNC_AVAILABLE:
            # Fallback to sync version
            return self._generate_embedding_sync(text, retries)
        
        for attempt in range(retries):
            try:
                start_time = time.time()
                payload = {'inputs': [text]}
                
                # Create connector with optimized settings if not exists
                if not self.connector and self.connector_config:
                    self.connector = aiohttp.TCPConnector(**self.connector_config)
                
                async with aiohttp.ClientSession(connector=self.connector) as session:
                    async with session.post(
                        self.api_url,
                        headers=self.headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=45)
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            embeddings = result.get('embeddings', [])
                            
                            if embeddings and len(embeddings) > 0:
                                self.total_requests += 1
                                self.total_time += time.time() - start_time
                                
                                logger.debug(f"Generated async embedding in {time.time() - start_time:.2f}s")
                                return embeddings[0]
                            else:
                                raise ValueError("Empty embedding response")
                        else:
                            error_text = await response.text()
                            raise Exception(f"API error {response.status}: {error_text}")
                            
            except Exception as e:
                wait_time = (2 ** attempt) * 1
                logger.warning(f"Async embedding attempt {attempt + 1} failed: {e}")
                
                if attempt < retries - 1:
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {retries} async embedding attempts failed")
                    raise
    
    def _generate_embedding_sync(self, text: str, retries: int = 3) -> List[float]:
        """Synchronous embedding generation (fallback)."""
        for attempt in range(retries):
            try:
                start_time = time.time()
                payload = {'inputs': [text]}
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=45
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embeddings = result.get('embeddings', [])
                    
                    if embeddings and len(embeddings) > 0:
                        self.total_requests += 1
                        self.total_time += time.time() - start_time
                        return embeddings[0]
                    else:
                        raise ValueError("Empty embedding response")
                else:
                    raise Exception(f"API error {response.status_code}: {response.text}")
                    
            except Exception as e:
                wait_time = (2 ** attempt) * 1
                logger.warning(f"Sync embedding attempt {attempt + 1} failed: {e}")
                
                if attempt < retries - 1:
                    time.sleep(wait_time)
                else:
                    raise
    
    def embed_text_fast(self, text: str) -> List[float]:
        """
        Fast embedding that tries async first, falls back to sync.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if ASYNC_AVAILABLE:
            try:
                # Try to run async in the current event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self.embed_text_async(text))
                        return future.result(timeout=60)
                else:
                    # If no loop is running, start one
                    return asyncio.run(self.embed_text_async(text))
            except Exception as e:
                logger.warning(f"Async embedding failed, falling back to sync: {e}")
                return self.embed_text(text)
        else:
            return self.embed_text(text)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension (4096 for Qwen3-Embedding-8B)
        """
        return 4096
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_queries = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_queries * 100) if total_queries > 0 else 0
        avg_api_time = (self.total_time / self.total_requests) if self.total_requests > 0 else 0
        
        return {
            'total_queries': total_queries,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate_percent': round(cache_hit_rate, 1),
            'total_api_requests': self.total_requests,
            'average_api_time_seconds': round(avg_api_time, 2),
            'cache_size': len(self.embedding_cache),
            'connection_pooling': 'enabled' if self.connector_config else 'disabled',
            'optimization_level': 'enhanced_v2'
        }

    def test_connection(self) -> Dict[str, Any]:
        """
        Test the embedding service connection.
        
        Returns:
            Dictionary with test results
        """
        try:
            test_text = "test embedding"
            embedding = self.embed_text(test_text)
            
            return {
                "success": True,
                "model": self.config.DEEPINFRA_EMBEDDING_MODEL,
                "embedding_dimension": len(embedding),
                "test_text": test_text,
                "first_5_values": embedding[:5],
                "performance_stats": self.get_performance_stats()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Global embedding service instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """
    Get or create a global embedding service instance.
    
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    
    if _embedding_service is None:
        try:
            _embedding_service = EmbeddingService()
        except Exception as e:
            logger.warning(f"Failed to initialize embedding service: {e}")
            raise
    
    return _embedding_service


def embed_text(text: str) -> List[float]:
    """
    Convenience function to embed a single text (fast version).
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding as list of floats
    """
    service = get_embedding_service()
    return service.embed_text_fast(text)


def embed_query(query: str) -> List[float]:
    """
    Convenience function to embed a search query (fast version).
    
    Args:
        query: Query to embed
        
    Returns:
        Query embedding as list of floats
    """
    service = get_embedding_service()
    # Use fast embedding with query preprocessing
    processed_query = service._preprocess_query(query)
    return service.embed_text_fast(processed_query)