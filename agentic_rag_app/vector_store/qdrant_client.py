"""
Qdrant Vector Store Implementation for RAG System

This module provides a QdrantVectorStore class that interfaces with Qdrant
for vector storage and similarity search operations.
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import PointStruct, VectorParams, SearchParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Optimized Qdrant Vector Store for RAG system.
    
    Provides high-performance methods for vector storage and similarity search
    with connection pooling and caching optimizations.
    """
    
    def __init__(self, collection_name: str, config: Any = None):
        """
        Initialize Qdrant Vector Store with performance optimizations.
        
        Args:
            collection_name: Name of the Qdrant collection
            config: Configuration object with Qdrant settings
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is not installed. Install it with: pip install qdrant-client")
        
        self.collection_name = collection_name
        self.config = config
        self.client = None
        
        # Performance optimizations
        self._search_cache = {}  # Cache for recent searches
        self._cache_size = 100   # Maximum cached searches
        self._collection_info_cache = None
        self._collection_info_cache_time = 0
        self._cache_ttl = 300  # 5 minutes cache TTL
        
        # Performance metrics
        self.search_count = 0
        self.cache_hits = 0
        self.total_search_time = 0.0
        
        # Initialize client with optimizations
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize Qdrant client with performance optimizations."""
        try:
            if self.config:
                url = getattr(self.config, 'QDRANT_URL', None)
                api_key = getattr(self.config, 'QDRANT_API_KEY', None)
            else:
                import os
                url = os.getenv('QDRANT_URL')
                api_key = os.getenv('QDRANT_API_KEY')
            
            if not url:
                raise ValueError("QDRANT_URL not configured")
            
            # Initialize with optimized settings for performance
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                timeout=30,          # Reduced timeout for faster failures
                prefer_grpc=True,    # Use gRPC for better performance if available
                grpc_port=6334       # Standard gRPC port
            )
            
            logger.info(f"Connected to optimized Qdrant at {url}")
            logger.info(f"Connection pooling and caching enabled")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            # Fallback to basic HTTP client
            try:
                self.client = QdrantClient(
                    url=url,
                    api_key=api_key,
                    timeout=30
                )
                logger.info("Connected using HTTP fallback")
            except Exception as fallback_error:
                logger.error(f"Fallback connection also failed: {fallback_error}")
                raise
    
    def get_collection_info(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get information about the collection with caching.
        
        Args:
            use_cache: Whether to use cached collection info
            
        Returns:
            Dictionary with collection information
        """
        import time
        
        current_time = time.time()
        
        # Return cached info if available and not expired
        if (use_cache and self._collection_info_cache and 
            current_time - self._collection_info_cache_time < self._cache_ttl):
            return self._collection_info_cache
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            result = {
                "name": collection_info.config.params.vectors.size if hasattr(collection_info.config.params, 'vectors') else 'unknown',
                "vectors_count": collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else 0,
                "points_count": collection_info.points_count if hasattr(collection_info, 'points_count') else 0,
                "status": collection_info.status.value if hasattr(collection_info, 'status') else 'unknown'
            }
            
            # Cache the result
            if use_cache:
                self._collection_info_cache = result
                self._collection_info_cache_time = current_time
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}
    
    def list_collections(self) -> List[str]:
        """
        List all available collections.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.client.get_collections()
            return [collection.name for collection in collections.collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def search(self, query_vector: List[float] = None, query_text: str = None, top_k: int = 5, 
               score_threshold: float = 0.0, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        High-performance search for similar vectors with caching.
        
        Args:
            query_vector: Vector to search for
            query_text: Text query (if using text-based search)
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            use_cache: Whether to use search result caching
            
        Returns:
            List of search results with content and scores
        """
        import time
        import hashlib
        
        start_time = time.time()
        
        try:
            if query_vector is None and query_text is not None:
                raise ValueError("Text-only search requires an embedding model. Please provide query_vector.")
            
            if query_vector is None:
                raise ValueError("Either query_vector or query_text must be provided")
            
            # Generate cache key for this search
            cache_key = None
            if use_cache:
                vector_str = ','.join(f'{x:.6f}' for x in query_vector[:10])  # Use first 10 elements for key
                cache_key = hashlib.md5(f"{vector_str}_{top_k}_{score_threshold}".encode()).hexdigest()
                
                if cache_key in self._search_cache:
                    self.cache_hits += 1
                    logger.debug(f"Cache hit for search (key: {cache_key[:8]})")
                    return self._search_cache[cache_key]
            
            # Perform optimized vector search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
                search_params=models.SearchParams(
                    hnsw_ef=128,        # Increased for better accuracy
                    exact=False         # Use approximate search for speed
                )
            )
            
            # Optimized result formatting
            results = []
            for point in search_result:
                # Extract common fields efficiently
                payload = point.payload
                content = payload.get("content", "")
                
                result = {
                    "id": point.id,
                    "score": point.score,
                    "content": content,
                    "text": payload.get("text", content),
                    "metadata": payload.get("metadata", {}),
                    "payload": payload
                }
                results.append(result)
            
            # Cache results if enabled
            if use_cache and cache_key:
                self._manage_search_cache()
                self._search_cache[cache_key] = results
            
            # Update performance metrics
            elapsed = time.time() - start_time
            self.search_count += 1
            self.total_search_time += elapsed
            
            logger.debug(f"Vector search completed in {elapsed:.3f}s - found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_points(self, ids: List[str] = None, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve points from the collection.
        
        Args:
            ids: Specific point IDs to retrieve
            limit: Maximum number of points to return
            offset: Number of points to skip
            
        Returns:
            List of points with their data
        """
        try:
            if ids:
                # Get specific points by IDs
                points = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=ids,
                    with_payload=True,
                    with_vectors=False
                )
            else:
                # Get points with pagination
                points, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
            
            # Format results
            results = []
            for point in points:
                result = {
                    "id": point.id,
                    "content": point.payload.get("content", ""),
                    "text": point.payload.get("text", point.payload.get("content", "")),
                    "metadata": point.payload.get("metadata", {}),
                    "payload": point.payload
                }
                results.append(result)
            
            logger.info(f"Retrieved {len(results)} points from collection")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve points: {e}")
            return []
    
    def count_points(self) -> int:
        """
        Count total number of points in the collection.
        
        Returns:
            Number of points in the collection
        """
        try:
            count_info = self.client.count(collection_name=self.collection_name)
            return count_info.count
        except Exception as e:
            logger.error(f"Failed to count points: {e}")
            return 0
    
    def search_by_text_filter(self, text_query: str, field: str = "content", 
                             limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for points containing specific text (using payload filtering).
        
        Args:
            text_query: Text to search for
            field: Field name to search in
            limit: Maximum results to return
            
        Returns:
            List of matching points
        """
        try:
            # Use scroll with filter for text search
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key=field,
                        match=models.MatchText(text=text_query)
                    )
                ]
            )
            
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results
            results = []
            for point in points:
                result = {
                    "id": point.id,
                    "content": point.payload.get("content", ""),
                    "text": point.payload.get("text", point.payload.get("content", "")),
                    "metadata": point.payload.get("metadata", {}),
                    "payload": point.payload
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} points matching text query")
            return results
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
    
    def get_sample_data(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get sample data from the collection for inspection.
        
        Args:
            limit: Number of sample points to retrieve
            
        Returns:
            List of sample points
        """
        return self.get_points(limit=limit)
    
    def _manage_search_cache(self):
        """Manage search cache size by removing oldest entries."""
        if len(self._search_cache) >= self._cache_size:
            # Remove oldest 20% of entries
            remove_count = int(self._cache_size * 0.2)
            keys_to_remove = list(self._search_cache.keys())[:remove_count]
            for key in keys_to_remove:
                del self._search_cache[key]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this vector store."""
        avg_search_time = (self.total_search_time / self.search_count) if self.search_count > 0 else 0
        cache_hit_rate = (self.cache_hits / self.search_count * 100) if self.search_count > 0 else 0
        
        return {
            'total_searches': self.search_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate_percent': round(cache_hit_rate, 1),
            'average_search_time_seconds': round(avg_search_time, 3),
            'cached_searches': len(self._search_cache),
            'cache_size_limit': self._cache_size,
            'optimization_level': 'enhanced_v2'
        }
    
    def clear_cache(self):
        """Clear all cached data."""
        self._search_cache.clear()
        self._collection_info_cache = None
        self._collection_info_cache_time = 0
        logger.info("Vector store caches cleared")


# Convenience functions for direct usage
def create_qdrant_client(url: str = None, api_key: str = None) -> QdrantClient:
    """
    Create a Qdrant client with the given parameters.
    
    Args:
        url: Qdrant server URL
        api_key: API key for authentication
        
    Returns:
        QdrantClient instance
    """
    if not QDRANT_AVAILABLE:
        raise ImportError("qdrant-client is not installed")
    
    if not url:
        import os
        url = os.getenv('QDRANT_URL')
        api_key = api_key or os.getenv('QDRANT_API_KEY')
    
    return QdrantClient(url=url, api_key=api_key)


def list_all_collections(url: str = None, api_key: str = None) -> List[str]:
    """
    List all collections in a Qdrant instance.
    
    Args:
        url: Qdrant server URL
        api_key: API key for authentication
        
    Returns:
        List of collection names
    """
    client = create_qdrant_client(url, api_key)
    collections = client.get_collections()
    return [collection.name for collection in collections.collections]


def inspect_collection(collection_name: str, url: str = None, api_key: str = None) -> Dict[str, Any]:
    """
    Inspect a collection and return summary information.
    
    Args:
        collection_name: Name of collection to inspect
        url: Qdrant server URL
        api_key: API key for authentication
        
    Returns:
        Dictionary with collection information and sample data
    """
    vector_store = QdrantVectorStore(collection_name)
    
    info = vector_store.get_collection_info()
    sample_data = vector_store.get_sample_data(limit=3)
    point_count = vector_store.count_points()
    
    return {
        "collection_info": info,
        "point_count": point_count,
        "sample_data": sample_data,
        "collections_available": vector_store.list_collections()
    }