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
    from qdrant_client.http.models import PointStruct, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Qdrant Vector Store for RAG system.
    
    Provides methods to connect to Qdrant, store vectors, and perform similarity search.
    """
    
    def __init__(self, collection_name: str, config: Any = None):
        """
        Initialize Qdrant Vector Store.
        
        Args:
            collection_name: Name of the Qdrant collection
            config: Configuration object with Qdrant settings
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is not installed. Install it with: pip install qdrant-client")
        
        self.collection_name = collection_name
        self.config = config
        self.client = None
        
        # Initialize client
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize Qdrant client with configuration."""
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
            
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                timeout=60
            )
            
            logger.info(f"Connected to Qdrant at {url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.config.params.vectors.size if hasattr(collection_info.config.params, 'vectors') else 'unknown',
                "vectors_count": collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else 0,
                "points_count": collection_info.points_count if hasattr(collection_info, 'points_count') else 0,
                "status": collection_info.status.value if hasattr(collection_info, 'status') else 'unknown'
            }
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
               score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            query_vector: Vector to search for
            query_text: Text query (if using text-based search)
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with content and scores
        """
        try:
            if query_vector is None and query_text is not None:
                # If only text is provided, you would need to embed it first
                # For now, we'll raise an error
                raise ValueError("Text-only search requires an embedding model. Please provide query_vector.")
            
            if query_vector is None:
                raise ValueError("Either query_vector or query_text must be provided")
            
            # Perform vector search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results
            results = []
            for point in search_result:
                result = {
                    "id": point.id,
                    "score": point.score,
                    "content": point.payload.get("content", ""),
                    "text": point.payload.get("text", point.payload.get("content", "")),
                    "metadata": point.payload.get("metadata", {}),
                    "payload": point.payload
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} results for vector search")
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