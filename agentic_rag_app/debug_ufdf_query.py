#!/usr/bin/env python3
"""Debug script to query vector storage for UFDF references."""

import logging
import os
from dotenv import load_dotenv
from vector_store.qdrant_client import get_qdrant_store
from models.factory import get_model_factory

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_ufdf():
    """Query the vector store for UFDF references with different search strategies."""
    
    # Get Qdrant URL from environment
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    print(f"Using Qdrant URL: {qdrant_url}")
    
    # Initialize components
    qdrant_store = get_qdrant_store(qdrant_url)
    model_factory = get_model_factory()
    
    # Get collection info
    print("=== QDRANT COLLECTION INFO ===")
    collection_info = qdrant_store.get_collection_info()
    print(f"Collection info: {collection_info}")
    print()
    
    # Test queries for UFDF (simplified for faster testing)
    test_queries = ["ufdf", "UFDF"]
    
    print("=== VECTOR SEARCH RESULTS ===")
    for query in test_queries:
        print(f"\n--- Searching for: '{query}' ---")
        try:
            results = qdrant_store.search(query, top_k=2)
            if results:
                for i, result in enumerate(results, 1):
                    print(f"Result {i}:")
                    print(f"  Score: {result.get('score', 'N/A')}")
                    print(f"  Content: {result.get('content', 'N/A')[:150]}...")
                    metadata = result.get('metadata', {})
                    print(f"  Source: {metadata.get('source', 'N/A')}")
                    print()
            else:
                print("  No results found")
        except Exception as e:
            print(f"  Error searching: {e}")
            break
    
    # Test with query engine
    print("\n=== QUERY ENGINE TEST ===")
    try:
        embedding_model = model_factory.get_embedding_model("qwen3-embed")
        chat_model = model_factory.get_chat_model("qwen3-32b")
        
        query_engine = qdrant_store.get_query_engine(
            similarity_top_k=5,
            llm=chat_model,
            embed_model=embedding_model,
            response_mode="tree_summarize",
        )
        
        print("Testing query engine with 'ufdf'...")
        response = query_engine.query("What is ufdf?")
        print(f"Query engine response: {str(response)}")
        
    except Exception as e:
        print(f"Query engine error: {e}")
    
    # Test raw collection access
    print("\n=== RAW COLLECTION ACCESS ===")
    try:
        client = qdrant_store.client
        collection_name = qdrant_store.collection_name
        
        # Get total count
        collection_info = client.get_collection(collection_name)
        print(f"Total points in collection: {collection_info.points_count}")
        
        # Scroll through some documents to check content
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=10,
            with_payload=True
        )
        
        print("\nSample documents:")
        for i, point in enumerate(points, 1):
            payload = point.payload or {}
            content = payload.get('text', '')[:100]
            print(f"Doc {i}: {content}...")
            
            # Check if any contain 'ufdf' case-insensitive
            if 'ufdf' in content.lower():
                print(f"  *** FOUND UFDF REFERENCE ***")
                print(f"  Full content: {content}")
        
    except Exception as e:
        print(f"Raw collection access error: {e}")

if __name__ == "__main__":
    query_ufdf()