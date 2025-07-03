#!/usr/bin/env python3
"""
Explore and examine the existing vector storage data
"""

import sys
import os
import json
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import Config
from vector_store.qdrant_client import QdrantVectorStore

def explore_vector_storage():
    """Explore the existing vector storage to understand its structure."""
    print("🔍 Exploring Vector Storage...")
    print("="*60)
    
    # Initialize
    config = Config()
    vector_store = QdrantVectorStore(
        collection_name=config.QDRANT_COLLECTION_NAME,
        config=config
    )
    
    # 1. Collection information
    print("\n📊 Collection Information:")
    collection_info = vector_store.get_collection_info()
    print(json.dumps(collection_info, indent=2))
    
    # 2. Point count
    print(f"\n📈 Total Points: {vector_store.count_points()}")
    
    # 3. Sample some actual data points
    print("\n🔬 Sample Data Points:")
    try:
        # Get some points to examine structure
        sample_points = vector_store.get_points(limit=3)
        
        for i, point in enumerate(sample_points):
            print(f"\n--- Point {i+1} ---")
            print(f"Type: {type(point)}")
            
            # If it's a dict, examine its structure
            if isinstance(point, dict):
                print("Keys:", list(point.keys()))
                
                # Show some content
                if 'id' in point:
                    print(f"ID: {point['id']}")
                if 'content' in point:
                    content_preview = point['content'][:200] + "..." if len(str(point['content'])) > 200 else point['content']
                    print(f"Content: {content_preview}")
                if 'text' in point:
                    text_preview = point['text'][:200] + "..." if len(str(point['text'])) > 200 else point['text']
                    print(f"Text: {text_preview}")
                if 'metadata' in point:
                    print(f"Metadata: {point['metadata']}")
                if 'payload' in point:
                    print(f"Payload keys: {list(point['payload'].keys()) if isinstance(point['payload'], dict) else 'Not a dict'}")
            else:
                # If it's an object, try to inspect its attributes
                print(f"Attributes: {dir(point)}")
                if hasattr(point, 'id'):
                    print(f"ID: {point.id}")
                if hasattr(point, 'payload'):
                    print(f"Payload: {point.payload}")
                if hasattr(point, 'vector'):
                    print(f"Vector dim: {len(point.vector) if point.vector else 'None'}")
    
    except Exception as e:
        print(f"Error getting sample points: {e}")
    
    # 4. Try a basic search
    print("\n🔍 Test Search:")
    try:
        # Try searching with an empty query vector (just to see what happens)
        search_results = vector_store.search(query_text="artificial intelligence", top_k=2)
        print(f"Search returned {len(search_results)} results")
        
        for i, result in enumerate(search_results):
            print(f"\n--- Search Result {i+1} ---")
            print(f"Type: {type(result)}")
            
            if isinstance(result, dict):
                print("Keys:", list(result.keys()))
                if 'content' in result:
                    content_preview = result['content'][:100] + "..." if len(str(result['content'])) > 100 else result['content']
                    print(f"Content: {content_preview}")
            else:
                if hasattr(result, 'id'):
                    print(f"ID: {result.id}")
                if hasattr(result, 'score'):
                    print(f"Score: {result.score}")
                if hasattr(result, 'payload'):
                    payload = result.payload
                    if isinstance(payload, dict):
                        content_preview = payload.get('content', payload.get('text', ''))[:100]
                        print(f"Content: {content_preview}...")
    
    except Exception as e:
        print(f"Error in test search: {e}")
    
    # 5. List all collections
    print("\n📚 Available Collections:")
    try:
        collections = vector_store.list_collections()
        print(f"Collections: {collections}")
    except Exception as e:
        print(f"Error listing collections: {e}")
    
    print("\n" + "="*60)
    print("✅ Vector storage exploration completed!")

if __name__ == "__main__":
    explore_vector_storage()