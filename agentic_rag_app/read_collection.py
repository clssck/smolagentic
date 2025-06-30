#!/usr/bin/env python3
"""
Script to read and display contents of the Qdrant collection
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from vector_store.qdrant_client import QdrantVectorStore
from utils.config import Config

def main():
    try:
        # Initialize configuration
        config = Config()
        
        # Use agentic_rag collection by default
        collection_name = os.getenv("QDRANT_COLLECTION_NAME", "agentic_rag")
        
        print(f"🔍 Reading Qdrant collection: {collection_name}")
        print(f"📍 URL: {config.QDRANT_URL}")
        print("-" * 50)
        
        # Initialize vector store
        vector_store = QdrantVectorStore(collection_name, config)
        
        # Get collection info
        print("📊 Collection Information:")
        info = vector_store.get_collection_info()
        if info:
            print(f"   Status: {info.get('status', 'Unknown')}")
            print(f"   Vector Count: {info.get('vectors_count', 0)}")
            print(f"   Indexed Vectors: {info.get('indexed_vectors_count', 0)}")
            print(f"   Points Count: {info.get('points_count', 0)}")
        
        # Count points
        point_count = vector_store.count_points()
        print(f"   Total Points: {point_count}")
        
        if point_count > 0:
            print("\n📝 Sample Data (first 5 points):")
            sample_data = vector_store.get_sample_data(limit=5)
            
            for i, point in enumerate(sample_data, 1):
                print(f"\n   Point {i}:")
                
                # Handle both dict and object formats
                if hasattr(point, 'id'):
                    point_id = point.id
                    payload = point.payload
                elif isinstance(point, dict):
                    point_id = point.get('id')
                    payload = point.get('payload', {})
                else:
                    point_id = str(point)
                    payload = {}
                
                print(f"     ID: {point_id}")
                if payload:
                    for key, value in payload.items():
                        # Truncate long text values
                        if isinstance(value, str) and len(value) > 100:
                            value = value[:100] + "..."
                        print(f"     {key}: {value}")
        else:
            print("\n   No points found in the collection.")
            
    except Exception as e:
        print(f"❌ Error reading collection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()