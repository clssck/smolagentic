#!/usr/bin/env python3
"""
Demo script showing how to read and query Qdrant collections

This script demonstrates various ways to interact with Qdrant collections:
1. List available collections
2. Inspect collection information
3. Retrieve sample data
4. Perform text-based searches
5. Query specific points

Usage:
    python demo_qdrant_usage.py
    
Requirements:
    - QDRANT_URL environment variable
    - QDRANT_API_KEY environment variable
    - qdrant-client package installed
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from vector_store.qdrant_client import QdrantVectorStore, list_all_collections, inspect_collection
from src.utils.config import Config


def main():
    """Demonstrate Qdrant collection reading and querying."""
    print("🔍 QDRANT COLLECTION READER & QUERY DEMO")
    print("=" * 60)
    
    # Check environment
    print("\\n1. ENVIRONMENT CHECK:")
    config = Config()
    
    if not config.QDRANT_URL or not config.QDRANT_API_KEY:
        print("❌ Missing Qdrant configuration:")
        print(f"   QDRANT_URL: {'✅ Set' if config.QDRANT_URL else '❌ Missing'}")
        print(f"   QDRANT_API_KEY: {'✅ Set' if config.QDRANT_API_KEY else '❌ Missing'}")
        print("\\nPlease set QDRANT_URL and QDRANT_API_KEY environment variables.")
        return
    
    print("✅ Qdrant configuration found")
    print(f"   URL: {config.QDRANT_URL}")
    print(f"   Collection: {config.QDRANT_COLLECTION_NAME}")
    
    try:
        # List all collections
        print("\\n2. LISTING ALL COLLECTIONS:")
        collections = list_all_collections()
        
        if collections:
            print(f"✅ Found {len(collections)} collection(s):")
            for i, collection in enumerate(collections, 1):
                print(f"   {i}. {collection}")
        else:
            print("❌ No collections found or connection failed")
            return
        
        # Use the configured collection or the first available one
        target_collection = config.QDRANT_COLLECTION_NAME
        if target_collection not in collections:
            print(f"⚠️  Configured collection '{target_collection}' not found")
            target_collection = collections[0]
            print(f"   Using first available collection: {target_collection}")
        
        # Initialize vector store
        print(f"\\n3. CONNECTING TO COLLECTION: {target_collection}")
        vector_store = QdrantVectorStore(target_collection, config)
        
        # Get collection information
        print("\\n4. COLLECTION INFORMATION:")
        info = vector_store.get_collection_info()
        print(f"   Collection info: {info}")
        
        point_count = vector_store.count_points()
        print(f"   Total points: {point_count}")
        
        if point_count == 0:
            print("⚠️  Collection is empty - no data to query")
            return
        
        # Get sample data
        print("\\n5. SAMPLE DATA:")
        sample_data = vector_store.get_sample_data(limit=3)
        
        if sample_data:
            print(f"✅ Retrieved {len(sample_data)} sample points:")
            for i, point in enumerate(sample_data, 1):
                print(f"\\n   Point {i}:")
                print(f"     ID: {point['id']}")
                print(f"     Content preview: {point['content'][:100]}...")
                if point['metadata']:
                    print(f"     Metadata: {point['metadata']}")
        else:
            print("❌ No sample data retrieved")
        
        # Text-based search (if collection has content)
        if sample_data and any(point['content'] for point in sample_data):
            print("\\n6. TEXT-BASED SEARCH:")
            
            # Extract some keywords from sample data for search
            search_terms = ["machine", "learning", "AI", "data", "model", "system"]
            
            for term in search_terms[:2]:  # Test first 2 terms
                print(f"\\n   Searching for '{term}':")
                try:
                    results = vector_store.search_by_text_filter(term, limit=3)
                    if results:
                        print(f"   ✅ Found {len(results)} results:")
                        for j, result in enumerate(results, 1):
                            print(f"      {j}. ID: {result['id']}")
                            print(f"         Content: {result['content'][:80]}...")
                    else:
                        print(f"   📭 No results found for '{term}'")
                except Exception as e:
                    print(f"   ❌ Search failed: {e}")
        
        # Demonstrate pagination
        print("\\n7. PAGINATION DEMO:")
        try:
            page_size = 5
            for page in range(2):  # Show first 2 pages
                offset = page * page_size
                points = vector_store.get_points(limit=page_size, offset=offset)
                
                if points:
                    print(f"   Page {page + 1} (offset {offset}):")
                    for point in points:
                        print(f"     - {point['id']}: {point['content'][:50]}...")
                else:
                    print(f"   Page {page + 1}: No more data")
                    break
        except Exception as e:
            print(f"   ❌ Pagination failed: {e}")
        
        # Show how to query specific IDs
        if sample_data:
            print("\\n8. QUERYING SPECIFIC POINTS:")
            # Take first 2 IDs from sample data
            test_ids = [point['id'] for point in sample_data[:2]]
            
            try:
                specific_points = vector_store.get_points(ids=test_ids)
                if specific_points:
                    print(f"   ✅ Retrieved {len(specific_points)} specific points:")
                    for point in specific_points:
                        print(f"     ID: {point['id']}")
                        print(f"     Content: {point['content'][:100]}...")
                else:
                    print("   📭 No points found for specified IDs")
            except Exception as e:
                print(f"   ❌ Specific point query failed: {e}")
        
        # Summary
        print("\\n9. SUMMARY:")
        print(f"✅ Successfully connected to Qdrant")
        print(f"✅ Found {len(collections)} collection(s)")
        print(f"✅ Queried collection '{target_collection}' with {point_count} points")
        print(f"✅ Demonstrated various query methods")
        
        print("\\n🎉 Demo completed successfully!")
        print("\\n📚 USAGE EXAMPLES:")
        print("   # List collections")
        print("   collections = list_all_collections()")
        print()
        print("   # Create vector store")
        print("   vs = QdrantVectorStore('my_collection', config)")
        print()
        print("   # Get sample data")
        print("   sample = vs.get_sample_data(limit=10)")
        print()
        print("   # Search by text")
        print("   results = vs.search_by_text_filter('machine learning')")
        print()
        print("   # Get specific points")
        print("   points = vs.get_points(ids=['id1', 'id2'])")
        print()
        print("   # Vector search (requires embedding)")
        print("   # results = vs.search(query_vector=[0.1, 0.2, ...], top_k=5)")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()