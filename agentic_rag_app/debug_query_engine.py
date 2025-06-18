#!/usr/bin/env python3
"""Debug the query engine to see why UFDF search fails"""

import os
from dotenv import load_dotenv
from vector_store.qdrant_client import get_qdrant_store
from models.factory import get_model_factory

load_dotenv()

def debug_query_engine():
    # Initialize components
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_store = get_qdrant_store(qdrant_url)
    model_factory = get_model_factory()
    
    # Test direct vector search first
    print("=== TESTING DIRECT VECTOR SEARCH ===")
    search_results = qdrant_store.search("ufdf", top_k=5)
    print(f"Direct search results for 'ufdf': {len(search_results)} found")
    
    for i, result in enumerate(search_results, 1):
        print(f"\nResult {i}:")
        print(f"  Score: {result.get('score', 'N/A')}")
        print(f"  Content: {result.get('content', 'N/A')[:200]}...")
        metadata = result.get('metadata', {})
        print(f"  Source: {metadata.get('file_name', 'N/A')}")
    
    # Test query engine
    print("\n=== TESTING QUERY ENGINE ===")
    try:
        embedding_model = model_factory.get_embedding_model("qwen3-embed")
        chat_model = model_factory.get_chat_model("qwen3-30b-a3b")
        
        query_engine = qdrant_store.get_query_engine(
            similarity_top_k=5,
            llm=chat_model,
            embed_model=embedding_model,
            response_mode="tree_summarize",
        )
        
        print("Testing query engine with 'What is ufdf?'...")
        response = query_engine.query("What is ufdf?")
        print(f"Query engine response: {str(response)}")
        
        # Test with different queries
        test_queries = [
            "ufdf",
            "UFDF", 
            "unified file definition format",
            "ufdf overview"
        ]
        
        for query in test_queries:
            print(f"\n--- Testing: '{query}' ---")
            try:
                response = query_engine.query(query)
                print(f"Response: {str(response)[:300]}...")
            except Exception as e:
                print(f"Error: {e}")
        
    except Exception as e:
        print(f"Query engine setup error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_query_engine()