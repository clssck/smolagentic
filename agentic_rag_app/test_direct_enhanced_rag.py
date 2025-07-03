#!/usr/bin/env python3
"""
Test Enhanced RAG Tool Directly

This script tests the enhanced RAG tool directly to show the improved citations.
"""

import sys
sys.path.append('.')

from src.core.enhanced_rag_tool import EnhancedRAGTool
from vector_store.qdrant_client import QdrantVectorStore
from src.utils.config import Config


def test_enhanced_rag_directly():
    """Test the enhanced RAG tool directly."""
    print("🧪 Testing Enhanced RAG Tool Directly")
    print("=" * 60)
    
    try:
        # Create vector store and enhanced RAG tool
        config = Config()
        vector_store = QdrantVectorStore(
            collection_name=config.QDRANT_COLLECTION_NAME, 
            config=config
        )
        
        enhanced_rag = EnhancedRAGTool(vector_store=vector_store)
        
        print("✅ Enhanced RAG tool created successfully")
        
        # Test query
        test_query = "what is the donnan effect"
        print(f"\n🔍 Testing query: {test_query}")
        print("-" * 40)
        
        result = enhanced_rag.forward(test_query)
        
        print("\n📄 Enhanced RAG Tool Output:")
        print("=" * 40)
        print(result)
        print("=" * 40)
        
        print("\n✅ Enhanced RAG tool test completed!")
        print("\nThis shows the detailed document sources and citations that should be preserved.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_enhanced_rag_directly()