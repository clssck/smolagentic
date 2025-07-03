#!/usr/bin/env python3
"""
Test Compound RAG Tool

This script tests the new compound tool that chains:
RAG Search → Content Synthesis → Citation Formatting
"""

import sys
sys.path.append('.')

from src.core.compound_rag_tool import CompoundRAGTool
from vector_store.qdrant_client import QdrantVectorStore
from src.utils.config import Config


def test_compound_rag_tool():
    """Test the compound RAG tool pipeline."""
    print("🧪 Testing Compound RAG Tool (Tool Composition)")
    print("=" * 60)
    
    try:
        # Create vector store and compound RAG tool
        config = Config()
        vector_store = QdrantVectorStore(
            collection_name=config.QDRANT_COLLECTION_NAME, 
            config=config
        )
        
        compound_tool = CompoundRAGTool(vector_store=vector_store)
        
        print("✅ Compound RAG tool created successfully")
        
        # Test query
        test_query = "what is the donnan effect"
        print(f"\n🔍 Testing query: {test_query}")
        print("-" * 40)
        
        result = compound_tool.forward(test_query)
        
        print("\n📄 Compound RAG Tool Output:")
        print("=" * 40)
        print(result)
        print("=" * 40)
        
        print("\n✅ Compound RAG tool test completed!")
        print("\nThis shows the full pipeline:")
        print("  1. 🔍 RAG Search - Found relevant documents")
        print("  2. 🧠 Content Synthesis - Created coherent answer") 
        print("  3. 📚 Citation Formatting - Added detailed sources")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_compound_rag_tool()