#!/usr/bin/env python3
"""
Test Citation Comparison

This script shows the difference between what the enhanced RAG tool returns
and what the manager agent system returns.
"""

import sys
sys.path.append('.')

from src.core.enhanced_rag_tool import EnhancedRAGTool
from vector_store.qdrant_client import QdrantVectorStore
from src.utils.config import Config
from src.core.manager_agent_system import ManagerAgentSystem
from src.core.debug_agent_wrapper import create_debug_manager


def test_citations_comparison():
    """Compare enhanced RAG tool output vs manager agent output."""
    print("🧪 Testing Citations: Enhanced RAG Tool vs Manager Agent")
    print("=" * 70)
    
    try:
        # Create vector store and enhanced RAG tool
        config = Config()
        vector_store = QdrantVectorStore(
            collection_name=config.QDRANT_COLLECTION_NAME, 
            config=config
        )
        
        enhanced_rag = EnhancedRAGTool(vector_store=vector_store)
        
        # Test query
        test_query = "what is the donnan effect"
        
        print(f"🔍 Query: {test_query}")
        print("=" * 70)
        
        # 1. Test Enhanced RAG Tool directly
        print("\n1️⃣ ENHANCED RAG TOOL OUTPUT (What it should show):")
        print("-" * 60)
        rag_result = enhanced_rag.forward(test_query)
        print(rag_result)
        
        print("\n" + "=" * 70)
        
        # 2. Test Manager Agent System
        print("\n2️⃣ MANAGER AGENT SYSTEM OUTPUT (What you currently see):")
        print("-" * 60)
        manager_system = ManagerAgentSystem()
        debug_manager = create_debug_manager(manager_system)
        manager_result = debug_manager.run_query(test_query)
        print(manager_result)
        
        print("\n" + "=" * 70)
        print("\n📊 ANALYSIS:")
        print("- Enhanced RAG Tool: Provides detailed document sources, file paths, content sections")
        print("- Manager Agent: Synthesizes its own answer, losing detailed citations")
        print("- Issue: Manager agent doesn't preserve the tool's detailed output format")
        
        print("\n💡 SOLUTION:")
        print("- Use the enhanced RAG tool directly for RAG queries")
        print("- Or create a bypass mode in the manager for RAG queries")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_citations_comparison()