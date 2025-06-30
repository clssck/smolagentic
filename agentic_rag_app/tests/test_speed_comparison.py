#!/usr/bin/env python3
"""Speed comparison test for optimized vs original system."""

import os

# Add project root to path
import sys
import time

sys.path.append(str(os.path.dirname(os.path.abspath(__file__))))

from src.adapters.model_adapters import OpenRouterModelAdapter
from src.agents.optimized_rag_agent import OptimizedRAGAgent
from src.agents.rag_agent import RAGAgent
from src.core.interfaces import ComponentConfig
from src.core.optimized_retriever import OptimizedQdrantRetrieverTool
from src.core.retrievers import QdrantRetrieverTool


def create_model():
    """Create optimized model."""
    config = ComponentConfig(
        name="speed_test_model",
        config={
            "model_id": "meta-llama/llama-3.1-70b-instruct",
            "api_key_env": "OPENROUTER_API_KEY",
            "max_tokens": 800,  # Reduced for speed
            "temperature": 0.3,
        },
    )
    return OpenRouterModelAdapter(config)


def test_retriever_speed():
    """Test retriever speed comparison."""
    print("🔍 RETRIEVER SPEED TEST")
    print("=" * 40)

    queries = [
        "Claude AI assistant capabilities",
        "Agentic RAG system architecture",
        "vector database optimization",
        "document processing pipeline",
    ]

    # Test original retriever
    print("Original Retriever:")
    original = QdrantRetrieverTool()
    original_times = []

    for query in queries:
        start = time.time()
        result = original.forward(query, top_k=3, score_threshold=0.5)
        elapsed = time.time() - start
        original_times.append(elapsed)
        found = "No relevant documents" not in result
        print(f"  {elapsed:.3f}s - {query[:30]}... ({'✅' if found else '❌'})")

    original_avg = sum(original_times) / len(original_times)

    # Test optimized retriever
    print("\nOptimized Retriever:")
    optimized = OptimizedQdrantRetrieverTool()
    optimized_times = []

    for query in queries:
        start = time.time()
        result = optimized.forward(query, top_k=3)
        elapsed = time.time() - start
        optimized_times.append(elapsed)
        found = "No relevant documents" not in result and "Search error" not in result
        print(f"  {elapsed:.3f}s - {query[:30]}... ({'✅' if found else '❌'})")

    optimized_avg = sum(optimized_times) / len(optimized_times)

    # Results
    speedup = original_avg / optimized_avg if optimized_avg > 0 else 1
    print("\n📊 Retriever Results:")
    print(f"   Original: {original_avg:.3f}s average")
    print(f"   Optimized: {optimized_avg:.3f}s average")
    print(f"   Speedup: {speedup:.1f}x")

    return original_avg, optimized_avg


def test_agent_speed():
    """Test full agent speed comparison."""
    print("\n🤖 AGENT SPEED TEST")
    print("=" * 40)

    try:
        model = create_model()

        # Test queries
        test_queries = ["What is Claude AI?", "How does the Agentic RAG system work?"]

        # Test original agent
        print("Original RAG Agent:")
        original_retriever = QdrantRetrieverTool()
        original_agent = RAGAgent(model=model, tools=[original_retriever])
        original_times = []

        for query in test_queries:
            print(f"  Testing: {query}")
            start = time.time()
            try:
                result = original_agent.run(query, max_steps=2)
                elapsed = time.time() - start
                original_times.append(elapsed)
                print(f"    ✅ {elapsed:.2f}s - Response: {str(result)[:50]}...")
            except Exception as e:
                elapsed = time.time() - start
                original_times.append(elapsed)
                print(f"    ❌ {elapsed:.2f}s - Error: {str(e)[:50]}...")

        original_avg = (
            sum(original_times) / len(original_times) if original_times else 0
        )

        # Test optimized agent
        print("\nOptimized RAG Agent:")
        optimized_retriever = OptimizedQdrantRetrieverTool()
        optimized_agent = OptimizedRAGAgent(model=model, tools=[optimized_retriever])
        optimized_times = []

        for query in test_queries:
            print(f"  Testing: {query}")
            start = time.time()
            try:
                result = optimized_agent.run(query, max_steps=2)
                elapsed = time.time() - start
                optimized_times.append(elapsed)
                print(f"    ✅ {elapsed:.2f}s - Response: {str(result)[:50]}...")
            except Exception as e:
                elapsed = time.time() - start
                optimized_times.append(elapsed)
                print(f"    ❌ {elapsed:.2f}s - Error: {str(e)[:50]}...")

        optimized_avg = (
            sum(optimized_times) / len(optimized_times) if optimized_times else 0
        )

        # Results
        if original_avg > 0 and optimized_avg > 0:
            speedup = original_avg / optimized_avg
            print("\n📊 Agent Results:")
            print(f"   Original: {original_avg:.2f}s average")
            print(f"   Optimized: {optimized_avg:.2f}s average")
            print(f"   Speedup: {speedup:.1f}x")

            # Performance stats
            if hasattr(optimized_agent, "get_performance_stats"):
                stats = optimized_agent.get_performance_stats()
                print(f"   Optimized agent stats: {stats}")

        return original_avg, optimized_avg

    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return 0, 0


def main():
    """Run speed comparison tests."""
    print("⚡ AGENTIC RAG SPEED OPTIMIZATION TEST")
    print("=" * 50)

    # Environment check
    required_keys = ["OPENROUTER_API_KEY", "QDRANT_API_KEY", "DEEPINFRA_API_KEY"]
    missing = [key for key in required_keys if not os.getenv(key)]

    if missing:
        print(f"❌ Missing API keys: {missing}")
        return

    print("✅ All API keys available")
    print()

    # Test retrievers
    retriever_orig, retriever_optimized = test_retriever_speed()

    # Test agents
    agent_orig, agent_optimized = test_agent_speed()

    # Summary
    print("\n🎯 FINAL SUMMARY")
    print("=" * 50)

    if retriever_orig > 0 and retriever_optimized > 0:
        ret_speedup = retriever_orig / retriever_optimized
        print(
            f"🔍 Retriever: {ret_speedup:.1f}x faster ({retriever_orig:.3f}s → {retriever_optimized:.3f}s)"
        )

    if agent_orig > 0 and agent_optimized > 0:
        agent_speedup = agent_orig / agent_optimized
        print(
            f"🤖 Full Agent: {agent_speedup:.1f}x faster ({agent_orig:.2f}s → {agent_optimized:.2f}s)"
        )

    print("\n🚀 Performance Optimizations Applied:")
    print("   • Optimized HNSW search parameters (128→64→32)")
    print("   • Performance caching (2000 entries, 2hr TTL)")
    print("   • Optimized timeouts and connection pooling")
    print("   • Reduced max_tokens (4000→1500→800)")
    print("   • Lower temperature for consistent responses")
    print("   • Optimized formatting and truncation")


if __name__ == "__main__":
    main()
