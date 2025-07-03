#!/usr/bin/env python3
"""
Quick Vector Database Test
"""

import sys
import time
import traceback

# Add src to path
sys.path.append('src')

try:
    from utils.config import Config
    from vector_store.qdrant_client import QdrantVectorStore
    from utils.embedding_service import embed_query
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def test_vector_db():
    """Quick vector database test"""
    print("🚀 Quick Vector Database Test")
    print("=" * 40)
    
    results = {}
    
    try:
        # 1. Setup
        print("\n1. 🔌 Testing connection...")
        config = Config()
        vector_store = QdrantVectorStore(
            collection_name=config.QDRANT_COLLECTION_NAME,
            config=config
        )
        
        # Check collection info
        info = vector_store.get_collection_info()
        print(f"   ✅ Collection info: {info}")
        results["connection"] = "✅ PASS"
        
        # 2. Basic search test
        print("\n2. 🔍 Testing basic search...")
        test_queries = [
            "Python programming",
            "machine learning",
            "neural networks",
            "database optimization",
            "web development"
        ]
        
        search_times = []
        successful_searches = 0
        
        for query in test_queries:
            try:
                start = time.time()
                
                # Test semantic search
                embedding = embed_query(query)
                semantic_results = vector_store.search(query_vector=embedding, top_k=3)
                
                # Test text search
                text_results = vector_store.search_by_text_filter(query, limit=3)
                
                search_time = time.time() - start
                search_times.append(search_time)
                
                has_results = (semantic_results and len(semantic_results) > 0) or (text_results and len(text_results) > 0)
                if has_results:
                    successful_searches += 1
                
                print(f"   📊 '{query}': {len(semantic_results) if semantic_results else 0} semantic, {len(text_results) if text_results else 0} text ({search_time:.3f}s)")
                
            except Exception as e:
                print(f"   ❌ Query failed: {query} - {e}")
        
        avg_time = sum(search_times) / len(search_times) if search_times else 0
        success_rate = successful_searches / len(test_queries)
        
        print(f"   📈 Average search time: {avg_time:.3f}s")
        print(f"   📊 Success rate: {success_rate:.1%}")
        
        if success_rate >= 0.8 and avg_time < 2.0:
            results["search"] = "✅ PASS"
        else:
            results["search"] = "⚠️ MARGINAL"
        
        # 3. Performance test
        print("\n3. ⚡ Testing performance...")
        performance_queries = [f"test query {i}" for i in range(20)]
        
        start_time = time.time()
        perf_times = []
        
        for query in performance_queries:
            try:
                start = time.time()
                embedding = embed_query(query)
                vector_store.search(query_vector=embedding, top_k=5)
                perf_times.append(time.time() - start)
            except Exception as e:
                print(f"   ⚠️ Performance query failed: {e}")
        
        total_time = time.time() - start_time
        avg_perf_time = sum(perf_times) / len(perf_times) if perf_times else float('inf')
        qps = len(perf_times) / total_time if total_time > 0 else 0
        
        print(f"   📊 Processed {len(perf_times)}/20 queries in {total_time:.2f}s")
        print(f"   ⚡ Average query time: {avg_perf_time:.3f}s")
        print(f"   🚀 Queries per second: {qps:.1f}")
        
        if avg_perf_time < 1.0 and qps > 5:
            results["performance"] = "✅ PASS"
        elif avg_perf_time < 2.0 and qps > 2:
            results["performance"] = "⚠️ MARGINAL"
        else:
            results["performance"] = "❌ FAIL"
        
        # 4. Edge cases
        print("\n4. 🎭 Testing edge cases...")
        edge_queries = [
            ("", "empty"),
            ("a", "single char"),
            ("!@#$", "special chars"),
            ("x" * 100, "long query"),
            ("SELECT * FROM users", "SQL"),
        ]
        
        edge_successes = 0
        for query, desc in edge_queries:
            try:
                if query.strip():
                    embedding = embed_query(query)
                    vector_store.search(query_vector=embedding, top_k=2)
                vector_store.search_by_text_filter(query, limit=2)
                edge_successes += 1
                print(f"   ✅ {desc}: handled")
            except Exception as e:
                print(f"   ⚠️ {desc}: {e}")
        
        edge_rate = edge_successes / len(edge_queries)
        print(f"   📊 Edge case success: {edge_rate:.1%}")
        
        if edge_rate >= 0.8:
            results["edge_cases"] = "✅ PASS"
        else:
            results["edge_cases"] = "⚠️ MARGINAL"
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 QUICK TEST SUMMARY")
    print("=" * 40)
    
    for test_name, result in results.items():
        print(f"{result} {test_name.upper().replace('_', ' ')}")
    
    passed = sum(1 for r in results.values() if "✅" in r)
    total = len(results)
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Vector database is BULLETPROOF! 🎉")
    elif passed >= total * 0.75:
        print("✅ Vector database is in good shape")
    else:
        print("⚠️ Vector database needs attention")
    
    return True


if __name__ == "__main__":
    test_vector_db()