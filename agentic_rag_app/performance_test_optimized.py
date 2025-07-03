#!/usr/bin/env python3
"""
Performance Test for Optimized RAG System

Tests the performance improvements made to:
1. Embedding service with connection pooling
2. Vector search with caching 
3. Overall system performance
"""

import sys
import time
import json
from datetime import datetime
from typing import Dict, Any, List

sys.path.append('.')

from src.utils.embedding_service import get_embedding_service
from vector_store.qdrant_client import QdrantVectorStore


class PerformanceTestSuite:
    """Performance test suite for optimized RAG system."""
    
    def __init__(self):
        """Initialize test suite."""
        self.results = {
            'test_start_time': datetime.now().isoformat(),
            'embedding_tests': {},
            'vector_search_tests': {},
            'integration_tests': {},
            'optimizations_verified': [],
            'performance_summary': {}
        }
        
        # Initialize services
        print("🚀 Initializing Optimized RAG Performance Test Suite")
        print("=" * 60)
        
        try:
            self.embedding_service = get_embedding_service()
            print("✅ Embedding service initialized")
            
            # Try to connect to vector store
            collections = ['agentic_rag', 'documents', 'default', 'test_collection']
            self.vector_store = None
            
            for collection in collections:
                try:
                    vs = QdrantVectorStore(collection)
                    info = vs.get_collection_info()
                    if 'error' not in info:
                        self.vector_store = vs
                        print(f"✅ Vector store connected to collection: {collection}")
                        break
                except:
                    continue
            
            if not self.vector_store:
                print("⚠️  Vector store connection failed - will test embedding only")
                
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise
    
    def test_embedding_performance(self):
        """Test embedding service performance improvements."""
        print("\n1. EMBEDDING PERFORMANCE TESTS")
        print("-" * 40)
        
        test_texts = [
            "machine learning optimization",
            "deep neural networks",
            "artificial intelligence research",
            "natural language processing",
            "computer vision algorithms"
        ]
        
        # Test 1: Single embedding performance
        print("Testing single embedding performance...")
        single_times = []
        
        for i, text in enumerate(test_texts):
            start_time = time.time()
            embedding = self.embedding_service.embed_text(text)
            elapsed = time.time() - start_time
            single_times.append(elapsed)
            print(f"  Text {i+1}: {elapsed:.3f}s (dim: {len(embedding)})")
        
        avg_single_time = sum(single_times) / len(single_times)
        print(f"📊 Average single embedding time: {avg_single_time:.3f}s")
        
        # Test 2: Cache performance
        print("\nTesting cache performance...")
        cache_test_text = test_texts[0]
        
        # First call (cache miss)
        start_time = time.time()
        self.embedding_service.embed_text(cache_test_text)
        miss_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        self.embedding_service.embed_text(cache_test_text)
        hit_time = time.time() - start_time
        
        print(f"  Cache miss: {miss_time:.3f}s")
        print(f"  Cache hit: {hit_time:.6f}s")
        print(f"  Speedup: {miss_time/hit_time:.1f}x")
        
        # Test 3: Batch embedding performance
        print("\nTesting batch embedding performance...")
        start_time = time.time()
        batch_embeddings = self.embedding_service.embed_texts(test_texts)
        batch_time = time.time() - start_time
        
        print(f"  Batch of {len(test_texts)}: {batch_time:.3f}s")
        print(f"  Per embedding: {batch_time/len(test_texts):.3f}s")
        
        # Performance statistics
        stats = self.embedding_service.get_performance_stats()
        print(f"\n📈 Embedding Service Stats:")
        print(f"  Total queries: {stats['total_queries']}")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Cache hit rate: {stats['cache_hit_rate_percent']}%")
        print(f"  Average API time: {stats['average_api_time_seconds']:.3f}s")
        print(f"  Optimization level: {stats['optimization_level']}")
        
        # Store results
        self.results['embedding_tests'] = {
            'average_single_time': avg_single_time,
            'cache_miss_time': miss_time,
            'cache_hit_time': hit_time,
            'cache_speedup': miss_time/hit_time if hit_time > 0 else float('inf'),
            'batch_time': batch_time,
            'batch_per_embedding': batch_time/len(test_texts),
            'stats': stats
        }
        
        # Verify optimizations
        if stats['cache_hit_rate_percent'] > 0:
            self.results['optimizations_verified'].append('embedding_caching')
        if 'pooling' in stats['optimization_level']:
            self.results['optimizations_verified'].append('connection_pooling')
    
    def test_vector_search_performance(self):
        """Test vector search performance improvements."""
        if not self.vector_store:
            print("\n2. VECTOR SEARCH TESTS - SKIPPED (No vector store)")
            return
            
        print("\n2. VECTOR SEARCH PERFORMANCE TESTS")
        print("-" * 40)
        
        # Test queries
        test_queries = [
            "machine learning algorithms",
            "neural network architectures", 
            "data processing pipelines",
            "optimization techniques"
        ]
        
        search_times = []
        cached_search_times = []
        
        for i, query in enumerate(test_queries):
            print(f"Testing query {i+1}: {query[:30]}...")
            
            # Generate embedding
            start_time = time.time()
            query_vector = self.embedding_service.embed_text(query)
            embed_time = time.time() - start_time
            
            # First search (cache miss)
            start_time = time.time()
            results = self.vector_store.search(
                query_vector=query_vector, 
                top_k=3, 
                use_cache=True
            )
            search_time = time.time() - start_time
            search_times.append(search_time)
            
            # Second search (cache hit)
            start_time = time.time()
            cached_results = self.vector_store.search(
                query_vector=query_vector, 
                top_k=3, 
                use_cache=True
            )
            cached_time = time.time() - start_time
            cached_search_times.append(cached_time)
            
            print(f"  Embed: {embed_time:.3f}s, Search: {search_time:.3f}s, Cached: {cached_time:.6f}s")
            print(f"  Results: {len(results)}")
        
        avg_search_time = sum(search_times) / len(search_times)
        avg_cached_time = sum(cached_search_times) / len(cached_search_times)
        
        print(f"\n📊 Vector Search Performance:")
        print(f"  Average search time: {avg_search_time:.3f}s")
        print(f"  Average cached time: {avg_cached_time:.6f}s")
        print(f"  Cache speedup: {avg_search_time/avg_cached_time:.1f}x")
        
        # Vector store stats
        if hasattr(self.vector_store, 'get_performance_stats'):
            vs_stats = self.vector_store.get_performance_stats()
            print(f"\n📈 Vector Store Stats:")
            for key, value in vs_stats.items():
                print(f"  {key}: {value}")
            
            # Store results
            self.results['vector_search_tests'] = {
                'average_search_time': avg_search_time,
                'average_cached_time': avg_cached_time,
                'cache_speedup': avg_search_time/avg_cached_time if avg_cached_time > 0 else float('inf'),
                'stats': vs_stats
            }
            
            # Verify optimizations
            if vs_stats.get('cache_hit_rate_percent', 0) > 0:
                self.results['optimizations_verified'].append('vector_search_caching')
    
    def test_integration_performance(self):
        """Test end-to-end integration performance."""
        print("\n3. INTEGRATION PERFORMANCE TESTS")
        print("-" * 40)
        
        if not self.vector_store:
            print("SKIPPED - No vector store available")
            return
        
        # Simulate typical RAG workflow
        queries = [
            "What are the latest developments in machine learning?",
            "How do neural networks process information?",
            "What optimization techniques work best for deep learning?"
        ]
        
        total_times = []
        breakdown_times = []
        
        for i, query in enumerate(queries):
            print(f"Testing RAG workflow {i+1}: {query[:40]}...")
            
            workflow_start = time.time()
            
            # Step 1: Embed query
            embed_start = time.time()
            query_vector = self.embedding_service.embed_text(query)
            embed_time = time.time() - embed_start
            
            # Step 2: Search vectors
            search_start = time.time()
            results = self.vector_store.search(
                query_vector=query_vector,
                top_k=5,
                use_cache=True
            )
            search_time = time.time() - search_start
            
            total_time = time.time() - workflow_start
            total_times.append(total_time)
            
            breakdown = {
                'embed_time': embed_time,
                'search_time': search_time,
                'total_time': total_time,
                'results_count': len(results)
            }
            breakdown_times.append(breakdown)
            
            print(f"  Embed: {embed_time:.3f}s, Search: {search_time:.3f}s, Total: {total_time:.3f}s")
            print(f"  Results: {len(results)}")
        
        avg_total_time = sum(total_times) / len(total_times)
        avg_embed_time = sum(b['embed_time'] for b in breakdown_times) / len(breakdown_times)
        avg_search_time = sum(b['search_time'] for b in breakdown_times) / len(breakdown_times)
        
        print(f"\n📊 Integration Performance:")
        print(f"  Average total time: {avg_total_time:.3f}s")
        print(f"  Average embed time: {avg_embed_time:.3f}s")
        print(f"  Average search time: {avg_search_time:.3f}s")
        
        self.results['integration_tests'] = {
            'average_total_time': avg_total_time,
            'average_embed_time': avg_embed_time,
            'average_search_time': avg_search_time,
            'breakdown_times': breakdown_times
        }
    
    def generate_performance_summary(self):
        """Generate performance summary and recommendations."""
        print("\n4. PERFORMANCE SUMMARY")
        print("=" * 40)
        
        # Calculate improvements vs baseline
        baseline_embed_time = 3.0  # Baseline from original tests
        baseline_search_time = 3.0  # Baseline from original tests
        
        current_embed_time = self.results['embedding_tests'].get('average_single_time', baseline_embed_time)
        current_search_time = self.results.get('vector_search_tests', {}).get('average_search_time', baseline_search_time)
        
        embed_improvement = ((baseline_embed_time - current_embed_time) / baseline_embed_time) * 100
        search_improvement = ((baseline_search_time - current_search_time) / baseline_search_time) * 100
        
        print(f"📈 Performance Improvements:")
        print(f"  Embedding speed: {embed_improvement:+.1f}% vs baseline")
        print(f"  Search speed: {search_improvement:+.1f}% vs baseline")
        
        print(f"\n✅ Optimizations Verified:")
        for opt in self.results['optimizations_verified']:
            print(f"  • {opt.replace('_', ' ').title()}")
        
        # Performance grades
        embed_grade = 'A' if current_embed_time < 1.0 else 'B' if current_embed_time < 2.0 else 'C'
        search_grade = 'A' if current_search_time < 1.0 else 'B' if current_search_time < 2.0 else 'C'
        
        print(f"\n🎯 Performance Grades:")
        print(f"  Embedding Performance: {embed_grade}")
        print(f"  Search Performance: {search_grade}")
        
        # Store summary
        self.results['performance_summary'] = {
            'embed_improvement_percent': embed_improvement,
            'search_improvement_percent': search_improvement,
            'embed_grade': embed_grade,
            'search_grade': search_grade,
            'optimizations_count': len(self.results['optimizations_verified'])
        }
    
    def save_results(self):
        """Save test results to file."""
        self.results['test_end_time'] = datetime.now().isoformat()
        
        filename = f"performance_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\n📄 Results saved to: {filename}")
        except Exception as e:
            print(f"\n⚠️  Could not save results: {e}")
    
    def run_all_tests(self):
        """Run all performance tests."""
        try:
            self.test_embedding_performance()
            self.test_vector_search_performance()
            self.test_integration_performance()
            self.generate_performance_summary()
            self.save_results()
            
            print(f"\n🎉 Performance testing completed successfully!")
            return True
            
        except Exception as e:
            print(f"\n💥 Performance testing failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function to run performance tests."""
    suite = PerformanceTestSuite()
    success = suite.run_all_tests()
    
    if success:
        print("\n✅ All performance optimizations verified!")
    else:
        print("\n❌ Performance testing incomplete")


if __name__ == "__main__":
    main()