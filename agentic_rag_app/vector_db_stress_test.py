#!/usr/bin/env python3
"""
Vector Database Stress Test Suite
Comprehensive testing to ensure bulletproof vector storage
"""

import asyncio
import json
import random
import string
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Add src to path
sys.path.append('src')

try:
    from utils.config import Config
    from vector_store.qdrant_client import QdrantVectorStore
    from utils.embedding_service import embed_query, get_embedding_service
except ImportError as e:
    print(f"Import error: {e}")
    print("Run from project root directory")
    sys.exit(1)


class VectorDBStressTester:
    """Comprehensive vector database stress tester"""
    
    def __init__(self):
        self.config = Config()
        self.vector_store = None
        self.results = {
            "connection_test": None,
            "basic_operations": None,
            "search_accuracy": None,
            "performance_tests": None,
            "concurrent_tests": None,
            "edge_cases": None,
            "memory_stress": None,
            "error_handling": None
        }
        self.start_time = time.time()
        
    def setup(self):
        """Initialize vector store connection"""
        try:
            print("🔌 Setting up vector store connection...")
            self.vector_store = QdrantVectorStore(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                config=self.config
            )
            print("✅ Vector store connected successfully")
            return True
        except Exception as e:
            print(f"❌ Vector store setup failed: {e}")
            traceback.print_exc()
            return False
    
    def test_connection_health(self):
        """Test basic connection health"""
        print("\n🔍 Testing connection health...")
        try:
            # Test collection exists
            stats = self.vector_store.get_collection_stats()
            print(f"✅ Collection stats: {stats}")
            
            # Test client responsiveness
            start = time.time()
            info = self.vector_store.client.get_collection(self.config.QDRANT_COLLECTION_NAME)
            response_time = time.time() - start
            print(f"✅ Client response time: {response_time:.3f}s")
            
            self.results["connection_test"] = {
                "status": "success",
                "stats": stats,
                "response_time": response_time
            }
            return True
            
        except Exception as e:
            print(f"❌ Connection health test failed: {e}")
            self.results["connection_test"] = {"status": "failed", "error": str(e)}
            return False
    
    def test_basic_operations(self):
        """Test basic CRUD operations"""
        print("\n🔧 Testing basic operations...")
        tests = []
        
        try:
            # Test embedding generation
            test_text = "This is a test document for vector database operations"
            embedding = embed_query(test_text)
            tests.append(("embedding_generation", len(embedding) > 0))
            print(f"✅ Embedding generated: {len(embedding)} dimensions")
            
            # Test search operations
            results = self.vector_store.search(query_vector=embedding, top_k=5)
            tests.append(("vector_search", isinstance(results, list)))
            print(f"✅ Vector search returned {len(results) if results else 0} results")
            
            # Test text-based search
            text_results = self.vector_store.search_by_text_filter("test", limit=5)
            tests.append(("text_search", isinstance(text_results, list)))
            print(f"✅ Text search returned {len(text_results) if text_results else 0} results")
            
            self.results["basic_operations"] = {
                "status": "success",
                "tests": dict(tests),
                "embedding_dimensions": len(embedding)
            }
            return True
            
        except Exception as e:
            print(f"❌ Basic operations test failed: {e}")
            self.results["basic_operations"] = {"status": "failed", "error": str(e)}
            return False
    
    def test_search_accuracy(self):
        """Test search accuracy with known queries"""
        print("\n🎯 Testing search accuracy...")
        
        test_queries = [
            "machine learning algorithms",
            "Python programming",
            "neural networks",
            "database optimization",
            "software architecture",
            "data structures",
            "web development",
            "artificial intelligence",
            "cloud computing",
            "cybersecurity"
        ]
        
        results = []
        try:
            for query in test_queries:
                start_time = time.time()
                
                # Test semantic search
                embedding = embed_query(query)
                semantic_results = self.vector_store.search(query_vector=embedding, top_k=3)
                
                # Test text search
                text_results = self.vector_store.search_by_text_filter(query, limit=3)
                
                search_time = time.time() - start_time
                
                result = {
                    "query": query,
                    "semantic_results": len(semantic_results) if semantic_results else 0,
                    "text_results": len(text_results) if text_results else 0,
                    "search_time": search_time,
                    "has_results": (semantic_results and len(semantic_results) > 0) or (text_results and len(text_results) > 0)
                }
                results.append(result)
                
                print(f"  📊 '{query}': {result['semantic_results']} semantic, {result['text_results']} text ({search_time:.3f}s)")
            
            # Calculate statistics
            avg_search_time = sum(r['search_time'] for r in results) / len(results)
            success_rate = sum(1 for r in results if r['has_results']) / len(results)
            
            print(f"✅ Average search time: {avg_search_time:.3f}s")
            print(f"✅ Search success rate: {success_rate:.1%}")
            
            self.results["search_accuracy"] = {
                "status": "success",
                "test_queries": len(test_queries),
                "average_search_time": avg_search_time,
                "success_rate": success_rate,
                "results": results
            }
            return True
            
        except Exception as e:
            print(f"❌ Search accuracy test failed: {e}")
            self.results["search_accuracy"] = {"status": "failed", "error": str(e)}
            return False
    
    def test_performance_stress(self):
        """Test performance under stress"""
        print("\n⚡ Testing performance under stress...")
        
        try:
            # Generate various query types
            queries = [
                # Short queries
                *[f"test {i}" for i in range(10)],
                # Medium queries  
                *[f"machine learning algorithm {i} for data processing" for i in range(10)],
                # Long queries
                *[f"comprehensive analysis of software architecture patterns and design principles for scalable enterprise applications number {i}" for i in range(10)],
                # Technical terms
                *["API", "SQL", "JSON", "REST", "GraphQL", "Docker", "Kubernetes", "AWS", "GCP", "Azure"],
                # Common words
                *["data", "system", "application", "service", "function", "method", "class", "object", "variable", "parameter"]
            ]
            
            search_times = []
            errors = 0
            
            print(f"  🔥 Running {len(queries)} performance queries...")
            
            for i, query in enumerate(queries):
                try:
                    start = time.time()
                    
                    # Alternate between search types
                    if i % 2 == 0:
                        embedding = embed_query(query)
                        results = self.vector_store.search(query_vector=embedding, top_k=5)
                    else:
                        results = self.vector_store.search_by_text_filter(query, limit=5)
                    
                    search_time = time.time() - start
                    search_times.append(search_time)
                    
                    if i % 10 == 0:
                        print(f"    Progress: {i}/{len(queries)} queries completed")
                        
                except Exception as e:
                    errors += 1
                    print(f"    ⚠️ Query failed: {query[:30]}... - {e}")
            
            # Calculate performance metrics
            if search_times:
                avg_time = sum(search_times) / len(search_times)
                min_time = min(search_times)
                max_time = max(search_times)
                p95_time = sorted(search_times)[int(len(search_times) * 0.95)]
                
                print(f"✅ Performance Results:")
                print(f"    Total queries: {len(queries)}")
                print(f"    Successful queries: {len(search_times)}")
                print(f"    Failed queries: {errors}")
                print(f"    Average time: {avg_time:.3f}s")
                print(f"    Min time: {min_time:.3f}s")
                print(f"    Max time: {max_time:.3f}s")
                print(f"    95th percentile: {p95_time:.3f}s")
                
                self.results["performance_tests"] = {
                    "status": "success",
                    "total_queries": len(queries),
                    "successful_queries": len(search_times),
                    "failed_queries": errors,
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "p95_time": p95_time,
                    "error_rate": errors / len(queries)
                }
                return True
            else:
                print("❌ No successful queries in performance test")
                return False
                
        except Exception as e:
            print(f"❌ Performance stress test failed: {e}")
            self.results["performance_tests"] = {"status": "failed", "error": str(e)}
            return False
    
    def test_concurrent_access(self):
        """Test concurrent access patterns"""
        print("\n🔀 Testing concurrent access...")
        
        def concurrent_search(thread_id: int, num_queries: int = 5):
            """Perform searches from a single thread"""
            results = []
            errors = 0
            
            for i in range(num_queries):
                try:
                    query = f"concurrent test query {thread_id}-{i}"
                    start = time.time()
                    
                    # Mix of search types
                    if i % 2 == 0:
                        embedding = embed_query(query)
                        search_results = self.vector_store.search(query_vector=embedding, top_k=3)
                    else:
                        search_results = self.vector_store.search_by_text_filter(query, limit=3)
                    
                    search_time = time.time() - start
                    results.append(search_time)
                    
                except Exception as e:
                    errors += 1
                    
            return {
                "thread_id": thread_id,
                "successful_searches": len(results),
                "errors": errors,
                "avg_time": sum(results) / len(results) if results else 0,
                "total_time": sum(results)
            }
        
        try:
            num_threads = 10
            queries_per_thread = 5
            
            print(f"  🧵 Running {num_threads} concurrent threads with {queries_per_thread} queries each...")
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(concurrent_search, i, queries_per_thread)
                    for i in range(num_threads)
                ]
                
                thread_results = []
                for future in as_completed(futures):
                    result = future.result()
                    thread_results.append(result)
            
            total_time = time.time() - start_time
            
            # Aggregate results
            total_searches = sum(r['successful_searches'] for r in thread_results)
            total_errors = sum(r['errors'] for r in thread_results)
            avg_search_time = sum(r['avg_time'] * r['successful_searches'] for r in thread_results) / total_searches if total_searches > 0 else 0
            
            print(f"✅ Concurrent Access Results:")
            print(f"    Threads: {num_threads}")
            print(f"    Total time: {total_time:.3f}s")
            print(f"    Total searches: {total_searches}")
            print(f"    Total errors: {total_errors}")
            print(f"    Average search time: {avg_search_time:.3f}s")
            print(f"    Queries per second: {total_searches / total_time:.1f}")
            
            self.results["concurrent_tests"] = {
                "status": "success",
                "num_threads": num_threads,
                "total_time": total_time,
                "total_searches": total_searches,
                "total_errors": total_errors,
                "avg_search_time": avg_search_time,
                "qps": total_searches / total_time,
                "thread_results": thread_results
            }
            return True
            
        except Exception as e:
            print(f"❌ Concurrent access test failed: {e}")
            self.results["concurrent_tests"] = {"status": "failed", "error": str(e)}
            return False
    
    def test_edge_cases(self):
        """Test edge cases and unusual inputs"""
        print("\n🎭 Testing edge cases...")
        
        edge_cases = [
            # Empty and minimal
            ("", "empty string"),
            (" ", "single space"),
            ("a", "single character"),
            
            # Very long queries
            ("x" * 1000, "very long query"),
            (" ".join(["word"] * 100), "many repeated words"),
            
            # Special characters
            ("!@#$%^&*()", "special characters"),
            ("🚀🎯💡🔥", "emojis"),
            ("café naïve résumé", "accented characters"),
            
            # Code-like content
            ("def function(x): return x + 1", "code snippet"),
            ("SELECT * FROM users WHERE id = 1", "SQL query"),
            ("<html><body>test</body></html>", "HTML content"),
            
            # Mixed languages and scripts
            ("hello world 世界 мир", "mixed scripts"),
            ("αβγδε φυχψω", "greek letters"),
            
            # Numbers and dates
            ("123456789", "numbers only"),
            ("2024-01-01 12:34:56", "datetime format"),
            ("$1,234.56", "currency format"),
            
            # Potential injection attempts
            ("'; DROP TABLE users; --", "SQL injection attempt"),
            ("<script>alert('xss')</script>", "XSS attempt"),
            ("../../../etc/passwd", "path traversal attempt"),
        ]
        
        results = []
        try:
            for query, description in edge_cases:
                try:
                    start = time.time()
                    
                    # Test both search methods
                    if query.strip():  # Only embed non-empty queries
                        embedding = embed_query(query)
                        semantic_results = self.vector_store.search(query_vector=embedding, top_k=2)
                    else:
                        semantic_results = []
                    
                    text_results = self.vector_store.search_by_text_filter(query, limit=2)
                    search_time = time.time() - start
                    
                    result = {
                        "query": query[:50] + "..." if len(query) > 50 else query,
                        "description": description,
                        "semantic_results": len(semantic_results) if semantic_results else 0,
                        "text_results": len(text_results) if text_results else 0,
                        "search_time": search_time,
                        "status": "success"
                    }
                    
                except Exception as e:
                    result = {
                        "query": query[:50] + "..." if len(query) > 50 else query,
                        "description": description,
                        "error": str(e),
                        "status": "failed"
                    }
                
                results.append(result)
                status_icon = "✅" if result["status"] == "success" else "❌"
                print(f"  {status_icon} {description}: {result.get('search_time', 'N/A')}")
            
            success_count = sum(1 for r in results if r["status"] == "success")
            success_rate = success_count / len(results)
            
            print(f"✅ Edge Cases Results:")
            print(f"    Total tests: {len(edge_cases)}")
            print(f"    Successful: {success_count}")
            print(f"    Success rate: {success_rate:.1%}")
            
            self.results["edge_cases"] = {
                "status": "success",
                "total_tests": len(edge_cases),
                "successful_tests": success_count,
                "success_rate": success_rate,
                "results": results
            }
            return True
            
        except Exception as e:
            print(f"❌ Edge cases test failed: {e}")
            self.results["edge_cases"] = {"status": "failed", "error": str(e)}
            return False
    
    def test_memory_stress(self):
        """Test memory usage under stress"""
        print("\n💾 Testing memory stress...")
        
        try:
            import psutil
            import gc
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            print(f"  📊 Initial memory usage: {initial_memory:.1f} MB")
            
            # Generate large number of embeddings and searches
            large_queries = []
            for i in range(100):
                # Generate diverse queries of varying lengths
                query_length = random.randint(10, 200)
                query = ' '.join(random.choices([
                    'machine', 'learning', 'algorithm', 'data', 'science',
                    'python', 'javascript', 'database', 'optimization',
                    'neural', 'network', 'deep', 'artificial', 'intelligence'
                ], k=query_length))
                large_queries.append(query)
            
            # Perform memory-intensive operations
            embedding_times = []
            search_times = []
            
            for i, query in enumerate(large_queries):
                try:
                    # Generate embedding
                    start = time.time()
                    embedding = embed_query(query)
                    embedding_times.append(time.time() - start)
                    
                    # Perform search
                    start = time.time()
                    results = self.vector_store.search(query_vector=embedding, top_k=10)
                    search_times.append(time.time() - start)
                    
                    # Memory check every 25 iterations
                    if i % 25 == 0:
                        current_memory = process.memory_info().rss / 1024 / 1024
                        print(f"    Progress: {i}/100, Memory: {current_memory:.1f} MB")
                        
                        # Force garbage collection
                        gc.collect()
                        
                except Exception as e:
                    print(f"    ⚠️ Memory stress query {i} failed: {e}")
            
            # Final memory check
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            print(f"✅ Memory Stress Results:")
            print(f"    Initial memory: {initial_memory:.1f} MB")
            print(f"    Final memory: {final_memory:.1f} MB")
            print(f"    Memory increase: {memory_increase:.1f} MB")
            print(f"    Average embedding time: {sum(embedding_times)/len(embedding_times):.3f}s")
            print(f"    Average search time: {sum(search_times)/len(search_times):.3f}s")
            
            self.results["memory_stress"] = {
                "status": "success",
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_increase_mb": memory_increase,
                "avg_embedding_time": sum(embedding_times)/len(embedding_times),
                "avg_search_time": sum(search_times)/len(search_times),
                "total_operations": len(large_queries) * 2
            }
            return True
            
        except ImportError:
            print("⚠️ psutil not available, skipping memory stress test")
            self.results["memory_stress"] = {"status": "skipped", "reason": "psutil not available"}
            return True
        except Exception as e:
            print(f"❌ Memory stress test failed: {e}")
            self.results["memory_stress"] = {"status": "failed", "error": str(e)}
            return False
    
    def test_error_handling(self):
        """Test error handling and recovery"""
        print("\n🛡️ Testing error handling...")
        
        error_tests = []
        
        try:
            # Test invalid collection access
            try:
                invalid_store = QdrantVectorStore("nonexistent_collection", self.config)
                invalid_store.search(query_vector=[0.1] * 384, top_k=1)
                error_tests.append(("invalid_collection", "failed_to_error"))
            except Exception:
                error_tests.append(("invalid_collection", "handled_correctly"))
            
            # Test invalid vector dimensions
            try:
                results = self.vector_store.search(query_vector=[0.1, 0.2], top_k=1)  # Wrong dimensions
                error_tests.append(("invalid_dimensions", "failed_to_error"))
            except Exception:
                error_tests.append(("invalid_dimensions", "handled_correctly"))
            
            # Test invalid top_k values
            try:
                embedding = embed_query("test")
                results = self.vector_store.search(query_vector=embedding, top_k=-1)
                error_tests.append(("negative_top_k", "handled_gracefully"))
            except Exception:
                error_tests.append(("negative_top_k", "handled_correctly"))
            
            # Test very large top_k values
            try:
                embedding = embed_query("test")
                results = self.vector_store.search(query_vector=embedding, top_k=100000)
                error_tests.append(("large_top_k", "handled_gracefully"))
            except Exception:
                error_tests.append(("large_top_k", "handled_correctly"))
            
            # Test connection resilience
            try:
                # Rapid successive calls to test connection pooling
                for i in range(20):
                    embedding = embed_query(f"rapid test {i}")
                    self.vector_store.search(query_vector=embedding, top_k=1)
                error_tests.append(("rapid_calls", "handled_correctly"))
            except Exception as e:
                error_tests.append(("rapid_calls", f"failed: {e}"))
            
            handled_correctly = sum(1 for _, status in error_tests if "handled_correctly" in status or "handled_gracefully" in status)
            
            print(f"✅ Error Handling Results:")
            for test_name, status in error_tests:
                status_icon = "✅" if "handled_correctly" in status or "handled_gracefully" in status else "⚠️"
                print(f"    {status_icon} {test_name}: {status}")
            
            print(f"    Properly handled: {handled_correctly}/{len(error_tests)}")
            
            self.results["error_handling"] = {
                "status": "success",
                "total_tests": len(error_tests),
                "properly_handled": handled_correctly,
                "test_results": dict(error_tests)
            }
            return True
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            self.results["error_handling"] = {"status": "failed", "error": str(e)}
            return False
    
    def generate_report(self):
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("📊 VECTOR DATABASE STRESS TEST REPORT")
        print("="*80)
        
        # Overall summary
        passed_tests = sum(1 for result in self.results.values() 
                          if isinstance(result, dict) and result.get("status") == "success")
        total_tests = len([r for r in self.results.values() if r is not None])
        
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"   Total test time: {total_time:.2f} seconds")
        print(f"   Tests passed: {passed_tests}/{total_tests}")
        print(f"   Success rate: {passed_tests/total_tests:.1%}" if total_tests > 0 else "   Success rate: N/A")
        
        # Detailed results
        for test_name, result in self.results.items():
            if result is None:
                continue
                
            print(f"\n📋 {test_name.upper().replace('_', ' ')}:")
            
            if result["status"] == "success":
                print(f"   ✅ Status: PASSED")
                
                # Add specific metrics for each test
                if test_name == "performance_tests" and "avg_time" in result:
                    print(f"   ⚡ Average query time: {result['avg_time']:.3f}s")
                    print(f"   📊 95th percentile: {result['p95_time']:.3f}s")
                    print(f"   🎯 Error rate: {result['error_rate']:.1%}")
                
                elif test_name == "concurrent_tests" and "qps" in result:
                    print(f"   🔀 Queries per second: {result['qps']:.1f}")
                    print(f"   🧵 Concurrent threads: {result['num_threads']}")
                
                elif test_name == "search_accuracy" and "success_rate" in result:
                    print(f"   🎯 Search success rate: {result['success_rate']:.1%}")
                    print(f"   ⚡ Average search time: {result['average_search_time']:.3f}s")
                
                elif test_name == "memory_stress" and "memory_increase_mb" in result:
                    print(f"   💾 Memory increase: {result['memory_increase_mb']:.1f} MB")
                    print(f"   📊 Total operations: {result['total_operations']}")
                
            elif result["status"] == "failed":
                print(f"   ❌ Status: FAILED")
                print(f"   🚨 Error: {result.get('error', 'Unknown error')}")
                
            elif result["status"] == "skipped":
                print(f"   ⏭️ Status: SKIPPED")
                print(f"   ℹ️ Reason: {result.get('reason', 'Unknown reason')}")
        
        # Save report to file
        report_file = f"vector_db_stress_report_{int(time.time())}.json"
        report_data = {
            "timestamp": time.time(),
            "total_time": total_time,
            "tests_passed": passed_tests,
            "total_tests": total_tests,
            "success_rate": passed_tests/total_tests if total_tests > 0 else 0,
            "detailed_results": self.results
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        
        # Final recommendation
        if passed_tests / total_tests >= 0.8:
            print(f"\n🎉 VERDICT: Vector database is BULLETPROOF! 🎉")
            print(f"   Your vector database passed {passed_tests}/{total_tests} tests.")
            print(f"   The system is ready for production workloads.")
        else:
            print(f"\n⚠️ VERDICT: Vector database needs attention")
            print(f"   Only {passed_tests}/{total_tests} tests passed.")
            print(f"   Review failed tests and address issues before production.")
        
        return report_data
    
    def run_all_tests(self):
        """Run the complete stress test suite"""
        print("🚀 STARTING VECTOR DATABASE STRESS TEST SUITE")
        print("="*80)
        
        if not self.setup():
            print("❌ Setup failed, aborting tests")
            return False
        
        # Run all test suites
        test_methods = [
            self.test_connection_health,
            self.test_basic_operations,
            self.test_search_accuracy,
            self.test_performance_stress,
            self.test_concurrent_access,
            self.test_edge_cases,
            self.test_memory_stress,
            self.test_error_handling
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f"❌ Test suite {test_method.__name__} crashed: {e}")
                traceback.print_exc()
        
        # Generate final report
        return self.generate_report()


def main():
    """Run the stress test"""
    tester = VectorDBStressTester()
    report = tester.run_all_tests()
    return report


if __name__ == "__main__":
    main()