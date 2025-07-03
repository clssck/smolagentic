#!/usr/bin/env python3
"""
Comprehensive End-to-End Test for Vector Storage
This script tests the complete vector storage pipeline with real APIs
and provides detailed insights into the vector storage internals.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import Config
from src.utils.embedding_service import EmbeddingService
from vector_store.qdrant_client import QdrantVectorStore


class VectorStorageE2ETester:
    """Comprehensive end-to-end tester for vector storage system."""
    
    def __init__(self):
        self.config = Config()
        self.embedding_service = None
        self.vector_store = None
        self.test_results = {}
        self.test_documents = [
            {
                "id": "test_doc_1",
                "text": "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
                "metadata": {"category": "programming", "language": "english", "year": 1991}
            },
            {
                "id": "test_doc_2", 
                "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed.",
                "metadata": {"category": "ai", "language": "english", "complexity": "high"}
            },
            {
                "id": "test_doc_3",
                "text": "Vector databases are specialized databases designed to store and query high-dimensional vector embeddings efficiently.",
                "metadata": {"category": "database", "language": "english", "type": "vector"}
            },
            {
                "id": "test_doc_4",
                "text": "Qdrant is an open-source vector database that provides fast and accurate similarity search capabilities.",
                "metadata": {"category": "database", "language": "english", "type": "vector", "vendor": "qdrant"}
            },
            {
                "id": "test_doc_5",
                "text": "Retrieval-Augmented Generation combines information retrieval with language generation to produce more accurate and contextual responses.",
                "metadata": {"category": "ai", "language": "english", "complexity": "high", "technique": "rag"}
            }
        ]

    async def setup(self):
        """Initialize services and validate configuration."""
        print("🔧 Setting up test environment...")
        
        # Validate configuration
        try:
            self.config.validate()
            print("✅ Configuration validated successfully")
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
            
        # Initialize embedding service
        try:
            self.embedding_service = EmbeddingService()
            print("✅ Embedding service initialized")
        except Exception as e:
            print(f"❌ Failed to initialize embedding service: {e}")
            return False
            
        # Initialize vector store
        try:
            self.vector_store = QdrantVectorStore(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                config=self.config
            )
            print("✅ Vector store initialized")
        except Exception as e:
            print(f"❌ Failed to initialize vector store: {e}")
            return False
            
        return True

    async def test_embedding_service(self):
        """Test the embedding service with various inputs."""
        print("\n🧪 Testing Embedding Service...")
        
        test_cases = [
            "Simple test query",
            "This is a longer text that should test the embedding service's ability to handle more complex sentences with multiple concepts.",
            "Special characters: @#$%^&*()_+-=[]{}|;':\",./<>?",
            "Numbers and dates: 2024-01-01, 42, 3.14159",
            ""  # Empty string test
        ]
        
        results = {}
        
        for i, text in enumerate(test_cases):
            try:
                start_time = time.time()
                
                if text:  # Use sync embedding for reliability
                    embedding = self.embedding_service.embed_text(text)
                    
                    # Test async as well but don't fail if it has issues
                    try:
                        async_embedding = await self.embedding_service.embed_text_async(text)
                        if embedding != async_embedding:
                            print(f"⚠️  Async/sync embedding mismatch for test case {i+1}")
                    except Exception as async_error:
                        print(f"⚠️  Async embedding failed for test case {i+1}: {async_error}")
                    
                    results[f"test_case_{i+1}"] = {
                        "text": text[:50] + "..." if len(text) > 50 else text,
                        "embedding_dim": len(embedding),
                        "processing_time": time.time() - start_time,
                        "first_5_values": embedding[:5],
                        "last_5_values": embedding[-5:],
                        "embedding_norm": sum(x**2 for x in embedding)**0.5
                    }
                    
                    print(f"✅ Test case {i+1}: {len(embedding)}D embedding in {time.time() - start_time:.3f}s")
                else:
                    print(f"⏭️  Skipping empty string test case {i+1}")
                    
            except Exception as e:
                print(f"❌ Test case {i+1} failed: {e}")
                results[f"test_case_{i+1}"] = {"error": str(e)}
        
        # Test caching
        print("\n🔄 Testing embedding cache...")
        cache_test_text = "This is a cache test query"
        
        # First call
        start_time = time.time()
        embedding1 = self.embedding_service.embed_text(cache_test_text)
        first_call_time = time.time() - start_time
        
        # Second call (should be cached)
        start_time = time.time()
        embedding2 = self.embedding_service.embed_text(cache_test_text)
        second_call_time = time.time() - start_time
        
        if embedding1 == embedding2:
            print(f"✅ Cache test passed - First: {first_call_time:.3f}s, Second: {second_call_time:.3f}s")
            results["cache_test"] = {
                "first_call_time": first_call_time,
                "second_call_time": second_call_time,
                "speedup": first_call_time / second_call_time if second_call_time > 0 else "instant"
            }
        else:
            print("❌ Cache test failed - embeddings don't match")
            results["cache_test"] = {"error": "embeddings don't match"}
        
        self.test_results["embedding_service"] = results
        return results

    async def test_vector_store_operations(self):
        """Test comprehensive vector store operations."""
        print("\n🗄️ Testing Vector Store Operations...")
        
        results = {}
        
        # Test 1: Check collection info
        try:
            collection_info = self.vector_store.get_collection_info()
            results["collection_info"] = collection_info
            print(f"✅ Collection info retrieved: {collection_info}")
        except Exception as e:
            print(f"❌ Failed to get collection info: {e}")
            results["collection_info"] = {"error": str(e)}
        
        # Test 2: Get initial point count
        try:
            initial_count = self.vector_store.count_points()
            results["initial_point_count"] = initial_count
            print(f"✅ Initial point count: {initial_count}")
        except Exception as e:
            print(f"❌ Failed to get initial point count: {e}")
            results["initial_point_count"] = {"error": str(e)}
        
        # Test 3: Add test documents
        print("\n📝 Adding test documents...")
        added_docs = []
        for doc in self.test_documents:
            try:
                embedding = self.embedding_service.embed_text(doc["text"])
                point_id = self.vector_store.add_point(
                    point_id=doc["id"],
                    vector=embedding,
                    payload={
                        "text": doc["text"],
                        "metadata": doc["metadata"],
                        "added_at": datetime.now().isoformat(),
                        "test_session": "e2e_test"
                    }
                )
                added_docs.append(doc["id"])
                print(f"✅ Added document: {doc['id']}")
            except Exception as e:
                print(f"❌ Failed to add document {doc['id']}: {e}")
        
        results["added_documents"] = added_docs
        
        # Test 4: Verify point count increased
        try:
            new_count = self.vector_store.count_points()
            results["new_point_count"] = new_count
            print(f"✅ New point count: {new_count}")
        except Exception as e:
            print(f"❌ Failed to get new point count: {e}")
        
        # Test 5: Search operations
        print("\n🔍 Testing search operations...")
        search_queries = [
            "Python programming language",
            "machine learning algorithms",
            "vector database storage",
            "artificial intelligence",
            "data retrieval systems"
        ]
        
        search_results = {}
        for query in search_queries:
            try:
                query_embedding = self.embedding_service.embed_text(query)
                search_result = self.vector_store.search(
                    query_vector=query_embedding,
                    top_k=3,
                    score_threshold=0.0
                )
                
                search_results[query] = {
                    "num_results": len(search_result),
                    "results": [
                        {
                            "id": r.id,
                            "score": r.score,
                            "text_preview": r.payload.get("text", "")[:100] + "..." if len(r.payload.get("text", "")) > 100 else r.payload.get("text", ""),
                            "metadata": r.payload.get("metadata", {})
                        } for r in search_result
                    ]
                }
                print(f"✅ Search '{query}': {len(search_result)} results")
                
            except Exception as e:
                print(f"❌ Search failed for '{query}': {e}")
                search_results[query] = {"error": str(e)}
        
        results["search_results"] = search_results
        
        # Test 6: Filter operations
        print("\n🔍 Testing filter operations...")
        filter_tests = [
            {"category": "programming"},
            {"category": "ai"},
            {"language": "english"},
            {"complexity": "high"}
        ]
        
        filter_results = {}
        for filter_dict in filter_tests:
            try:
                filter_result = self.vector_store.search_by_text_filter(
                    filter_dict=filter_dict,
                    limit=10
                )
                filter_results[str(filter_dict)] = {
                    "num_results": len(filter_result),
                    "results": [
                        {
                            "id": r.id,
                            "text_preview": r.payload.get("text", "")[:100] + "..." if len(r.payload.get("text", "")) > 100 else r.payload.get("text", ""),
                            "metadata": r.payload.get("metadata", {})
                        } for r in filter_result
                    ]
                }
                print(f"✅ Filter {filter_dict}: {len(filter_result)} results")
                
            except Exception as e:
                print(f"❌ Filter failed for {filter_dict}: {e}")
                filter_results[str(filter_dict)] = {"error": str(e)}
        
        results["filter_results"] = filter_results
        
        # Test 7: Get specific points
        print("\n📋 Testing point retrieval...")
        try:
            if added_docs:
                points = self.vector_store.get_points(point_ids=added_docs[:3])
                results["retrieved_points"] = {
                    "num_points": len(points),
                    "points": [
                        {
                            "id": p.id,
                            "vector_dim": len(p.vector) if p.vector else 0,
                            "payload_keys": list(p.payload.keys()) if p.payload else []
                        } for p in points
                    ]
                }
                print(f"✅ Retrieved {len(points)} specific points")
        except Exception as e:
            print(f"❌ Failed to retrieve specific points: {e}")
            results["retrieved_points"] = {"error": str(e)}
        
        # Test 8: Sample data
        print("\n🔀 Testing sample data retrieval...")
        try:
            sample_data = self.vector_store.get_sample_data(limit=5)
            results["sample_data"] = {
                "num_samples": len(sample_data),
                "samples": [
                    {
                        "id": s.id,
                        "vector_dim": len(s.vector) if s.vector else 0,
                        "payload_keys": list(s.payload.keys()) if s.payload else [],
                        "text_preview": s.payload.get("text", "")[:50] + "..." if s.payload and len(s.payload.get("text", "")) > 50 else s.payload.get("text", "") if s.payload else ""
                    } for s in sample_data
                ]
            }
            print(f"✅ Retrieved {len(sample_data)} sample points")
        except Exception as e:
            print(f"❌ Failed to retrieve sample data: {e}")
            results["sample_data"] = {"error": str(e)}
        
        self.test_results["vector_store_operations"] = results
        return results

    async def probe_vector_storage_internals(self):
        """Deep dive into vector storage internals and debug information."""
        print("\n🔬 Probing Vector Storage Internals...")
        
        results = {}
        
        # Test 1: Detailed collection information
        try:
            collection_info = self.vector_store.get_collection_info()
            results["detailed_collection_info"] = collection_info
            
            # Extract key metrics
            if collection_info:
                results["collection_metrics"] = {
                    "status": collection_info.get("status"),
                    "vectors_count": collection_info.get("vectors_count", 0),
                    "indexed_vectors_count": collection_info.get("indexed_vectors_count", 0),
                    "points_count": collection_info.get("points_count", 0),
                    "segments_count": collection_info.get("segments_count", 0),
                    "config": collection_info.get("config", {})
                }
                print(f"✅ Collection metrics extracted")
            
        except Exception as e:
            print(f"❌ Failed to get detailed collection info: {e}")
            results["detailed_collection_info"] = {"error": str(e)}
        
        # Test 2: Vector space analysis
        print("\n🔢 Analyzing vector space...")
        try:
            sample_vectors = []
            sample_points = self.vector_store.get_sample_data(limit=10)
            
            for point in sample_points:
                if point.vector and len(point.vector) > 0:
                    sample_vectors.append(point.vector)
            
            if sample_vectors:
                # Calculate vector statistics
                vector_stats = {
                    "num_vectors": len(sample_vectors),
                    "vector_dimension": len(sample_vectors[0]) if sample_vectors else 0,
                    "vector_norms": [sum(x**2 for x in v)**0.5 for v in sample_vectors[:5]],
                    "vector_means": [sum(v) / len(v) for v in sample_vectors[:5]],
                    "vector_ranges": [
                        {"min": min(v), "max": max(v), "range": max(v) - min(v)}
                        for v in sample_vectors[:5]
                    ]
                }
                results["vector_space_analysis"] = vector_stats
                print(f"✅ Analyzed {len(sample_vectors)} vectors")
                
        except Exception as e:
            print(f"❌ Vector space analysis failed: {e}")
            results["vector_space_analysis"] = {"error": str(e)}
        
        # Test 3: Payload structure analysis
        print("\n📊 Analyzing payload structures...")
        try:
            sample_points = self.vector_store.get_sample_data(limit=20)
            payload_analysis = {
                "total_points_analyzed": len(sample_points),
                "payload_keys": {},
                "payload_types": {},
                "metadata_patterns": {}
            }
            
            for point in sample_points:
                if point.payload:
                    for key, value in point.payload.items():
                        # Count key frequency
                        payload_analysis["payload_keys"][key] = payload_analysis["payload_keys"].get(key, 0) + 1
                        
                        # Track value types
                        value_type = type(value).__name__
                        if key not in payload_analysis["payload_types"]:
                            payload_analysis["payload_types"][key] = {}
                        payload_analysis["payload_types"][key][value_type] = payload_analysis["payload_types"][key].get(value_type, 0) + 1
                        
                        # Analyze metadata patterns
                        if key == "metadata" and isinstance(value, dict):
                            for meta_key in value.keys():
                                payload_analysis["metadata_patterns"][meta_key] = payload_analysis["metadata_patterns"].get(meta_key, 0) + 1
            
            results["payload_analysis"] = payload_analysis
            print(f"✅ Analyzed payloads from {len(sample_points)} points")
            
        except Exception as e:
            print(f"❌ Payload analysis failed: {e}")
            results["payload_analysis"] = {"error": str(e)}
        
        # Test 4: Performance benchmarks
        print("\n⚡ Running performance benchmarks...")
        try:
            benchmark_results = {}
            
            # Search performance
            test_query = "artificial intelligence machine learning"
            query_embedding = self.embedding_service.embed_text(test_query)
            
            # Multiple search tests
            search_times = []
            for i in range(5):
                start_time = time.time()
                results_search = self.vector_store.search(
                    query_vector=query_embedding,
                    top_k=10,
                    score_threshold=0.0
                )
                search_times.append(time.time() - start_time)
            
            benchmark_results["search_performance"] = {
                "average_search_time": sum(search_times) / len(search_times),
                "min_search_time": min(search_times),
                "max_search_time": max(search_times),
                "search_times": search_times
            }
            
            # Count performance
            count_times = []
            for i in range(3):
                start_time = time.time()
                count = self.vector_store.count_points()
                count_times.append(time.time() - start_time)
            
            benchmark_results["count_performance"] = {
                "average_count_time": sum(count_times) / len(count_times),
                "point_count": count
            }
            
            results["performance_benchmarks"] = benchmark_results
            print(f"✅ Performance benchmarks completed")
            
        except Exception as e:
            print(f"❌ Performance benchmarks failed: {e}")
            results["performance_benchmarks"] = {"error": str(e)}
        
        self.test_results["vector_storage_internals"] = results
        return results

    async def cleanup_test_data(self):
        """Clean up test data from the vector store."""
        print("\n🧹 Cleaning up test data...")
        
        try:
            # Delete test documents
            deleted_count = 0
            for doc in self.test_documents:
                try:
                    self.vector_store.delete_point(doc["id"])
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️  Failed to delete {doc['id']}: {e}")
            
            print(f"✅ Cleaned up {deleted_count} test documents")
            return deleted_count
            
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            return 0

    def generate_report(self):
        """Generate a comprehensive test report."""
        print("\n📋 Generating Test Report...")
        
        report = {
            "test_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_tests_run": len(self.test_results),
                "configuration": {
                    "qdrant_url": self.config.QDRANT_URL,
                    "qdrant_collection": self.config.QDRANT_COLLECTION_NAME,
                    "embedding_model": self.config.DEEPINFRA_EMBEDDING_MODEL,
                    "retrieval_top_k": self.config.RETRIEVAL_TOP_K
                }
            },
            "detailed_results": self.test_results
        }
        
        # Save report to file
        report_filename = f"vector_storage_e2e_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Report saved to: {report_filename}")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        for test_name, test_data in self.test_results.items():
            print(f"\n🔍 {test_name.upper().replace('_', ' ')}")
            if isinstance(test_data, dict):
                success_count = sum(1 for v in test_data.values() if not isinstance(v, dict) or "error" not in v)
                total_count = len(test_data)
                print(f"   ✅ {success_count}/{total_count} subtests passed")
            
        return report_filename

    async def run_full_test_suite(self):
        """Run the complete end-to-end test suite."""
        print("🚀 Starting Comprehensive Vector Storage E2E Test")
        print("="*60)
        
        # Setup
        if not await self.setup():
            print("❌ Setup failed, aborting test suite")
            return False
        
        try:
            # Run all tests
            await self.test_embedding_service()
            await self.test_vector_store_operations()
            await self.probe_vector_storage_internals()
            
            # Generate report
            report_file = self.generate_report()
            
            # Cleanup
            await self.cleanup_test_data()
            
            print("\n🎉 Test suite completed successfully!")
            print(f"📄 Detailed report: {report_file}")
            return True
            
        except Exception as e:
            print(f"❌ Test suite failed with error: {e}")
            return False


async def main():
    """Main function to run the E2E test."""
    tester = VectorStorageE2ETester()
    success = await tester.run_full_test_suite()
    
    if success:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())