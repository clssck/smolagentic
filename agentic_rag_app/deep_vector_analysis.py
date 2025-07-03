#!/usr/bin/env python3
"""
Deep Vector Storage Analysis
Comprehensive examination of vector storage with real embeddings and searches
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, List

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import Config
from src.utils.embedding_service import EmbeddingService
from vector_store.qdrant_client import QdrantVectorStore

def analyze_vector_storage():
    """Perform deep analysis of the vector storage system."""
    print("🔬 Deep Vector Storage Analysis")
    print("="*70)
    
    # Initialize services
    config = Config()
    embedding_service = EmbeddingService()
    vector_store = QdrantVectorStore(
        collection_name=config.QDRANT_COLLECTION_NAME,
        config=config
    )
    
    # Collect all analysis results
    analysis_results = {
        "timestamp": datetime.now().isoformat(),
        "collection_name": config.QDRANT_COLLECTION_NAME,
        "total_points": vector_store.count_points()
    }
    
    print(f"📊 Collection: {config.QDRANT_COLLECTION_NAME}")
    print(f"📈 Total Points: {analysis_results['total_points']}")
    
    # 1. DETAILED SAMPLE DATA ANALYSIS
    print("\n" + "="*50)
    print("📋 DETAILED SAMPLE DATA ANALYSIS")
    print("="*50)
    
    sample_points = vector_store.get_points(limit=10)
    sample_analysis = {
        "total_samples": len(sample_points),
        "payload_structures": {},
        "content_types": {},
        "data_sources": set(),
        "file_types": set()
    }
    
    for i, point in enumerate(sample_points):
        print(f"\n--- Sample Point {i+1} ---")
        print(f"ID: {point.get('id', 'N/A')}")
        
        payload = point.get('payload', {})
        content = point.get('text') or point.get('content', '')
        
        # Analyze payload structure
        if payload:
            payload_keys = str(sorted(payload.keys()))  # Convert to string for JSON serialization
            sample_analysis["payload_structures"][payload_keys] = sample_analysis["payload_structures"].get(payload_keys, 0) + 1
            
            # Track data sources
            if 'file_path' in payload:
                sample_analysis["data_sources"].add(payload.get('file_path', 'unknown'))
            if 'source' in payload:
                sample_analysis["data_sources"].add(payload.get('source', 'unknown'))
            if 'file_type' in payload:
                sample_analysis["file_types"].add(payload.get('file_type', 'unknown'))
            
            print(f"Payload Keys: {list(payload.keys())}")
            
            # Show specific payload details
            for key in ['file_path', 'source', 'file_name', 'file_type', '_node_type']:
                if key in payload:
                    print(f"  {key}: {payload[key]}")
        
        # Analyze content
        if content:
            content_preview = content[:200] + "..." if len(content) > 200 else content
            print(f"Content: {content_preview}")
            
            # Categorize content type
            if len(content) > 500:
                content_type = "long_text"
            elif len(content) > 100:
                content_type = "medium_text"
            elif len(content) > 0:
                content_type = "short_text"
            else:
                content_type = "empty"
            
            sample_analysis["content_types"][content_type] = sample_analysis["content_types"].get(content_type, 0) + 1
        else:
            sample_analysis["content_types"]["empty"] = sample_analysis["content_types"].get("empty", 0) + 1
    
    analysis_results["sample_analysis"] = {
        "total_samples": sample_analysis["total_samples"],
        "payload_structures": dict(sample_analysis["payload_structures"]),
        "content_types": sample_analysis["content_types"],
        "data_sources": list(sample_analysis["data_sources"])[:10],  # Limit to first 10
        "file_types": list(sample_analysis["file_types"])
    }
    
    # 2. COMPREHENSIVE SEARCH TESTING
    print("\n" + "="*50)
    print("🔍 COMPREHENSIVE SEARCH TESTING")
    print("="*50)
    
    search_queries = [
        "artificial intelligence machine learning",
        "vector database storage retrieval",
        "python programming code development",
        "data processing analysis",
        "neural networks deep learning",
        "natural language processing NLP",
        "information retrieval search",
        "document processing text",
        "embeddings similarity search",
        "RAG retrieval augmented generation"
    ]
    
    search_results = {}
    
    for query in search_queries:
        print(f"\n🔍 Searching: '{query}'")
        try:
            # Generate embedding for the query
            query_embedding = embedding_service.embed_text(query)
            
            # Perform vector search
            results = vector_store.search(
                query_vector=query_embedding,
                top_k=5,
                score_threshold=0.3
            )
            
            print(f"   Found {len(results)} results")
            
            search_analysis = {
                "query": query,
                "num_results": len(results),
                "results": []
            }
            
            for j, result in enumerate(results):
                if isinstance(result, dict):
                    # Handle dict-style results
                    result_data = {
                        "rank": j + 1,
                        "id": result.get('id', 'N/A'),
                        "score": result.get('score', 0.0),
                        "content_preview": (result.get('text') or result.get('content', ''))[:150] + "...",
                        "source": result.get('payload', {}).get('source', result.get('payload', {}).get('file_path', 'Unknown'))
                    }
                else:
                    # Handle object-style results
                    content = ""
                    if hasattr(result, 'payload') and result.payload:
                        content = result.payload.get('text', result.payload.get('content', result.payload.get('_node_content', '')))
                    
                    result_data = {
                        "rank": j + 1,
                        "id": getattr(result, 'id', 'N/A'),
                        "score": getattr(result, 'score', 0.0),
                        "content_preview": content[:150] + "..." if content else "No content",
                        "source": result.payload.get('source', result.payload.get('file_path', 'Unknown')) if hasattr(result, 'payload') and result.payload else 'Unknown'
                    }
                
                search_analysis["results"].append(result_data)
                print(f"   [{j+1}] Score: {result_data['score']:.3f} | {result_data['content_preview']}")
            
            search_results[query] = search_analysis
            
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            search_results[query] = {"error": str(e)}
    
    analysis_results["search_analysis"] = search_results
    
    # 3. VECTOR SPACE ANALYSIS
    print("\n" + "="*50)
    print("🧮 VECTOR SPACE ANALYSIS")
    print("="*50)
    
    try:
        # Get more points with vectors for analysis
        vector_points = []
        all_points = vector_store.get_points(limit=50)
        
        for point in all_points:
            if isinstance(point, dict) and 'vector' in point and point['vector']:
                vector_points.append(point['vector'])
            elif hasattr(point, 'vector') and point.vector:
                vector_points.append(point.vector)
        
        if vector_points:
            vector_analysis = {
                "num_vectors_analyzed": len(vector_points),
                "vector_dimension": len(vector_points[0]) if vector_points else 0,
                "vector_statistics": []
            }
            
            # Analyze first few vectors in detail
            for i, vector in enumerate(vector_points[:5]):
                stats = {
                    "vector_index": i,
                    "dimension": len(vector),
                    "magnitude": sum(x**2 for x in vector)**0.5,
                    "mean": sum(vector) / len(vector),
                    "min_value": min(vector),
                    "max_value": max(vector),
                    "non_zero_count": sum(1 for x in vector if abs(x) > 1e-10),
                    "sparsity": 1.0 - (sum(1 for x in vector if abs(x) > 1e-10) / len(vector))
                }
                vector_analysis["vector_statistics"].append(stats)
                
                print(f"Vector {i+1}: dim={stats['dimension']}, mag={stats['magnitude']:.3f}, mean={stats['mean']:.6f}")
            
            analysis_results["vector_space_analysis"] = vector_analysis
            
        else:
            print("⚠️  No vectors found in sample data")
            analysis_results["vector_space_analysis"] = {"error": "No vectors found"}
    
    except Exception as e:
        print(f"❌ Vector analysis failed: {e}")
        analysis_results["vector_space_analysis"] = {"error": str(e)}
    
    # 4. PERFORMANCE BENCHMARKING
    print("\n" + "="*50)
    print("⚡ PERFORMANCE BENCHMARKING")
    print("="*50)
    
    import time
    
    # Embedding performance
    test_texts = [
        "Short query",
        "This is a medium length query with multiple words and concepts to test embedding performance",
        "This is a very long query that contains many different concepts, words, phrases, and ideas to thoroughly test the embedding generation performance and see how it handles longer text inputs with diverse vocabulary and semantic content."
    ]
    
    embedding_performance = []
    for text in test_texts:
        start_time = time.time()
        embedding = embedding_service.embed_text(text)
        duration = time.time() - start_time
        
        perf_data = {
            "text_length": len(text),
            "words": len(text.split()),
            "embedding_time": duration,
            "embedding_dimension": len(embedding),
            "tokens_per_second": len(text.split()) / duration if duration > 0 else 0
        }
        embedding_performance.append(perf_data)
        print(f"Embedding: {perf_data['words']} words in {duration:.3f}s ({perf_data['tokens_per_second']:.1f} words/sec)")
    
    # Search performance
    test_query = "machine learning artificial intelligence"
    query_embedding = embedding_service.embed_text(test_query)
    
    search_times = []
    for i in range(10):
        start_time = time.time()
        results = vector_store.search(query_vector=query_embedding, top_k=10)
        search_times.append(time.time() - start_time)
    
    search_performance = {
        "average_search_time": sum(search_times) / len(search_times),
        "min_search_time": min(search_times),
        "max_search_time": max(search_times),
        "search_trials": len(search_times)
    }
    
    print(f"Search Performance: avg={search_performance['average_search_time']:.3f}s, min={search_performance['min_search_time']:.3f}s, max={search_performance['max_search_time']:.3f}s")
    
    analysis_results["performance_analysis"] = {
        "embedding_performance": embedding_performance,
        "search_performance": search_performance
    }
    
    # 5. SAVE COMPREHENSIVE REPORT
    report_filename = f"deep_vector_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"\n📄 Comprehensive analysis saved to: {report_filename}")
    
    # 6. SUMMARY
    print("\n" + "="*70)
    print("📊 ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"🔢 Collection '{config.QDRANT_COLLECTION_NAME}' contains {analysis_results['total_points']} points")
    
    if 'sample_analysis' in analysis_results:
        print(f"📝 Content types found: {list(analysis_results['sample_analysis']['content_types'].keys())}")
        print(f"📁 Data sources: {len(analysis_results['sample_analysis']['data_sources'])} different sources")
        print(f"📄 File types: {analysis_results['sample_analysis']['file_types']}")
    
    successful_searches = sum(1 for result in search_results.values() if 'error' not in result)
    print(f"🔍 {successful_searches}/{len(search_queries)} search queries returned results")
    
    if 'vector_space_analysis' in analysis_results and 'error' not in analysis_results['vector_space_analysis']:
        va = analysis_results['vector_space_analysis']
        print(f"🧮 Vector analysis: {va['num_vectors_analyzed']} vectors, {va['vector_dimension']}D space")
    
    if 'performance_analysis' in analysis_results:
        pa = analysis_results['performance_analysis']
        avg_embedding_time = sum(p['embedding_time'] for p in pa['embedding_performance']) / len(pa['embedding_performance'])
        print(f"⚡ Performance: {avg_embedding_time:.3f}s avg embedding, {pa['search_performance']['average_search_time']:.3f}s avg search")
    
    print("\n✅ Deep vector storage analysis completed successfully!")
    return report_filename

if __name__ == "__main__":
    analyze_vector_storage()