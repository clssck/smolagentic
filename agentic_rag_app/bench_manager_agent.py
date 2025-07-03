#!/usr/bin/env python3
"""
Manager Agent Benchmarking Script
Comprehensive benchmarking of the manager agent system and vector collection performance
"""

import time
import json
import statistics
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

def benchmark_manager_agent():
    """Comprehensive benchmarking of the manager agent system."""
    
    print("🚀 Manager Agent System Benchmarking")
    print("="*70)
    
    # Initialize the system
    try:
        # Import and initialize the manager system
        from src.core.manager_agent_system import ManagerAgentSystem
        system = ManagerAgentSystem()
        print("✅ Manager Agent System initialized successfully")
        
        # Get system status
        status = system.get_status()
        print(f"📊 System Status: {status.get('status', 'Unknown')}")
        print(f"🤖 Available Agents: {len(status.get('agents', []))}")
        
    except Exception as e:
        print(f"❌ Failed to initialize manager system: {e}")
        return None
    
    # Benchmark results collector
    benchmark_results = {
        "timestamp": datetime.now().isoformat(),
        "system_info": status,
        "query_tests": {},
        "routing_analysis": {},
        "performance_metrics": {},
        "agent_specialization_tests": {}
    }
    
    # 1. COMPREHENSIVE QUERY TESTING
    print("\n" + "="*50)
    print("🧪 COMPREHENSIVE QUERY TESTING")
    print("="*50)
    
    test_queries = {
        # RAG Agent queries (should route to knowledge base)
        "rag_queries": [
            "What is ultrafiltration and how does it work?",
            "Explain the UFDF security model",
            "What are the key features of diafiltration processes?",
            "How does the Agentic RAG system work?",
            "What is Donnan effect in membrane separation?",
            "Describe the UFDF processing pipeline",
            "What documents are available in the knowledge base?",
            "Tell me about membrane separation fundamentals"
        ],
        
        # Research Agent queries (should route to web search)
        "research_queries": [
            "What are the latest developments in AI in 2024?",
            "Current stock price of NVIDIA",
            "Recent news about large language models",
            "What happened in the tech industry this week?",
            "Latest research papers on vector databases",
            "Current weather in San Francisco",
            "Breaking news about artificial intelligence",
            "Recent developments in membrane technology"
        ],
        
        # Code Agent queries (should route to code generation)
        "code_queries": [
            "Write a Python function to calculate membrane flux",
            "Create a class for vector database operations",
            "Generate code for data preprocessing",
            "Write a function to parse configuration files",
            "Create a REST API endpoint for RAG queries",
            "Implement caching for embedding generation",
            "Write unit tests for the manager agent",
            "Create a decorator for performance monitoring"
        ],
        
        # Simple queries (should be handled directly)
        "simple_queries": [
            "Hello there!",
            "What is 2 + 2?",
            "Good morning",
            "How are you?",
            "What's the square root of 64?",
            "Thanks for helping",
            "Calculate 15 * 7",
            "What time is it?"
        ],
        
        # Mixed/Edge case queries
        "mixed_queries": [
            "Can you search for recent papers on ultrafiltration and summarize them?",
            "Write code to implement the UFDF algorithms mentioned in the docs",
            "What's the weather like and how might it affect membrane performance?",
            "Hello! Can you help me understand vector databases and write some code?",
            "Search for current AI trends and relate them to our RAG system",
            "Good morning! What's 5+5 and also explain machine learning?",
            "Write a function and then search for similar implementations online",
            "Greetings! Tell me about our system and the latest tech news"
        ]
    }
    
    # Test each query category
    for category, queries in test_queries.items():
        print(f"\n🔍 Testing {category.upper().replace('_', ' ')}:")
        category_results = []
        
        for i, query in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] Testing: '{query[:60]}{'...' if len(query) > 60 else ''}'")
            
            try:
                # Measure response time
                start_time = time.time()
                response = system.run_query(query)
                end_time = time.time()
                
                response_time = end_time - start_time
                
                # Extract routing information if available
                routing_info = "unknown"
                if hasattr(system, '_last_routing_info'):
                    routing_info = system._last_routing_info
                
                result = {
                    "query": query,
                    "response_time": response_time,
                    "response_length": len(str(response)),
                    "routing_decision": routing_info,
                    "success": True
                }
                
                category_results.append(result)
                print(f"    ✅ {response_time:.3f}s | Routed to: {routing_info}")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                category_results.append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        benchmark_results["query_tests"][category] = category_results
    
    # 2. ROUTING ACCURACY ANALYSIS
    print("\n" + "="*50)
    print("🎯 ROUTING ACCURACY ANALYSIS")
    print("="*50)
    
    expected_routing = {
        "rag_queries": "rag_agent",
        "research_queries": "research_agent", 
        "code_queries": "code_agent",
        "simple_queries": "direct",
        "mixed_queries": "various"  # These can route to different agents
    }
    
    routing_analysis = {}
    total_correct = 0
    total_tested = 0
    
    for category, results in benchmark_results["query_tests"].items():
        if category == "mixed_queries":
            continue  # Skip mixed queries for accuracy calculation
            
        successful_results = [r for r in results if r.get("success", False)]
        expected_route = expected_routing.get(category, "unknown")
        
        correct_routes = 0
        for result in successful_results:
            actual_route = result.get("routing_decision", "unknown")
            if expected_route == "direct" and actual_route in ["direct", "manager"]:
                correct_routes += 1
            elif expected_route != "direct" and expected_route in str(actual_route):
                correct_routes += 1
        
        accuracy = correct_routes / len(successful_results) if successful_results else 0
        
        routing_analysis[category] = {
            "expected_route": expected_route,
            "total_queries": len(results),
            "successful_queries": len(successful_results),
            "correct_routes": correct_routes,
            "accuracy": accuracy
        }
        
        total_correct += correct_routes
        total_tested += len(successful_results)
        
        print(f"  {category}: {correct_routes}/{len(successful_results)} correct ({accuracy:.1%})")
    
    overall_accuracy = total_correct / total_tested if total_tested > 0 else 0
    print(f"\n🎯 Overall Routing Accuracy: {total_correct}/{total_tested} ({overall_accuracy:.1%})")
    
    benchmark_results["routing_analysis"] = routing_analysis
    benchmark_results["overall_accuracy"] = overall_accuracy
    
    # 3. PERFORMANCE METRICS ANALYSIS
    print("\n" + "="*50)
    print("⚡ PERFORMANCE METRICS ANALYSIS")
    print("="*50)
    
    all_response_times = []
    category_performance = {}
    
    for category, results in benchmark_results["query_tests"].items():
        successful_results = [r for r in results if r.get("success", False)]
        if not successful_results:
            continue
            
        response_times = [r["response_time"] for r in successful_results]
        response_lengths = [r["response_length"] for r in successful_results]
        
        all_response_times.extend(response_times)
        
        category_perf = {
            "avg_response_time": statistics.mean(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "median_response_time": statistics.median(response_times),
            "avg_response_length": statistics.mean(response_lengths),
            "total_queries": len(successful_results)
        }
        
        category_performance[category] = category_perf
        
        print(f"  {category}:")
        print(f"    Avg: {category_perf['avg_response_time']:.3f}s")
        print(f"    Range: {category_perf['min_response_time']:.3f}s - {category_perf['max_response_time']:.3f}s")
        print(f"    Avg Length: {category_perf['avg_response_length']:.0f} chars")
    
    # Overall performance
    if all_response_times:
        overall_performance = {
            "total_queries_tested": len(all_response_times),
            "avg_response_time": statistics.mean(all_response_times),
            "min_response_time": min(all_response_times),
            "max_response_time": max(all_response_times),
            "median_response_time": statistics.median(all_response_times),
            "std_dev_response_time": statistics.stdev(all_response_times) if len(all_response_times) > 1 else 0
        }
        
        print(f"\n⚡ Overall Performance:")
        print(f"  Total Queries: {overall_performance['total_queries_tested']}")
        print(f"  Average Response Time: {overall_performance['avg_response_time']:.3f}s")
        print(f"  Median Response Time: {overall_performance['median_response_time']:.3f}s")
        print(f"  Response Time Range: {overall_performance['min_response_time']:.3f}s - {overall_performance['max_response_time']:.3f}s")
        print(f"  Standard Deviation: {overall_performance['std_dev_response_time']:.3f}s")
        
        benchmark_results["performance_metrics"] = {
            "category_performance": category_performance,
            "overall_performance": overall_performance
        }
    
    # 4. AGENT SPECIALIZATION STRESS TEST
    print("\n" + "="*50)
    print("🔥 AGENT SPECIALIZATION STRESS TEST")
    print("="*50)
    
    stress_tests = {
        "complex_rag": [
            "Give me a comprehensive analysis of all ultrafiltration processes mentioned in our documents",
            "Compare and contrast the different membrane separation techniques in our knowledge base",
            "What are all the security considerations mentioned across our UFDF documentation?",
            "Provide a detailed overview of the entire UFDF system based on all available documents"
        ],
        "complex_research": [
            "Find the latest research on membrane fouling mitigation strategies published in 2024",
            "What are the current market trends in water treatment technology?", 
            "Search for recent patents related to ultrafiltration improvements",
            "Find current industry standards for membrane performance testing"
        ],
        "complex_code": [
            "Create a complete Python module for membrane performance calculations with classes and error handling",
            "Design and implement a caching system for the RAG pipeline with Redis integration",
            "Write a comprehensive test suite for the manager agent system",
            "Implement a monitoring dashboard for the agentic RAG system with real-time metrics"
        ]
    }
    
    for test_type, queries in stress_tests.items():
        print(f"\n🔥 {test_type.upper().replace('_', ' ')} STRESS TEST:")
        stress_results = []
        
        for i, query in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] Complex query: '{query[:80]}{'...' if len(query) > 80 else ''}'")
            
            try:
                start_time = time.time()
                response = system.run_query(query)
                end_time = time.time()
                
                response_time = end_time - start_time
                response_quality = len(str(response))  # Simple quality metric
                
                stress_results.append({
                    "query": query,
                    "response_time": response_time,
                    "response_quality_score": response_quality,
                    "success": True
                })
                
                print(f"    ✅ {response_time:.3f}s | Quality Score: {response_quality}")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                stress_results.append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        benchmark_results["agent_specialization_tests"][test_type] = stress_results
    
    # 5. SAVE COMPREHENSIVE BENCHMARK REPORT
    report_filename = f"manager_agent_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w') as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    
    print(f"\n📄 Benchmark report saved to: {report_filename}")
    
    # 6. FINAL SUMMARY
    print("\n" + "="*70)
    print("📊 MANAGER AGENT BENCHMARK SUMMARY")
    print("="*70)
    
    total_queries = sum(len(results) for results in benchmark_results["query_tests"].values())
    successful_queries = sum(len([r for r in results if r.get("success", False)]) 
                           for results in benchmark_results["query_tests"].values())
    
    print(f"🔢 Total Queries Tested: {total_queries}")
    print(f"✅ Successful Queries: {successful_queries} ({successful_queries/total_queries:.1%})")
    print(f"🎯 Routing Accuracy: {overall_accuracy:.1%}")
    
    if "overall_performance" in benchmark_results.get("performance_metrics", {}):
        perf = benchmark_results["performance_metrics"]["overall_performance"]
        print(f"⚡ Average Response Time: {perf['avg_response_time']:.3f}s")
        print(f"📏 Response Time Range: {perf['min_response_time']:.3f}s - {perf['max_response_time']:.3f}s")
    
    # Performance grades
    if overall_accuracy >= 0.9:
        accuracy_grade = "🏆 Excellent"
    elif overall_accuracy >= 0.8:
        accuracy_grade = "🥇 Very Good"
    elif overall_accuracy >= 0.7:
        accuracy_grade = "🥈 Good"
    else:
        accuracy_grade = "🥉 Needs Improvement"
    
    avg_time = benchmark_results.get("performance_metrics", {}).get("overall_performance", {}).get("avg_response_time", 0)
    if avg_time < 2.0:
        speed_grade = "🏆 Excellent"
    elif avg_time < 5.0:
        speed_grade = "🥇 Very Good"
    elif avg_time < 10.0:
        speed_grade = "🥈 Good"
    else:
        speed_grade = "🥉 Needs Improvement"
    
    print(f"\n🏆 PERFORMANCE GRADES:")
    print(f"  Routing Accuracy: {accuracy_grade}")
    print(f"  Response Speed: {speed_grade}")
    
    print("\n✅ Manager Agent benchmarking completed successfully!")
    return report_filename

def main():
    """Main function to run the benchmark."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Manager Agent Benchmarking Script")
        print("Usage: python bench_manager_agent.py")
        print("This script will test the manager agent system performance and routing accuracy.")
        return
    
    report_file = benchmark_manager_agent()
    if report_file:
        print(f"\n📄 Detailed results available in: {report_file}")
        print("🚀 Benchmark completed successfully!")
    else:
        print("❌ Benchmark failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()