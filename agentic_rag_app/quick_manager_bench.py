#!/usr/bin/env python3
"""
Quick Manager Agent Benchmarking
Focused testing of manager agent routing and performance
"""

import subprocess
import time
import json
from datetime import datetime

def run_chat_command(query, timeout=30):
    """Run a chat command and measure performance."""
    try:
        start_time = time.time()
        
        result = subprocess.run(
            ["python", "main.py", "--chat", query],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            "query": query,
            "response_time": response_time,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "response_length": len(result.stdout) if result.stdout else 0
        }
        
    except subprocess.TimeoutExpired:
        return {
            "query": query,
            "response_time": timeout,
            "success": False,
            "error": "Timeout",
            "response_length": 0
        }
    except Exception as e:
        return {
            "query": query,
            "response_time": 0,
            "success": False,
            "error": str(e),
            "response_length": 0
        }

def main():
    """Run focused manager agent benchmarks."""
    print("🚀 Quick Manager Agent Benchmark")
    print("="*50)
    
    # Define test queries for different agent types
    test_queries = [
        # Simple queries (should route to direct handling)
        "Hello!",
        "What is 5 + 3?",
        "Good morning",
        
        # Knowledge base queries (should route to RAG agent)
        "What is UFDF?",
        "Tell me about ultrafiltration",
        "What documents are available?",
        
        # Research queries (should route to research agent)
        "What's the weather today?",
        "Latest AI news",
        "Current stock market",
        
        # Code queries (should route to code agent)
        "Write a Python hello world",
        "Create a function to add numbers",
        "Generate code for sorting"
    ]
    
    results = []
    
    print(f"Testing {len(test_queries)} queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Testing: '{query}'")
        
        result = run_chat_command(query)
        results.append(result)
        
        if result["success"]:
            print(f"✅ {result['response_time']:.2f}s | {result['response_length']} chars")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Analyze results
    print("\n" + "="*50)
    print("📊 BENCHMARK RESULTS")
    print("="*50)
    
    successful_queries = [r for r in results if r["success"]]
    failed_queries = [r for r in results if not r["success"]]
    
    print(f"✅ Successful: {len(successful_queries)}/{len(results)} ({len(successful_queries)/len(results)*100:.1f}%)")
    print(f"❌ Failed: {len(failed_queries)}")
    
    if successful_queries:
        response_times = [r["response_time"] for r in successful_queries]
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"\n⚡ Performance:")
        print(f"   Average: {avg_time:.2f}s")
        print(f"   Range: {min_time:.2f}s - {max_time:.2f}s")
        
        # Categorize by query type
        simple_queries = ["Hello!", "What is 5 + 3?", "Good morning"]
        kb_queries = ["What is UFDF?", "Tell me about ultrafiltration", "What documents are available?"]
        research_queries = ["What's the weather today?", "Latest AI news", "Current stock market"]
        code_queries = ["Write a Python hello world", "Create a function to add numbers", "Generate code for sorting"]
        
        categories = {
            "Simple": simple_queries,
            "Knowledge Base": kb_queries,
            "Research": research_queries,
            "Code": code_queries
        }
        
        print(f"\n🎯 Performance by Category:")
        for category, queries in categories.items():
            category_results = [r for r in successful_queries if r["query"] in queries]
            if category_results:
                cat_times = [r["response_time"] for r in category_results]
                cat_avg = sum(cat_times) / len(cat_times)
                print(f"   {category}: {cat_avg:.2f}s avg ({len(category_results)}/{len(queries)} success)")
    
    if failed_queries:
        print(f"\n❌ Failed Queries:")
        for result in failed_queries:
            error = result.get("error", "Unknown")
            print(f"   '{result['query']}': {error}")
    
    # Save detailed results
    report_filename = f"quick_manager_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(results),
            "successful_queries": len(successful_queries),
            "failed_queries": len(failed_queries),
            "results": results
        }, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {report_filename}")
    print("✅ Quick benchmark completed!")

if __name__ == "__main__":
    main()