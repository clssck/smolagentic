#!/usr/bin/env python3
"""Direct test of UFDF search without ReAct agent complexity"""

import os
from dotenv import load_dotenv
from tools.enhanced_search_tool import EnhancedSearchTool

load_dotenv()

def test_direct_search():
    qdrant_url = os.getenv("QDRANT_URL")
    print(f"Using QDRANT_URL: {qdrant_url}")
    
    # Test the enhanced search tool directly
    search_tool = EnhancedSearchTool(qdrant_url=qdrant_url)
    
    print("\n=== Testing Enhanced Search Tool ===")
    result = search_tool.call("ufdf")
    print(f"Search result:\n{result}")
    
    print("\n=== Testing UFDF variations ===")
    variations = ["UFDF", "ufdf overview", "unified file definition format"]
    
    for query in variations:
        print(f"\n--- Query: {query} ---")
        result = search_tool.call(query)
        print(f"Result: {result[:300]}...")

if __name__ == "__main__":
    test_direct_search()