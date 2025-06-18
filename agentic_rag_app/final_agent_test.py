#!/usr/bin/env python3
"""Final test of the fixed agent"""

import os
from dotenv import load_dotenv
from agents.rag_agent import get_agentic_rag

load_dotenv()

def final_test():
    print("=== FINAL AGENT TEST ===")
    
    # Test with reasoning enabled
    agent = get_agentic_rag(enable_reasoning=True)
    
    test_queries = [
        "What is UFDF?",
        "Tell me about ultrafiltration",
        "What is diafiltration?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)
        try:
            response = agent.chat(query)
            print(f"✅ Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print()

if __name__ == "__main__":
    final_test()