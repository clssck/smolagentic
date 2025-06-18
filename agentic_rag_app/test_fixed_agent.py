#!/usr/bin/env python3
"""Test the fixed RAG agent with UFDF query"""

import os
from dotenv import load_dotenv
from agents.rag_agent import get_agentic_rag

load_dotenv()

def test_fixed_agent():
    print("Testing fixed RAG agent...")
    
    # Get the agent instance (will use environment variable automatically)
    qdrant_url = os.getenv("QDRANT_URL")
    print(f"QDRANT_URL from env: {qdrant_url}")
    
    agent = get_agentic_rag(enable_reasoning=True)
    
    # Test the UFDF query that was causing issues
    print("\n=== Testing UFDF Query ===")
    try:
        response = agent.chat("What is ufdf?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test a few more queries
    test_queries = [
        "Tell me about UFDF",
        "What does UFDF stand for?",
        "Explain the UFDF framework"
    ]
    
    for query in test_queries:
        print(f"\n=== Testing: {query} ===")
        try:
            response = agent.chat(query)
            print(f"Response: {response[:300]}...")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_fixed_agent()