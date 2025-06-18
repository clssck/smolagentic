#!/usr/bin/env python3
"""Quick test of the fixed agent"""

import os
from dotenv import load_dotenv
from agents.rag_agent import get_agentic_rag

load_dotenv()

def quick_test():
    print("Quick agent test...")
    
    # Test just one simple query
    agent = get_agentic_rag(enable_reasoning=False)  # Disable reasoning for speed
    
    print("Testing UFDF query...")
    try:
        response = agent.chat("What is UFDF?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()