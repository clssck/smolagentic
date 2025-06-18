#!/usr/bin/env python3
"""Minimal agent test with detailed logging"""

import os
import logging
from dotenv import load_dotenv
from agents.rag_agent import AgenticRAG

# Setup detailed logging
logging.basicConfig(level=logging.DEBUG)

load_dotenv()

def minimal_test():
    print("Creating minimal agent...")
    
    try:
        # Create agent with minimal configuration
        agent = AgenticRAG(
            chat_model_name="qwen3-30b-a3b",
            enable_reasoning=False  # Disable reasoning to simplify
        )
        print("Agent created successfully!")
        
        print("Testing chat...")
        response = agent.chat("hello")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    minimal_test()