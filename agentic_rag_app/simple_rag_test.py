#!/usr/bin/env python3
"""Simple RAG test without ReAct agent complexity"""

import os
from dotenv import load_dotenv
from tools.enhanced_search_tool import EnhancedSearchTool
from models.factory import get_model_factory

load_dotenv()

def simple_rag_test():
    print("Simple RAG test without ReAct agent...")
    
    # Use direct search + LLM without ReAct wrapper
    search_tool = EnhancedSearchTool()
    model_factory = get_model_factory()
    
    # Get a chat model
    chat_model = model_factory.get_chat_model("qwen3-30b-a3b")
    
    # Test query
    query = "What is UFDF?"
    
    print(f"Searching for: {query}")
    search_results = search_tool.call(query)
    
    print(f"Search results:\n{search_results[:500]}...")
    
    # Create a simple prompt
    prompt = f"""Based on the following search results, answer the question: {query}

Search Results:
{search_results}

Please provide a clear, informative answer based on the search results above."""
    
    print("\nGenerating response...")
    response = chat_model.complete(prompt)
    
    print(f"Final Answer:\n{response}")

if __name__ == "__main__":
    simple_rag_test()