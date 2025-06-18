#!/usr/bin/env python3
"""Test the query engine directly"""

import os
from dotenv import load_dotenv
from vector_store.qdrant_client import get_qdrant_store
from models.factory import get_model_factory

load_dotenv()

def test_query_engine():
    print("Testing query engine directly...")
    
    # Get components
    qdrant_store = get_qdrant_store()
    model_factory = get_model_factory()
    
    chat_model = model_factory.get_chat_model("qwen3-30b-a3b")
    embedding_model = model_factory.get_embedding_model("qwen3-embed")
    
    from llama_index.core.indices.prompt_helper import PromptHelper
    
    print("Creating query engine with explicit context limits...")
    
    # Set proper context limits for Qwen 3 30B (128K context window)
    prompt_helper = PromptHelper(
        context_window=128000,
        num_output=4000,
        chunk_overlap_ratio=0.1,
        chunk_size_limit=None  # Let it use optimal chunking
    )
    
    query_engine = qdrant_store.get_query_engine(
        similarity_top_k=10,  # Can use more with 128K context
        llm=chat_model,
        embed_model=embedding_model,
        response_mode="tree_summarize",  # Can use better mode now
        prompt_helper=prompt_helper
    )
    
    print("Testing query...")
    response = query_engine.query("What is UFDF?")
    print(f"Response: {response}")

if __name__ == "__main__":
    test_query_engine()