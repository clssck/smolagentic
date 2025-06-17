#!/usr/bin/env python3
"""
Simple script to ingest UFDF documents into Qdrant collection
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from qdrant_client import QdrantClient
from models.factory import get_model_factory

def main():
    """Simple ingestion"""
    print("Starting simple ingestion...")
    
    # Get embedding model
    model_factory = get_model_factory()
    embed_model = model_factory.get_embedding_model("qwen3-embed")
    
    # Setup Qdrant client
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    # Create vector store
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name="agentic_rag"
    )
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Load documents
    docs_path = "/Users/clssck/Library/CloudStorage/OneDrive-Personal/RAG_Projects/test_data/ufdf_docs"
    print(f"Loading documents from: {docs_path}")
    
    documents = SimpleDirectoryReader(docs_path).load_data()
    print(f"Loaded {len(documents)} documents")
    
    # Create index and ingest
    print("Creating index and ingesting documents...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    
    print("Ingestion completed!")
    
    # Check collection status
    try:
        info = client.get_collection("agentic_rag")
        print(f"Collection points count: {info.points_count}")
    except Exception as e:
        print(f"Error getting collection info: {e}")

if __name__ == "__main__":
    main()