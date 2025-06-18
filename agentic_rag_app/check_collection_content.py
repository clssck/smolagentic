#!/usr/bin/env python3
"""Check what's actually in the Qdrant collection"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

def check_content():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    collection_name = "agentic_rag"
    
    # Get a few sample documents
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=5,
        with_payload=True,
        with_vectors=False
    )
    
    print(f"Sample documents from {collection_name}:")
    print("=" * 50)
    
    for i, point in enumerate(points, 1):
        payload = point.payload or {}
        text = payload.get('text', '')
        
        print(f"\nDocument {i}:")
        print(f"  ID: {point.id}")
        print(f"  Text (first 200 chars): {text[:200]}...")
        
        # Check for variations of UFDF
        text_lower = text.lower()
        if any(term in text_lower for term in ['ufdf', 'unified', 'framework', 'definition']):
            print(f"  *** Contains potential UFDF-related terms ***")
        
        # Show metadata
        metadata = {k: v for k, v in payload.items() if k != 'text'}
        if metadata:
            print(f"  Metadata: {metadata}")

if __name__ == "__main__":
    check_content()