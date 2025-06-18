#!/usr/bin/env python3
"""Simple Qdrant query for UFDF"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

def simple_ufdf_query():
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if qdrant_api_key:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        client = QdrantClient(url=qdrant_url)
    
    collection_name = "agentic_rag"
    
    # Get collection info
    collection = client.get_collection(collection_name)
    print(f"Collection: {collection_name}")
    print(f"Total points: {collection.points_count}")
    
    # Search by scrolling through documents looking for "ufdf"
    print("\nSearching for UFDF references...")
    
    points, next_page_offset = client.scroll(
        collection_name=collection_name,
        limit=100,
        with_payload=True,
        with_vectors=False
    )
    
    found_count = 0
    for point in points:
        payload = point.payload or {}
        text = payload.get('text', '').lower()
        
        if 'ufdf' in text:
            found_count += 1
            print(f"\n=== FOUND UFDF REFERENCE #{found_count} ===")
            print(f"Point ID: {point.id}")
            
            # Show context around UFDF
            ufdf_index = text.find('ufdf')
            start = max(0, ufdf_index - 100)
            end = min(len(text), ufdf_index + 100)
            context = text[start:end]
            
            print(f"Context: ...{context}...")
            
            # Show metadata
            metadata = {k: v for k, v in payload.items() if k != 'text'}
            if metadata:
                print(f"Metadata: {metadata}")
    
    if found_count == 0:
        print("No UFDF references found in first 100 documents")
        print("Let me search more...")
        
        # Search more documents if needed
        while next_page_offset and found_count == 0:
            points, next_page_offset = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False
            )
            
            for point in points:
                payload = point.payload or {}
                text = payload.get('text', '').lower()
                
                if 'ufdf' in text:
                    found_count += 1
                    print(f"\n=== FOUND UFDF REFERENCE #{found_count} ===")
                    print(f"Point ID: {point.id}")
                    
                    ufdf_index = text.find('ufdf')
                    start = max(0, ufdf_index - 100)
                    end = min(len(text), ufdf_index + 100)
                    context = text[start:end]
                    
                    print(f"Context: ...{context}...")
                    
                    metadata = {k: v for k, v in payload.items() if k != 'text'}
                    if metadata:
                        print(f"Metadata: {metadata}")
                    break
            
            if found_count > 0:
                break

if __name__ == "__main__":
    simple_ufdf_query()