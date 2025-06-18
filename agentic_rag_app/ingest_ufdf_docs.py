#!/usr/bin/env python3
"""Script to ingest UFDF documents into Qdrant collection
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector_store.qdrant_client import get_qdrant_store

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

async def main() -> None:
    """Main function to ingest UFDF documents"""
    # Path to the UFDF documents
    docs_path = "/Users/clssck/Library/CloudStorage/OneDrive-Personal/RAG_Projects/test_data/ufdf_docs"

    # Check if directory exists
    if not os.path.exists(docs_path):
        logger.error(f"Directory not found: {docs_path}")
        return

    # Get Qdrant store instance
    logger.info("Initializing Qdrant store...")
    qdrant_store = get_qdrant_store()

    # Get collection info before ingestion
    logger.info("Getting collection info before ingestion...")
    info_before = qdrant_store.get_collection_info()
    logger.info(f"Collection info before: {info_before}")

    # Ingest documents
    logger.info(f"Starting ingestion from directory: {docs_path}")
    try:
        result = await qdrant_store.ingest_documents(docs_path)
        logger.info("Ingestion completed successfully!")
        logger.info(f"Ingestion result: {result}")

        # Get collection info after ingestion
        logger.info("Getting collection info after ingestion...")
        info_after = qdrant_store.get_collection_info()
        logger.info(f"Collection info after: {info_after}")

        # Show document count increase
        docs_before = info_before.get("documents_count", 0) if info_before else 0
        docs_after = info_after.get("documents_count", 0) if info_after else 0
        docs_added = docs_after - docs_before

        logger.info(f"Documents added: {docs_added}")
        logger.info(f"Total documents in collection: {docs_after}")

    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
