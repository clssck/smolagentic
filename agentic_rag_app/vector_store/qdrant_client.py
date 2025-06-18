"""Qdrant vector store integration for enhanced document retrieval.

This module provides the HybridQdrantStore class that handles document storage,
indexing, and retrieval using Qdrant vector database with advanced features
like hierarchical chunking and metadata extraction.
"""

import logging
import os
from typing import Any

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.extractors import (
    KeywordExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
)
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from models.factory import get_model_factory
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from utils.config_loader import ModelType, get_config_loader
from utils.docling_integration import get_docling_processor, is_document_supported

logger = logging.getLogger(__name__)

class HybridQdrantStore:
    """Enhanced Qdrant vector store with hierarchical chunking and metadata extraction.

    Provides advanced document storage and retrieval capabilities using Qdrant
    vector database with support for various document formats and intelligent
    chunking strategies.
    """

    def __init__(self,
                 qdrant_url: str | None = None,
                 collection_name: str = "agentic_rag",
                 embedding_model_name: str = "qwen3-embed") -> None:
        """Initialize the HybridQdrantStore.

        Args:
            qdrant_url: URL of the Qdrant vector database instance
            collection_name: Name of the collection to use/create
            embedding_model_name: Name of the embedding model to use
        """
        self.config = get_config_loader()
        self.model_factory = get_model_factory()
        self.collection_name = collection_name

        # Use environment variable if qdrant_url not provided
        if qdrant_url is None:
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

        # Initialize Qdrant client
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if qdrant_api_key:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            self.client = QdrantClient(url=qdrant_url)

        # Get embedding model
        self.embedding_model = self.model_factory.get_embedding_model(embedding_model_name)
        embed_config = self.config.get_model_config(embedding_model_name, ModelType.EMBEDDING)
        self.embed_dim = embed_config.get("dimensions", 4096)

        # Setup collection
        self._setup_collection()

        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
        )

        # Setup hybrid chunking pipeline
        self._setup_chunking_pipeline()

    def _setup_collection(self) -> None:
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name
                for col in collections.collections
            )

            if not collection_exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embed_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection: %s", self.collection_name)
            else:
                logger.info("Using existing Qdrant collection: %s", self.collection_name)

        except Exception:
            logger.exception("Error setting up Qdrant collection")
            raise

    def _setup_chunking_pipeline(self) -> None:
        # Get chat model for extractors (use default from env)
        default_chat_model = os.getenv("DEFAULT_CHAT_MODEL", "qwen3-32b")
        chat_model = self.model_factory.get_chat_model(default_chat_model)

        # Hybrid chunking strategy
        node_parsers = [
            # Hierarchical chunking for document structure
            HierarchicalNodeParser.from_defaults(
                chunk_sizes=[2048, 512, 128],
            ),
            # Semantic chunking for meaning preservation
            SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=self.embedding_model,
            ),
            # Sentence-based chunking as fallback
            SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50,
            ),
        ]

        # Metadata extractors for enhanced retrieval
        extractors = [
            TitleExtractor(nodes=5, llm=chat_model),
            QuestionsAnsweredExtractor(questions=3, llm=chat_model),
            SummaryExtractor(summaries=["prev", "self"], llm=chat_model),
            KeywordExtractor(keywords=10, llm=chat_model),
        ]

        # Create ingestion pipeline
        self.pipeline = IngestionPipeline(
            transformations=node_parsers + extractors,
            vector_store=self.vector_store,
        )

    def ingest_documents(self, input_dir: str = "test_data") -> dict[str, Any]:
        """Ingest documents from a directory into the vector store.

        Args:
            input_dir: Directory containing documents to ingest

        Returns:
            Dictionary with ingestion status and statistics
        """
        try:
            # Get Docling processor
            docling_processor = get_docling_processor()

            # Find all supported files in directory
            from pathlib import Path

            input_path = Path(input_dir)
            if not input_path.exists():
                logger.error("Input directory does not exist: %s", input_dir)
                return {"status": "error", "message": f"Directory not found: {input_dir}"}

            documents = []
            supported_files = []

            # Collect all supported files
            for file_path in input_path.rglob("*"):
                if file_path.is_file() and is_document_supported(file_path):
                    supported_files.append(file_path)

            if not supported_files:
                logger.warning("No supported documents found in %s", input_dir)
                return {"status": "warning", "message": "No supported documents found", "count": 0}

            logger.info("Found %d supported files to process", len(supported_files))

            # Process each file with Docling
            for file_path in supported_files:
                try:
                    # Convert document to markdown using Docling
                    markdown_content = docling_processor.convert_to_markdown(file_path)

                    if markdown_content:
                        # Extract metadata
                        metadata = docling_processor.extract_metadata(file_path)
                        metadata["source"] = str(file_path)
                        metadata["processed_with"] = "docling"

                        # Create LlamaIndex Document
                        doc = Document(
                            text=markdown_content,
                            metadata=metadata,
                        )
                        documents.append(doc)
                        logger.info("Successfully processed %s", file_path.name)
                    else:
                        logger.warning("Failed to extract content from %s", file_path.name)

                except Exception:
                    logger.exception("Error processing %s", file_path.name)
                    continue

            if not documents:
                logger.warning("No documents were successfully processed")
                return {"status": "warning", "message": "No documents processed successfully", "count": 0}

            logger.info("Found %d documents to ingest", len(documents))

            # Process documents through pipeline
            nodes = self.pipeline.run(documents=documents)

            # Create storage context and index
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
            )

            self.index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=self.embedding_model,
            )

            logger.info("Successfully ingested %d nodes from %d documents", len(nodes), len(documents))

            return {
                "status": "success",
                "documents_count": len(documents),
                "nodes_count": len(nodes),
                "collection": self.collection_name,
            }

        except Exception as e:
            logger.exception("Error ingesting documents")
            return {"status": "error", "message": str(e)}

    def get_retriever(self, similarity_top_k: int = 5, **kwargs: Any) -> Any:
        """Get a retriever for similarity search.

        Args:
            similarity_top_k: Number of top similar results to retrieve
            **kwargs: Additional arguments for retriever configuration

        Returns:
            Configured retriever instance
        """
        if not hasattr(self, "index"):
            # Try to load existing index
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
            )
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=storage_context,
                embed_model=self.embedding_model,
            )

        return self.index.as_retriever(
            similarity_top_k=similarity_top_k,
            **kwargs,
        )

    def get_query_engine(self, similarity_top_k: int = 5, **kwargs: Any) -> Any:
        """Get a query engine with retrieval and generation capabilities.

        Args:
            similarity_top_k: Number of top similar results to retrieve
            **kwargs: Additional arguments for query engine configuration

        Returns:
            Configured query engine instance
        """
        if not hasattr(self, "index"):
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
            )
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=storage_context,
                embed_model=self.embedding_model,
            )

        return self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            **kwargs,
        )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for relevant documents using vector similarity.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of search results with content, scores, and metadata
        """
        try:
            retriever = self.get_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query)

            return [
                {
                    "content": node.text,
                    "score": node.score,
                    "metadata": node.metadata,
                    "node_id": node.node_id,
                }
                for node in nodes
            ]
        except Exception:
            logger.exception("Error searching")
            return []

    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the Qdrant collection.

        Returns:
            Dictionary containing collection metadata and statistics
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vector_size": info.config.params.vectors.size,
                "distance_metric": info.config.params.vectors.distance,
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            logger.exception("Error getting collection info")
            return {"error": str(e)}

# Global instance
_qdrant_store = None

def get_qdrant_store(qdrant_url: str | None = None) -> HybridQdrantStore:
    """Get or create a singleton HybridQdrantStore instance.

    Args:
        qdrant_url: URL of the Qdrant vector database instance.
                   If None, uses QDRANT_URL environment variable or default.

    Returns:
        The HybridQdrantStore singleton instance
    """
    global _qdrant_store
    if _qdrant_store is None:
        # Use environment variable if no URL provided
        if qdrant_url is None:
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        
        if qdrant_url is None:
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        _qdrant_store = HybridQdrantStore(qdrant_url=qdrant_url)
    return _qdrant_store

