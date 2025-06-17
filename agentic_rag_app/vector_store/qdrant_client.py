import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import (
    SentenceSplitter, 
    SemanticSplitterNodeParser,
    HierarchicalNodeParser
)
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.embeddings import BaseEmbedding

from models.factory import get_model_factory
from utils.config_loader import get_config_loader, ModelType
from utils.docling_integration import get_docling_processor, is_document_supported

logger = logging.getLogger(__name__)

class HybridQdrantStore:
    def __init__(self, 
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "agentic_rag",
                 embedding_model_name: str = "qwen3-embed"):
        
        self.config = get_config_loader()
        self.model_factory = get_model_factory()
        self.collection_name = collection_name
        
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
            collection_name=self.collection_name
        )
        
        # Setup hybrid chunking pipeline
        self._setup_chunking_pipeline()
    
    def _setup_collection(self):
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
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error setting up Qdrant collection: {e}")
            raise
    
    def _setup_chunking_pipeline(self):
        # Get chat model for extractors (use default from env)
        default_chat_model = os.getenv("DEFAULT_CHAT_MODEL", "qwen3-32b")
        chat_model = self.model_factory.get_chat_model(default_chat_model)
        
        # Hybrid chunking strategy
        node_parsers = [
            # Hierarchical chunking for document structure
            HierarchicalNodeParser.from_defaults(
                chunk_sizes=[2048, 512, 128]
            ),
            # Semantic chunking for meaning preservation
            SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=self.embedding_model
            ),
            # Sentence-based chunking as fallback
            SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50
            )
        ]
        
        # Metadata extractors for enhanced retrieval
        extractors = [
            TitleExtractor(nodes=5, llm=chat_model),
            QuestionsAnsweredExtractor(questions=3, llm=chat_model),
            SummaryExtractor(summaries=["prev", "self"], llm=chat_model),
            KeywordExtractor(keywords=10, llm=chat_model)
        ]
        
        # Create ingestion pipeline
        self.pipeline = IngestionPipeline(
            transformations=node_parsers + extractors,
            vector_store=self.vector_store
        )
    
    def ingest_documents(self, input_dir: str = "test_data") -> Dict[str, Any]:
        try:
            # Get Docling processor
            docling_processor = get_docling_processor()
            
            # Find all supported files in directory
            import os
            from pathlib import Path
            
            input_path = Path(input_dir)
            if not input_path.exists():
                logger.error(f"Input directory does not exist: {input_dir}")
                return {"status": "error", "message": f"Directory not found: {input_dir}"}
            
            documents = []
            supported_files = []
            
            # Collect all supported files
            for file_path in input_path.rglob("*"):
                if file_path.is_file() and is_document_supported(file_path):
                    supported_files.append(file_path)
            
            if not supported_files:
                logger.warning(f"No supported documents found in {input_dir}")
                return {"status": "warning", "message": "No supported documents found", "count": 0}
            
            logger.info(f"Found {len(supported_files)} supported files to process")
            
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
                            metadata=metadata
                        )
                        documents.append(doc)
                        logger.info(f"Successfully processed {file_path.name}")
                    else:
                        logger.warning(f"Failed to extract content from {file_path.name}")
                        
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
                    continue
            
            if not documents:
                logger.warning("No documents were successfully processed")
                return {"status": "warning", "message": "No documents processed successfully", "count": 0}
            
            logger.info(f"Found {len(documents)} documents to ingest")
            
            # Process documents through pipeline
            nodes = self.pipeline.run(documents=documents)
            
            # Create storage context and index
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            self.index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=self.embedding_model
            )
            
            logger.info(f"Successfully ingested {len(nodes)} nodes from {len(documents)} documents")
            
            return {
                "status": "success",
                "documents_count": len(documents),
                "nodes_count": len(nodes),
                "collection": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_retriever(self, similarity_top_k: int = 5, **kwargs):
        if not hasattr(self, 'index'):
            # Try to load existing index
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=storage_context,
                embed_model=self.embedding_model
            )
        
        return self.index.as_retriever(
            similarity_top_k=similarity_top_k,
            **kwargs
        )
    
    def get_query_engine(self, similarity_top_k: int = 5, **kwargs):
        if not hasattr(self, 'index'):
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=storage_context,
                embed_model=self.embedding_model
            )
        
        return self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            **kwargs
        )
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            retriever = self.get_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query)
            
            results = []
            for node in nodes:
                results.append({
                    "content": node.text,
                    "score": node.score,
                    "metadata": node.metadata,
                    "node_id": node.node_id
                })
            
            return results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.name,
                "vector_size": info.config.params.vectors.size,
                "distance_metric": info.config.params.vectors.distance,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}

# Global instance
_qdrant_store = None

def get_qdrant_store(qdrant_url: str = None) -> HybridQdrantStore:
    global _qdrant_store
    if _qdrant_store is None:
        # Use environment variable if no URL provided
        if qdrant_url is None:
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        _qdrant_store = HybridQdrantStore(qdrant_url=qdrant_url)
    return _qdrant_store