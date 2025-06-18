"""RAG Agent module for intelligent document question answering.

This module provides the AgenticRAG class that combines retrieval-augmented
generation with reasoning capabilities for enhanced document interaction.
"""

from collections.abc import AsyncGenerator
import logging
import os
from typing import Any

from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from models.factory import get_model_factory
from tools.reasoning_tool import get_reasoning_tools
from utils.config_loader import ModelType, get_config_loader
from vector_store.qdrant_client import get_qdrant_store

logger = logging.getLogger(__name__)

class AgenticRAG:
    """Intelligent RAG assistant with optional reasoning capabilities.

    Combines document retrieval with AI chat models to provide accurate,
    context-aware responses. Optionally includes advanced reasoning tools
    for complex analytical tasks.
    """
    def __init__(self,
                 chat_model_name: str | None = None,
                 embedding_model_name: str | None = None,
                 qdrant_url: str | None = None,
                 enable_reasoning: bool = True) -> None:
        """Initialize the AgenticRAG system with specified models and configuration.

        Args:
            chat_model_name: Name of the chat model to use. If None, uses default.
            embedding_model_name: Name of the embedding model. If None, uses default.
            qdrant_url: URL of the Qdrant vector database instance.
            enable_reasoning: Whether to enable advanced reasoning capabilities.
        """
        self.config = get_config_loader()
        self.model_factory = get_model_factory()
        self.qdrant_store = get_qdrant_store(qdrant_url)
        self.enable_reasoning = enable_reasoning

        # Use defaults from config if not specified
        if chat_model_name is None:
            chat_model_name = os.getenv("DEFAULT_CHAT_MODEL", "qwen3-32b")
        if embedding_model_name is None:
            embedding_model_name = os.getenv("DEFAULT_EMBEDDING_MODEL", "qwen3-embed")

        # Initialize models
        self.chat_model = self.model_factory.get_chat_model(chat_model_name)
        self.embedding_model = self.model_factory.get_embedding_model(embedding_model_name)
        
        # Set global LlamaIndex settings for this instance
        Settings.llm = self.chat_model
        Settings.embed_model = self.embedding_model

        # Setup RAG components
        self._setup_rag_pipeline()
        self._setup_agent()

    def _setup_rag_pipeline(self) -> None:
        """Set up the RAG pipeline with query engine and tools."""
        # Create storage context with our Qdrant vector store
        storage_context = StorageContext.from_defaults(
            vector_store=self.qdrant_store.vector_store
        )
        
        # Create VectorStoreIndex using the clean LlamaIndex pattern
        vector_index = VectorStoreIndex(
            nodes=[],  # Empty nodes since we're loading from existing vector store
            storage_context=storage_context,
            embed_model=self.embedding_model
        )
        
        # Create query engine using the simple approach
        self.query_engine = vector_index.as_query_engine(
            similarity_top_k=10,
            response_mode="compact",
            llm=self.chat_model
        )

        # Create query engine tool
        self.rag_tool = QueryEngineTool(
            query_engine=self.query_engine,
            metadata=ToolMetadata(
                name="knowledge_search",
                description="Search through the knowledge base to find relevant information. Use this tool when you need to find specific facts, data, or detailed information from the documents.",
            ),
        )

        # Initialize tools list
        self.tools = [self.rag_tool]

        # Add reasoning tools if enabled
        if self.enable_reasoning:
            reasoning_tools = get_reasoning_tools("qwen3-32b-reasoning")
            self.tools.extend(reasoning_tools)
            logger.info("Reasoning tools enabled")

    def _setup_agent(self) -> None:
        """Set up the ReAct agent with system prompt and tools."""
        # Create memory buffer for conversation history
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

        # System prompt for the agent
        reasoning_capabilities = """
- Use deep_reasoning tool for complex problems requiring step-by-step analysis
- Use compare_approaches tool when evaluating multiple solutions or methods
- Show your thinking process for mathematical, logical, or analytical problems""" if self.enable_reasoning else ""

        system_prompt = f"""You are an intelligent RAG assistant with access to a knowledge base{' and advanced reasoning capabilities' if self.enable_reasoning else ''}.

Your capabilities:
- Search through documents using the knowledge_search tool
- Provide accurate, detailed answers based on the retrieved information
- Engage in natural conversation while staying grounded in facts
- Ask clarifying questions when needed
- Synthesize information from multiple sources{reasoning_capabilities}

Guidelines:
- Use the knowledge_search tool ONCE per query to find information
- If the search doesn't return useful results, don't repeat the same search
- Cite sources when providing factual information  
- Be honest about limitations and uncertainty
- Provide context and explanations, not just raw facts
- If information isn't in the knowledge base, clearly state that and stop searching{'''
- For complex problems requiring analysis, use the deep_reasoning tool
- When comparing options or approaches, use the compare_approaches tool
- Show your reasoning process for mathematical or logical problems''' if self.enable_reasoning else ''}

Remember: Your goal is to be helpful, accurate, and trustworthy."""

        # Create ReAct agent
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.chat_model,
            memory=self.memory,
            system_prompt=system_prompt,
            verbose=True,
            max_iterations=5,  # Reduced to prevent infinite loops
        )

    def chat(self, message: str, stream: bool = False) -> str:
        """Process a chat message and return the response.

        Args:
            message: The user's message or question.
            stream: Whether to stream the response or return complete text.

        Returns:
            The agent's response as a string.
        """
        try:
            if stream:
                return self._stream_chat(message)
            response = self.agent.chat(message)
            return str(response)
        except Exception as e:
            logger.exception("Error in chat")
            return f"I apologize, but I encountered an error: {e!s}"

    def _stream_chat(self, message: str) -> AsyncGenerator[str, None]:
        try:
            streaming_response = self.agent.stream_chat(message)
            yield from streaming_response.response_gen
        except Exception as e:
            logger.exception("Error in streaming chat")
            yield f"Error: {e!s}"

    def get_chat_history(self) -> list[dict[str, str]]:
        """Get the conversation history.

        Returns:
            List of message dictionaries with 'role' and 'content' keys.
        """
        return [
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in self.memory.get_all()
        ]

    def clear_history(self) -> None:
        """Clear the conversation history and reset the memory buffer."""
        self.memory.reset()
        logger.info("Chat history cleared")

    def search_knowledge_base(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search the knowledge base for relevant documents.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            List of search results with content, scores, and metadata.
        """
        return self.qdrant_store.search(query, top_k)

    def switch_chat_model(self, model_name: str) -> None:
        """Switch to a different chat model and reinitialize the agent.

        Args:
            model_name: Name of the new chat model to use.

        Raises:
            Exception: If the model switch fails.
        """
        try:
            self.chat_model = self.model_factory.get_chat_model(model_name)
            self._setup_rag_pipeline()
            self._setup_agent()
            logger.info("Switched to chat model: %s", model_name)
        except Exception:
            logger.exception("Error switching chat model")
            raise

    def switch_embedding_model(self, model_name: str) -> None:
        """Switch to a different embedding model.

        Args:
            model_name: Name of the new embedding model to use.

        Raises:
            Exception: If the model switch fails.

        Note:
            Changing the embedding model may require re-indexing documents
            for optimal performance.
        """
        try:
            self.embedding_model = self.model_factory.get_embedding_model(model_name)
            # Note: Changing embedding model requires re-indexing documents
            logger.warning("Switched embedding model to %s. Consider re-indexing documents for consistency.", model_name)
        except Exception:
            logger.exception("Error switching embedding model")
            raise

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status information.

        Returns:
            Dictionary containing current models, available models, and system state.
        """
        chat_models = self.model_factory.list_available_models(ModelType.CHAT)
        embed_models = self.model_factory.list_available_models(ModelType.EMBEDDING)
        qdrant_info = self.qdrant_store.get_collection_info()

        return {
            "current_chat_model": getattr(self.chat_model, "model", "unknown"),
            "current_embedding_model": getattr(self.embedding_model, "model_name", "unknown"),
            "available_chat_models": chat_models,
            "available_embedding_models": embed_models,
            "qdrant_collection": qdrant_info,
            "chat_history_length": len(self.memory.get_all()),
        }

# Global instance
_agentic_rag = None

def get_agentic_rag(qdrant_url: str | None = None, enable_reasoning: bool = True) -> AgenticRAG:
    """Get or create a singleton AgenticRAG instance.

    Args:
        qdrant_url: URL of the Qdrant vector database instance.
        enable_reasoning: Whether to enable advanced reasoning capabilities.

    Returns:
        The AgenticRAG singleton instance.
    """
    global _agentic_rag
    if _agentic_rag is None:
        _agentic_rag = AgenticRAG(qdrant_url=qdrant_url, enable_reasoning=enable_reasoning)
    return _agentic_rag
