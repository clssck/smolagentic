import logging
import os
from typing import Dict, Any, List, Optional, AsyncGenerator
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.base.response.schema import RESPONSE_TYPE

from models.factory import get_model_factory
from vector_store.qdrant_client import get_qdrant_store
from utils.config_loader import get_config_loader, ModelType
from tools.reasoning_tool import get_reasoning_tools

logger = logging.getLogger(__name__)

class AgenticRAG:
    def __init__(self, 
                 chat_model_name: Optional[str] = None,
                 embedding_model_name: Optional[str] = None,
                 qdrant_url: str = "http://localhost:6333",
                 enable_reasoning: bool = True):
        
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
        
        # Setup RAG components
        self._setup_rag_pipeline()
        self._setup_agent()
    
    def _setup_rag_pipeline(self):
        # Create query engine with enhanced retrieval
        self.query_engine = self.qdrant_store.get_query_engine(
            similarity_top_k=5,
            llm=self.chat_model,
            embed_model=self.embedding_model,
            response_mode="tree_summarize"
        )
        
        # Create tools for the agent
        self.rag_tool = QueryEngineTool(
            query_engine=self.query_engine,
            metadata=ToolMetadata(
                name="knowledge_search",
                description="Search through the knowledge base to find relevant information. Use this tool when you need to find specific facts, data, or detailed information from the documents."
            )
        )
        
        # Initialize tools list
        self.tools = [self.rag_tool]
        
        # Add reasoning tools if enabled
        if self.enable_reasoning:
            reasoning_tools = get_reasoning_tools("qwen3-32b-reasoning")
            self.tools.extend(reasoning_tools)
            logger.info("Reasoning tools enabled")
    
    def _setup_agent(self):
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
- Always use the knowledge_search tool when you need specific information
- Cite sources when providing factual information
- Be honest about limitations and uncertainty
- Provide context and explanations, not just raw facts
- If information isn't in the knowledge base, clearly state that{'''
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
            max_iterations=10
        )
    
    def chat(self, message: str, stream: bool = False) -> str:
        try:
            if stream:
                return self._stream_chat(message)
            else:
                response = self.agent.chat(message)
                return str(response)
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _stream_chat(self, message: str) -> AsyncGenerator[str, None]:
        try:
            streaming_response = self.agent.stream_chat(message)
            for token in streaming_response.response_gen:
                yield token
        except Exception as e:
            logger.error(f"Error in streaming chat: {e}")
            yield f"Error: {str(e)}"
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        history = []
        for message in self.memory.get_all():
            history.append({
                "role": message.role.value,
                "content": message.content
            })
        return history
    
    def clear_history(self):
        self.memory.reset()
        logger.info("Chat history cleared")
    
    def search_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.qdrant_store.search(query, top_k)
    
    def switch_chat_model(self, model_name: str):
        try:
            self.chat_model = self.model_factory.get_chat_model(model_name)
            self._setup_rag_pipeline()
            self._setup_agent()
            logger.info(f"Switched to chat model: {model_name}")
        except Exception as e:
            logger.error(f"Error switching chat model: {e}")
            raise
    
    def switch_embedding_model(self, model_name: str):
        try:
            self.embedding_model = self.model_factory.get_embedding_model(model_name)
            # Note: Changing embedding model requires re-indexing documents
            logger.warning(f"Switched embedding model to {model_name}. Consider re-indexing documents for consistency.")
        except Exception as e:
            logger.error(f"Error switching embedding model: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        chat_models = self.model_factory.list_available_models(ModelType.CHAT)
        embed_models = self.model_factory.list_available_models(ModelType.EMBEDDING)
        qdrant_info = self.qdrant_store.get_collection_info()
        
        return {
            "current_chat_model": getattr(self.chat_model, 'model', 'unknown'),
            "current_embedding_model": getattr(self.embedding_model, 'model_name', 'unknown'),
            "available_chat_models": chat_models,
            "available_embedding_models": embed_models,
            "qdrant_collection": qdrant_info,
            "chat_history_length": len(self.memory.get_all())
        }

# Global instance
_agentic_rag = None

def get_agentic_rag(qdrant_url: str = "http://localhost:6333", enable_reasoning: bool = True) -> AgenticRAG:
    global _agentic_rag
    if _agentic_rag is None:
        _agentic_rag = AgenticRAG(qdrant_url=qdrant_url, enable_reasoning=enable_reasoning)
    return _agentic_rag