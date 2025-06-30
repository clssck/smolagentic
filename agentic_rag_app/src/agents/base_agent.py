"""
Base Agent Class

Provides the foundation for all agents in the agentic RAG system.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from smolagents import LiteLLMModel, ToolCallingAgent


class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(
        self,
        name: str,
        model_id: str,
        tools: List[Any] = None,
        max_steps: int = 6,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        description: str = "",
        **kwargs
    ):
        """
        Initialize the base agent
        
        Args:
            name: Agent name
            model_id: LLM model identifier
            tools: List of tools available to the agent
            max_steps: Maximum reasoning steps
            temperature: Model temperature
            max_tokens: Maximum tokens per response
            description: Agent description
        """
        self.name = name
        self.model_id = model_id
        self.tools = tools or []
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.description = description
        
        # Initialize model
        self.model = LiteLLMModel(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Initialize agent
        self.agent = self._create_agent()
        
        # Setup logging
        self.logger = logging.getLogger(f"agent.{name}")
        
    def _create_agent(self) -> ToolCallingAgent:
        """Create the underlying agent instance"""
        return ToolCallingAgent(
            tools=self.tools,
            model=self.model,
            max_steps=self.max_steps,
            planning_interval=None
        )
    
    @abstractmethod
    def can_handle(self, query: str, context: Dict[str, Any] = None) -> bool:
        """
        Determine if this agent can handle the given query
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            True if agent can handle the query
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        pass
    
    def run(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the agent with the given query
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            Agent response with metadata
        """
        start_time = time.time()
        context = context or {}
        
        try:
            self.logger.info(f"Processing query: {query[:100]}...")
            
            # Prepare the full prompt
            system_prompt = self.get_system_prompt()
            full_query = f"{system_prompt}\n\nUser Query: {query}"
            
            # Run the agent
            response = self.agent.run(full_query)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            result = {
                "response": response,
                "agent_name": self.name,
                "execution_time": execution_time,
                "model_used": self.model_id,
                "tools_used": [tool.__class__.__name__ for tool in self.tools],
                "success": True,
                "error": None
            }
            
            self.logger.info(f"Query processed successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            self.logger.error(f"Error processing query: {error_msg}")
            
            return {
                "response": f"Error: {error_msg}",
                "agent_name": self.name,
                "execution_time": execution_time,
                "model_used": self.model_id,
                "tools_used": [tool.__class__.__name__ for tool in self.tools],
                "success": False,
                "error": error_msg
            }
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": self.name,
            "description": self.description,
            "model_id": self.model_id,
            "tools": [tool.__class__.__name__ for tool in self.tools],
            "max_steps": self.max_steps,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.model_id}')"