"""
Agent Factory

Factory class for creating and managing agents in the agentic RAG system.
"""

import time
from typing import Dict, Any, List, Optional, Type
from .base_agent import BaseAgent
from .manager_agent import ManagerAgent
from .research_agent import ResearchAgent
from .rag_agent import RAGAgent
from .code_agent import CodeAgent
from .simple_agent import SimpleAgent
from .vision_agent import VisionAgent


class AgentFactory:
    """Factory for creating and managing agents"""

    AGENT_TYPES = {
        "manager": ManagerAgent,
        "research": ResearchAgent,
        "rag": RAGAgent,
        "code": CodeAgent,
        "simple": SimpleAgent,
        "vision": VisionAgent,
    }

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize agent factory

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.created_agents: Dict[str, BaseAgent] = {}
        self.agent_registry: Dict[str, Dict[str, Any]] = {}

    def create_agent(self, agent_type: str, name: str = None, **kwargs) -> BaseAgent:
        """
        Create an agent of the specified type

        Args:
            agent_type: Type of agent to create
            name: Optional custom name for the agent
            **kwargs: Additional arguments for agent initialization

        Returns:
            Created agent instance
        """
        if agent_type not in self.AGENT_TYPES:
            raise ValueError(
                f"Unknown agent type: {agent_type}. Available types: {list(self.AGENT_TYPES.keys())}"
            )

        agent_class = self.AGENT_TYPES[agent_type]

        # Use provided name or default to agent type
        agent_name = name or f"{agent_type}_agent"

        # Merge with configuration
        agent_config = self._get_agent_config(agent_type)
        agent_config.update(kwargs)

        # Create agent
        agent = agent_class(**agent_config)

        # Register agent
        self.created_agents[agent_name] = agent
        self.agent_registry[agent_name] = {
            "type": agent_type,
            "class": agent_class.__name__,
            "config": agent_config,
            "created_at": self._get_timestamp(),
        }

        return agent

    def create_manager_system(
        self, specialized_agents: List[str] = None, vector_store=None, **kwargs
    ) -> ManagerAgent:
        """
        Create a complete manager system with specialized agents

        Args:
            specialized_agents: List of agent types to create
            vector_store: Vector store for RAG agent
            **kwargs: Additional configuration

        Returns:
            Manager agent with specialized agents
        """
        if specialized_agents is None:
            specialized_agents = ["research", "rag", "code", "simple", "vision"]

        # Create specialized agents
        agents = {}

        for agent_type in specialized_agents:
            if agent_type == "rag" and vector_store:
                # RAG agent needs vector store
                agent = self.create_agent(agent_type, vector_store=vector_store)
            else:
                agent = self.create_agent(agent_type)

            agents[agent_type] = agent

        # Create manager agent
        manager_config = self._get_agent_config("manager")
        manager_config.update(kwargs)
        manager_config["specialized_agents"] = agents

        manager = ManagerAgent(**manager_config)

        # Register manager
        self.created_agents["manager"] = manager
        self.agent_registry["manager"] = {
            "type": "manager",
            "class": "ManagerAgent",
            "config": manager_config,
            "specialized_agents": list(agents.keys()),
            "created_at": self._get_timestamp(),
        }

        return manager

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name"""
        return self.created_agents.get(name)

    def list_agents(self) -> List[str]:
        """List all created agents"""
        return list(self.created_agents.keys())

    def get_agent_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about an agent"""
        if name in self.agent_registry:
            registry_info = self.agent_registry[name].copy()
            if name in self.created_agents:
                agent_info = self.created_agents[name].get_info()
                registry_info["runtime_info"] = agent_info
            return registry_info
        return None

    def remove_agent(self, name: str) -> bool:
        """Remove an agent"""
        removed = False
        if name in self.created_agents:
            del self.created_agents[name]
            removed = True
        if name in self.agent_registry:
            del self.agent_registry[name]
            removed = True
        return removed

    def _get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get configuration for an agent type"""
        # Base configuration
        base_config = {
            "temperature": 0.1,
            "max_tokens": 1000,
        }

        # Type-specific configurations
        type_configs = {
            "manager": {
                "model_id": "groq/qwen/qwen3-32b",
                "max_tokens": 600,
                "max_steps": 3,
            },
            "research": {
                "model_id": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
                "max_tokens": 1200,
                "max_steps": 8,
            },
            "rag": {
                "model_id": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
                "max_tokens": 1200,
                "max_steps": 6,
            },
            "code": {
                "model_id": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
                "max_tokens": 2000,
                "max_steps": 10,
            },
            "simple": {
                "model_id": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
                "max_tokens": 800,
                "max_steps": 3,
            },
            "vision": {
                "model_id": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
                "max_tokens": 1500,
                "max_steps": 6,
            },
        }

        # Merge configurations
        config = base_config.copy()
        if agent_type in type_configs:
            config.update(type_configs[agent_type])

        # Override with factory config if provided
        if self.config and agent_type in self.config:
            config.update(self.config[agent_type])

        return config

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime

        return datetime.now().isoformat()

    def get_available_agent_types(self) -> List[str]:
        """Get list of available agent types"""
        return list(self.AGENT_TYPES.keys())

    def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory statistics"""
        return {
            "total_agents_created": len(self.created_agents),
            "agent_types_used": len(
                set(info["type"] for info in self.agent_registry.values())
            ),
            "available_agent_types": self.get_available_agent_types(),
            "active_agents": list(self.created_agents.keys()),
            "registry_size": len(self.agent_registry),
        }

    def export_agent_configs(self) -> Dict[str, Any]:
        """Export all agent configurations"""
        return {
            "factory_config": self.config,
            "agent_registry": self.agent_registry,
            "available_types": self.get_available_agent_types(),
        }

    def import_agent_configs(self, config_data: Dict[str, Any]) -> bool:
        """Import agent configurations"""
        try:
            if "factory_config" in config_data:
                self.config.update(config_data["factory_config"])

            if "agent_registry" in config_data:
                # Note: Only updates registry, doesn't recreate agents
                self.agent_registry.update(config_data["agent_registry"])

            return True
        except Exception as e:
            print(f"Error importing agent configs: {e}")
            return False

    def create_from_config(self, config_file: str) -> Dict[str, BaseAgent]:
        """
        Create agents from a configuration file

        Args:
            config_file: Path to configuration file

        Returns:
            Dictionary of created agents
        """
        import json

        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            created = {}

            if "agents" in config:
                for agent_config in config["agents"]:
                    agent_type = agent_config.get("type")
                    agent_name = agent_config.get("name")
                    agent_params = agent_config.get("params", {})

                    if agent_type and agent_type in self.AGENT_TYPES:
                        agent = self.create_agent(
                            agent_type, agent_name, **agent_params
                        )
                        created[agent_name or agent_type] = agent

            return created

        except Exception as e:
            print(f"Error creating agents from config: {e}")
            return {}


# Convenience functions
def create_manager_system(
    vector_store=None, config: Dict[str, Any] = None
) -> ManagerAgent:
    """
    Convenience function to create a complete manager system

    Args:
        vector_store: Vector store for RAG functionality
        config: Optional configuration

    Returns:
        Configured manager agent
    """
    factory = AgentFactory(config)
    return factory.create_manager_system(vector_store=vector_store)


def create_agent(agent_type: str, **kwargs) -> BaseAgent:
    """
    Convenience function to create a single agent

    Args:
        agent_type: Type of agent to create
        **kwargs: Agent configuration

    Returns:
        Created agent
    """
    factory = AgentFactory()
    return factory.create_agent(agent_type, **kwargs)
