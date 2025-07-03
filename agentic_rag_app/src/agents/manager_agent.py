"""
Manager Agent

Ultra-fast coordination agent that routes queries to specialized agents.
"""

import re
from typing import Dict, Any, List
from smolagents import CodeAgent, LiteLLMModel
from .base_agent import BaseAgent


class ManagerAgent(BaseAgent):
    """Manager agent for routing and coordination"""

    def __init__(self, specialized_agents: Dict[str, BaseAgent] = None, **kwargs):
        """
        Initialize manager agent

        Args:
            specialized_agents: Dictionary of specialized agents
        """
        self.specialized_agents = specialized_agents or {}

        # Set defaults
        config = {
            "name": "manager",
            "model_id": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
            "tools": [],
            "max_steps": 3,
            "temperature": 0.1,
            "max_tokens": 600,
            "description": "Ultra-fast coordination agent for routing queries",
        }

        # Update with provided kwargs
        config.update(kwargs)

        super().__init__(**config)

        # Manager uses CodeAgent for speed
        self.manager_model = LiteLLMModel(
            model_id=config["model_id"], temperature=0.1, max_tokens=600
        )
        self.manager_agent = CodeAgent(tools=[], model=self.manager_model, max_steps=3)

    def add_agent(self, name: str, agent: BaseAgent):
        """Add a specialized agent"""
        self.specialized_agents[name] = agent
        self.logger.info(f"Added specialized agent: {name}")

    def can_handle(self, query: str, context: Dict[str, Any] = None) -> bool:
        """Manager can handle any query by routing it"""
        return True

    def get_system_prompt(self) -> str:
        """Get manager system prompt"""
        agent_list = "\n".join(
            [
                f"- {name}: {agent.description}"
                for name, agent in self.specialized_agents.items()
            ]
        )

        return f"""You are an intelligent Manager Agent that coordinates with specialized agents to provide comprehensive assistance.

Available Specialized Agents:
{agent_list}

Your role is to:
1. Understand the nature and requirements of user queries
2. Determine which specialized agent would be most effective
3. Coordinate responses to provide the best possible assistance
4. Handle simple queries directly when appropriate

Consider the capabilities of each agent and the specific needs of the query. Think about whether the user is looking for existing knowledge, current information, technical assistance, or simple interaction.

Respond with ONLY the agent name (e.g., "research_agent") or "direct" for direct handling."""

    def _route_query(self, query: str) -> str:
        """Route query to appropriate agent"""
        try:
            routing_prompt = f"""Route this query to the appropriate agent:

Query: "{query}"

Respond with ONLY the agent name or "direct"."""

            response = self.manager_agent.run(routing_prompt)

            # Extract agent name from response
            agent_name = self._extract_agent_name(str(response))

            self.logger.info(f"Routed query to: {agent_name}")
            return agent_name

        except Exception as e:
            self.logger.error(f"Routing error: {e}")
            return "direct"

    def _extract_agent_name(self, response: str) -> str:
        """Extract agent name from routing response"""
        response = response.lower().strip()

        # Look for specific agent names
        for agent_name in self.specialized_agents.keys():
            if agent_name in response:
                return agent_name

        # Check for direct handling keywords
        if any(word in response for word in ["direct", "handle", "simple", "basic"]):
            return "direct"

        # Default fallback
        return "direct"

    def _handle_direct(self, query: str) -> str:
        """Handle simple queries directly"""
        direct_prompt = f"""You are a helpful assistant. Provide a concise, direct answer to this query:

Query: {query}

Keep your response brief and helpful."""

        try:
            response = self.manager_agent.run(direct_prompt)
            return str(response)
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"

    def run(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Route and execute query

        Args:
            query: User query
            context: Additional context

        Returns:
            Response from appropriate agent
        """
        import time

        start_time = time.time()

        try:
            # Route the query
            target_agent = self._route_query(query)

            if target_agent == "direct":
                # Handle directly
                response = self._handle_direct(query)
                execution_time = time.time() - start_time

                return {
                    "response": response,
                    "agent_name": "manager (direct)",
                    "execution_time": execution_time,
                    "model_used": self.model_id,
                    "tools_used": [],
                    "success": True,
                    "error": None,
                    "routing_decision": "direct",
                }

            elif target_agent in self.specialized_agents:
                # Delegate to specialized agent
                specialized_agent = self.specialized_agents[target_agent]
                result = specialized_agent.run(query, context)

                # Add routing information
                result["routing_decision"] = target_agent
                result["routed_by"] = "manager"

                return result

            else:
                # Fallback to direct handling
                response = self._handle_direct(query)
                execution_time = time.time() - start_time

                return {
                    "response": response,
                    "agent_name": "manager (fallback)",
                    "execution_time": execution_time,
                    "model_used": self.model_id,
                    "tools_used": [],
                    "success": True,
                    "error": None,
                    "routing_decision": "fallback",
                }

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)

            self.logger.error(f"Manager error: {error_msg}")

            return {
                "response": f"Manager error: {error_msg}",
                "agent_name": "manager",
                "execution_time": execution_time,
                "model_used": self.model_id,
                "tools_used": [],
                "success": False,
                "error": error_msg,
                "routing_decision": "error",
            }

    def get_available_agents(self) -> List[str]:
        """Get list of available specialized agents"""
        return list(self.specialized_agents.keys())

    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """Get information about a specific agent"""
        if agent_name in self.specialized_agents:
            return self.specialized_agents[agent_name].get_info()
        return None
