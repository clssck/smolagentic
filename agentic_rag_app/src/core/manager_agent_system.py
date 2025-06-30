#!/usr/bin/env python3
"""
Manager Agent System - Clean Integration
Based on smolagents examples - simple, fast, effective
"""

import json
import sys
from pathlib import Path
from typing import Any

try:
    from smolagents import (
        CodeAgent,
        LiteLLMModel,
        Tool,
        ToolCallingAgent,
        VisitWebpageTool,
        WebSearchTool,
        tool,
    )

    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False
    print("⚠️  smolagents not available, using fallback mode")

# Import existing components
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.utils.config import Config
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("⚠️  Utils config not available")

try:
    from vector_store.qdrant_client import QdrantVectorStore
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    print("⚠️  Vector store not available - will use fallback mode")


class RAGTool(Tool):
    """RAG tool that integrates with existing vector store"""

    name = "knowledge_search"
    description = "Search knowledge base for technical information and documentation"
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query for technical knowledge",
        }
    }
    output_type = "string"

    def __init__(self, vector_store=None, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store

    def forward(self, query: str) -> str:
        """Search the knowledge base"""
        try:
            if self.vector_store:
                # Use actual vector store
                results = self.vector_store.search(query, top_k=5)
                if results:
                    context = "\n\n".join(
                        [
                            f"Document {i + 1}: {result.get('content', result.get('text', str(result)))}"  # noqa: E501
                            for i, result in enumerate(results)
                        ]
                    )
                    return f"Knowledge base search results for '{query}':\n\n{context}"
                else:
                    return (
                        f"No relevant documents found in knowledge base for '{query}'"
                    )
            else:
                # Fallback mode
                return f"Knowledge base search for '{query}': [Vector store not available - using fallback]"
        except Exception as e:
            return f"Error searching knowledge base for '{query}': {e!s}"


# Use @tool decorator for cleaner tool definitions
@tool
def simple_chat(message: str) -> str:
    """Handle simple questions, greetings, basic math, and direct responses

    Args:
        message: Simple message or question to respond to
    """
    msg_lower = message.lower()

    # Greetings
    if any(
        greeting in msg_lower
        for greeting in ["hello", "hi", "hey", "good morning", "good afternoon"]
    ):
        return "Hello! I'm your AI assistant. I can help with research, knowledge base queries, or general chat. How can I assist you today?"

    # Thanks
    elif any(thanks in msg_lower for thanks in ["thank", "thanks", "appreciate"]):
        return "You're very welcome! I'm happy to help. Feel free to ask if you have any other questions!"

    # Simple math
    elif any(op in message for op in ["+", "-", "*", "/", "="]):
        try:
            # Extract mathematical expression
            import re

            math_expr = re.search(r"[\d\s\+\-\*\/\(\)\.]+", message)
            if math_expr:
                expr = math_expr.group().replace(" ", "")
                # Safe evaluation for basic math
                result = eval(expr)
                return f"The answer is {result}"
        except:
            pass
        return "I can help with simple math like 2+2, 5*3, or (10+5)/3. What calculation would you like me to do?"

    # Yes/No responses
    elif msg_lower.strip() in ["yes", "no", "ok", "okay", "sure"]:
        return "Got it! Is there anything specific you'd like help with?"

    # Default
    else:
        return "I understand. Is this something I should search for in my knowledge base, or would you like me to research current information about it?"


class ManagerAgentSystem:
    """
    Clean Manager Agent System
    Replaces complex routing with natural agent delegation
    """

    def __init__(self, config_path: str | None = None):
        self.config = self._load_config(config_path)
        self.vector_store = None
        self.manager_agent = None

        # Initialize components
        self._setup_vector_store()
        self._setup_agents()

    def _load_config(self, config_path: str | None) -> dict[str, Any]:
        """Load configuration"""
        default_config = {
            "models": {
                "manager": {
                    "name": "groq/qwen/qwen3-32b",  # Ultra-fast coordination via Groq
                    "temperature": 0.1,
                    "max_tokens": 800,
                    "reasoning": "Blazing fast coordination and delegation",
                },
                "research_agent": {
                    "name": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "reasoning": "Optimized for web search and research tasks",
                    "description": "Agent specialized in web search and current information research",
                },
                "rag_agent": {
                    "name": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
                    "temperature": 0.1,
                    "max_tokens": 1200,
                    "reasoning": "Optimized for knowledge base search and reasoning",
                    "description": "Agent specialized in knowledge base search and complex reasoning",
                },
                "simple_agent": {
                    "name": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
                    "temperature": 0.1,
                    "max_tokens": 600,
                    "reasoning": "Optimized for fast simple responses",
                    "description": "Agent specialized in simple chat and quick responses",
                },
            },
            "agents": {
                "research": {"enabled": True, "tools": ["web_search", "visit_webpage"]},
                "rag": {"enabled": True, "tools": ["knowledge_search"]},
                "simple": {"enabled": True, "tools": ["simple_chat"]},
            },
            "vector_store": {"enabled": True, "type": "qdrant"},
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                self._deep_merge(default_config, user_config)
            except Exception as e:
                print(f"⚠️  Failed to load config: {e}, using defaults")

        return default_config

    def _deep_merge(self, base: dict, update: dict) -> None:
        """Deep merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _setup_vector_store(self):
        """Initialize vector store if available"""
        try:
            if self.config["vector_store"]["enabled"] and VECTOR_STORE_AVAILABLE and UTILS_AVAILABLE:
                # Try to use existing vector store
                config = Config()
                self.vector_store = QdrantVectorStore(
                    collection_name="documents", config=config
                )
                print("✅ Vector store connected")
            else:
                self.vector_store = None
                if not VECTOR_STORE_AVAILABLE:
                    print("⚠️  Vector store components not available - using fallback mode")
                elif not UTILS_AVAILABLE:
                    print("⚠️  Config utilities not available - using fallback mode")
                else:
                    print("⚠️  Vector store disabled in configuration")
        except Exception as e:
            print(f"⚠️  Vector store setup failed: {e}")
            self.vector_store = None

    def _setup_agents(self):
        """Setup manager and specialized agents"""
        if not SMOLAGENTS_AVAILABLE:
            print("❌ smolagents not available - cannot create agents")
            return

        try:
            # Create models for each agent type
            manager_model_config = self.config["models"]["manager"]
            research_model_config = self.config["models"]["research_agent"]
            rag_model_config = self.config["models"]["rag_agent"]

            manager_model = LiteLLMModel(
                model_id=manager_model_config["name"],
                temperature=manager_model_config["temperature"],
                max_tokens=manager_model_config["max_tokens"],
            )

            research_model = LiteLLMModel(
                model_id=research_model_config["name"],
                temperature=research_model_config["temperature"],
                max_tokens=research_model_config["max_tokens"],
            )

            rag_model = LiteLLMModel(
                model_id=rag_model_config["name"],
                temperature=rag_model_config["temperature"],
                max_tokens=rag_model_config["max_tokens"],
            )

            # Create specialized agents
            specialized_agents = []

            # Research Agent (optimized Mistral Small 3.2)
            if self.config["agents"]["research"]["enabled"]:
                research_agent = ToolCallingAgent(
                    tools=[WebSearchTool(), VisitWebpageTool()],
                    model=research_model,
                    name="research_agent",
                    description=research_model_config.get(
                        "description", "Agent specialized in web search and research"
                    ),
                    max_steps=8,
                    verbosity_level=1,
                    planning_interval=3,
                    provide_run_summary=True,
                )
                specialized_agents.append(research_agent)
                print(
                    f"✅ Research agent created ({research_model_config['name'].split('/')[-1]} - {research_model_config['reasoning']})"
                )

            # RAG Agent (optimized Mistral Small 3.2)
            if self.config["agents"]["rag"]["enabled"]:
                rag_tool = RAGTool(vector_store=self.vector_store)
                rag_agent = ToolCallingAgent(
                    tools=[rag_tool],
                    model=rag_model,
                    name="rag_agent",
                    description=rag_model_config.get(
                        "description", "Agent specialized in knowledge base search"
                    ),
                    max_steps=6,
                    verbosity_level=1,
                    planning_interval=2,
                    provide_run_summary=True,
                )
                specialized_agents.append(rag_agent)
                print(
                    f"✅ RAG agent created ({rag_model_config['name'].split('/')[-1]} - {rag_model_config['reasoning']})"
                )

            # Manager Agent (ultra-fast Groq Qwen3-32B)
            simple_tools = []
            if self.config["agents"]["simple"]["enabled"]:
                simple_tools.append(simple_chat)  # Use the @tool decorated function

            self.manager_agent = CodeAgent(
                tools=simple_tools,
                model=manager_model,  # Ultra-fast Groq Qwen3-32B
                managed_agents=specialized_agents,
                max_steps=4,  # Reduced for faster coordination
                verbosity_level=1,
                planning_interval=2,
                stream_outputs=True,
                return_full_result=True,
            )

            print("✅ Manager agent system initialized")
            print(f"🚀 Manager: Groq Qwen3-32B ({manager_model_config['reasoning']})")
            print(
                f"⚡ Speed optimized: Manager {manager_model_config['max_tokens']} tokens, Agents up to {rag_model_config['max_tokens']} tokens"
            )

        except Exception as e:
            print(f"❌ Failed to setup agents: {e}")
            raise

    def run_query(self, query: str, **kwargs) -> str:
        """
        Run a query through the manager agent system

        Args:
            query: The user query
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Response string
        """
        if not self.manager_agent:
            return "❌ Manager agent not available. Please check your configuration."

        try:
            print(f"🤖 Processing: {query}")

            # Run through manager agent
            result = self.manager_agent.run(query)

            # Extract response from result
            if hasattr(result, "content"):
                response = result.content
            elif isinstance(result, str):
                response = result
            else:
                response = str(result)

            return response

        except Exception as e:
            error_msg = f"Error processing query: {e!s}"
            print(f"❌ {error_msg}")
            return error_msg

    def get_status(self) -> dict[str, Any]:
        """Get system status"""
        return {
            "manager_agent": self.manager_agent is not None,
            "vector_store": self.vector_store is not None,
            "smolagents_available": SMOLAGENTS_AVAILABLE,
            "config": self.config,
        }

    def list_available_components(self) -> dict[str, list[str]]:
        """List available components for compatibility"""
        return {
            "agents": ["manager_agent", "research_agent", "rag_agent"],
            "models": {
                "manager": self.config["models"]["manager"]["name"],
                "research": self.config["models"]["research_agent"]["name"],
                "rag": self.config["models"]["rag_agent"]["name"],
            },
            "tools": ["web_search", "knowledge_search", "simple_chat"],
        }


def create_manager_system(config_path: str | None = None) -> ManagerAgentSystem:
    """Create a manager agent system"""
    return ManagerAgentSystem(config_path)


def test_manager_system():
    """Test the manager agent system"""
    print("🚀 MANAGER AGENT SYSTEM TEST")
    print("=" * 50)

    try:
        system = create_manager_system()

        test_queries = [
            "hello there",
            "what is 2+2?",
            "explain Python functions",
            "latest AI developments",
            "search for UFDF information",
        ]

        for query in test_queries:
            print(f"\n📝 Query: '{query}'")
            print("-" * 30)

            try:
                response = system.run_query(query)
                print(f"✅ Response: {response[:200]}...")
            except Exception as e:
                print(f"❌ Error: {e}")

        # Show status
        print("\n📊 System Status:")
        status = system.get_status()
        for key, value in status.items():
            if key != "config":  # Skip detailed config
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_manager_system()
