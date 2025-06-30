"""
Manager Agent System
"""

import json
import sys
from pathlib import Path
from typing import Any

from smolagents import (
    CodeAgent,
    LiteLLMModel,
    PlanningStep,
    Tool,
    ToolCallingAgent,
    VisitWebpageTool,
    WebSearchTool,
    tool,
)

try:
    from src.agents.database_agent import DatabaseAgent
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

try:
    from src.core.model_pool import get_model_by_config, shared_models
    from src.core.shared_browser import (
        SharedBrowserSession, StatefulWebSearchTool, 
        StatefulWebVisitTool, BrowserNavigationTool
    )
    SHARED_TOOLS_AVAILABLE = True
except ImportError:
    SHARED_TOOLS_AVAILABLE = False

sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.utils.config import Config
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

try:
    from vector_store.qdrant_client import QdrantVectorStore
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False


class RAGTool(Tool):
    """RAG tool with search context and query history"""

    name = "knowledge_search"
    description = "Search knowledge base for technical information and documentation with context awareness"
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
        self.search_history = []  # Store search context
        self.query_context = {}  # Related queries and results
        self.session_id = str(int(time.time()))

    def forward(self, query: str) -> str:
        """Search the knowledge base with context awareness"""
        import time
        
        try:
            # Add query to history
            self.search_history.append({
                "query": query,
                "timestamp": time.time(),
                "results_found": 0
            })
            
            # Get related context from previous searches
            related_context = self._get_related_context(query)
            
            if self.vector_store:
                # Use actual vector store
                results = self.vector_store.search(query, top_k=5)
                if results:
                    # Update search history with results count
                    self.search_history[-1]["results_found"] = len(results)
                    
                    # Store query context for future searches
                    self.query_context[query] = {
                        "results": results,
                        "timestamp": time.time(),
                        "summary": f"Found {len(results)} relevant documents"
                    }
                    
                    context = "\n\n".join(
                        [
                            f"Document {i + 1}: {result.get('content', result.get('text', str(result)))}"
                            for i, result in enumerate(results)
                        ]
                    )
                    
                    response = f"Knowledge base search results for '{query}':\n\n{context}"
                    
                    # Add related context if available
                    if related_context:
                        response = f"{related_context}\n\n{response}"
                    
                    return response
                else:
                    return f"No relevant documents found in knowledge base for '{query}'"
            else:
                # Fallback mode with context
                fallback_response = f"Knowledge base search for '{query}': [Vector store not available - using fallback]"
                if related_context:
                    fallback_response = f"{related_context}\n\n{fallback_response}"
                return fallback_response
                
        except Exception as e:
            return f"Error searching knowledge base for '{query}': {e!s}"
    
    def _get_related_context(self, query: str) -> str:
        """Get context from related previous searches"""
        if len(self.search_history) < 2:
            return ""
        
        # Find semantically related queries (simple keyword matching)
        query_words = set(query.lower().split())
        related_searches = []
        
        for search in self.search_history[-5:]:  # Last 5 searches
            if search["query"] == query:
                continue
            
            search_words = set(search["query"].lower().split())
            # Simple similarity check
            common_words = query_words.intersection(search_words)
            if len(common_words) >= 1 and len(common_words) / len(query_words) > 0.3:
                related_searches.append(search)
        
        if not related_searches:
            return ""
        
        context = "Related previous searches:\n"
        for search in related_searches:
            context += f"- '{search['query']}' ({search['results_found']} results found)\n"
        
        return context
    
    def get_search_stats(self) -> dict:
        """Get search session statistics"""
        return {
            "session_id": self.session_id,
            "total_searches": len(self.search_history),
            "unique_queries": len(self.query_context),
            "recent_queries": [s["query"] for s in self.search_history[-3:]],
            "average_results": sum(s["results_found"] for s in self.search_history) / len(self.search_history) if self.search_history else 0
        }
    
    def clear_context(self):
        """Clear search context and history"""
        self.search_history = []
        self.query_context = {}
        self.session_id = str(int(time.time()))


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
    """Manager Agent System"""

    def __init__(self, config_path: str | None = None):
        self.config = self._load_config(config_path)
        self.vector_store = None
        self.manager_agent = None
        self.database_agent = None
        self.conversation_history = []  # Simple conversation memory
        self.debug_mode = False
        
        # Shared context managers
        self.browser_session = None
        self.rag_tool_instance = None
        
        self._setup_vector_store()
        self._setup_database()
        self._setup_shared_context()
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
                print("Vector store connected")
            else:
                self.vector_store = None
                if not VECTOR_STORE_AVAILABLE:
                    print("Vector store components not available")
                elif not UTILS_AVAILABLE:
                    print("Config utilities not available")
                else:
                    print("Vector store disabled in configuration")
        except Exception as e:
            print(f"Vector store setup failed: {e}")
            self.vector_store = None
    
    def _setup_database(self):
        """Setup database agent if available"""
        try:
            if DATABASE_AVAILABLE:
                self.database_agent = DatabaseAgent()
                print("Database agent connected")
            else:
                self.database_agent = None
        except Exception as e:
            print(f"Database agent setup failed: {e}")
            self.database_agent = None

    def _setup_shared_context(self):
        """Setup shared context managers for tools"""
        try:
            if SHARED_TOOLS_AVAILABLE:
                # Create shared browser session
                self.browser_session = SharedBrowserSession()
                print("Shared browser session created")
                
                # Preload models for better performance
                shared_models.preload_models(self.config["models"])
                print("Models preloaded in shared pool")
            else:
                print("Shared tools not available")
        except Exception as e:
            print(f"Shared context setup failed: {e}")

    def _plan_interrupt_callback(self, memory_step, agent):
        """Callback to interrupt after planning for debugging"""
        if isinstance(memory_step, PlanningStep) and self.debug_mode:
            print(f"🧠 Agent Plan: {memory_step}")
            response = input("Continue with this plan? (y/n): ")
            if response.lower() != 'y':
                agent.interrupt()

    def _setup_agents(self):
        """Setup manager and specialized agents"""
        try:
            # Get models from shared pool or create new ones
            manager_model_config = self.config["models"]["manager"]
            research_model_config = self.config["models"]["research_agent"]
            rag_model_config = self.config["models"]["rag_agent"]

            if SHARED_TOOLS_AVAILABLE:
                # Use shared model pool
                manager_model = get_model_by_config("manager", manager_model_config)
                research_model = get_model_by_config("research_agent", research_model_config)
                rag_model = get_model_by_config("rag_agent", rag_model_config)
                print("Using shared model pool")
            else:
                # Fallback to individual model creation
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
                print("Using individual model instances")

            # Create specialized agents
            specialized_agents = []

            # Research Agent (optimized Mistral Small 3.2)
            if self.config["agents"]["research"]["enabled"]:
                # Use shared browser tools if available
                if SHARED_TOOLS_AVAILABLE and self.browser_session:
                    research_tools = [
                        StatefulWebSearchTool(browser_session=self.browser_session),
                        StatefulWebVisitTool(browser_session=self.browser_session),
                        BrowserNavigationTool(browser_session=self.browser_session)
                    ]
                    print("Research agent using shared browser session")
                else:
                    research_tools = [WebSearchTool(), VisitWebpageTool()]
                    print("Research agent using individual tools")
                
                research_agent = ToolCallingAgent(
                    tools=research_tools,
                    model=research_model,
                    name="research_agent",
                    description=research_model_config.get(
                        "description", "Agent specialized in web search and research"
                    ),
                    max_steps=8,
                    verbosity_level=1,
                    planning_interval=3,
                    provide_run_summary=True,
                    step_callbacks=[self._plan_interrupt_callback] if self.debug_mode else [],
                )
                specialized_agents.append(research_agent)

            # RAG Agent (optimized Mistral Small 3.2)
            if self.config["agents"]["rag"]["enabled"]:
                # Create or reuse RAG tool instance for context sharing
                if not self.rag_tool_instance:
                    self.rag_tool_instance = RAGTool(vector_store=self.vector_store)
                
                rag_agent = ToolCallingAgent(
                    tools=[self.rag_tool_instance],
                    model=rag_model,
                    name="rag_agent",
                    description=rag_model_config.get(
                        "description", "Agent specialized in knowledge base search"
                    ),
                    max_steps=6,
                    verbosity_level=1,
                    planning_interval=2,
                    provide_run_summary=True,
                    step_callbacks=[self._plan_interrupt_callback] if self.debug_mode else [],
                )
                specialized_agents.append(rag_agent)

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
                step_callbacks=[self._plan_interrupt_callback] if self.debug_mode else [],
            )

            print("Manager agent system initialized")

        except Exception as e:
            print(f"Failed to setup agents: {e}")
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
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Check for database queries first
        if self.database_agent and self._is_database_query(query):
            try:
                response = self.database_agent.query(query)
                self.conversation_history.append({"role": "assistant", "content": response})
                return response
            except Exception as e:
                error_response = f"Database query error: {e}"
                self.conversation_history.append({"role": "assistant", "content": error_response})
                return error_response
        
        if not self.manager_agent:
            return "Manager agent not available."

        try:
            # Add recent context to query if history exists
            context_query = query
            if len(self.conversation_history) > 2:  # Has previous conversation
                recent_context = self.conversation_history[-4:]  # Last 2 exchanges
                context_summary = "\n".join([f"{msg['role']}: {msg['content'][:100]}..." 
                                           for msg in recent_context])
                context_query = f"Context:\n{context_summary}\n\nCurrent query: {query}"

            # Run through manager agent
            result = self.manager_agent.run(context_query)

            # Extract response from result
            if hasattr(result, "content"):
                response = result.content
            elif isinstance(result, str):
                response = result
            else:
                response = str(result)

            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Keep history manageable (last 20 messages)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            return response

        except Exception as e:
            error_response = f"Error processing query: {e}"
            self.conversation_history.append({"role": "assistant", "content": error_response})
            return error_response
    
    def _is_database_query(self, query: str) -> bool:
        """Check if query is database-related"""
        db_keywords = [
            "sql", "database", "table", "select", "query", 
            "users", "orders", "data", "count", "total",
            "show me", "how many", "list all"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in db_keywords)

    def enable_debug_mode(self):
        """Enable debug mode with plan interruption"""
        self.debug_mode = True
        print("🐛 Debug mode enabled - will pause after agent planning")
    
    def disable_debug_mode(self):
        """Disable debug mode"""
        self.debug_mode = False
        print("🐛 Debug mode disabled")
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🧹 Conversation history cleared")
    
    def get_conversation_history(self) -> list:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def get_browser_context(self) -> dict:
        """Get browser session context and stats"""
        if self.browser_session:
            return {
                "stats": self.browser_session.get_session_stats(),
                "browse_context": self.browser_session.get_browse_context(),
                "search_context": self.browser_session.get_search_context()
            }
        return {"error": "Browser session not available"}
    
    def get_rag_context(self) -> dict:
        """Get RAG tool context and stats"""
        if self.rag_tool_instance:
            return self.rag_tool_instance.get_search_stats()
        return {"error": "RAG tool not available"}
    
    def get_model_pool_stats(self) -> dict:
        """Get shared model pool statistics"""
        if SHARED_TOOLS_AVAILABLE:
            return shared_models.get_stats()
        return {"error": "Model pool not available"}
    
    def clear_all_context(self):
        """Clear all shared context and history"""
        self.clear_conversation_history()
        
        if self.browser_session:
            self.browser_session = SharedBrowserSession()
            print("🧹 Browser session reset")
        
        if self.rag_tool_instance:
            self.rag_tool_instance.clear_context()
            print("🧹 RAG context cleared")

    def get_status(self) -> dict[str, Any]:
        """Get system status"""
        status = {
            "manager_agent": self.manager_agent is not None,
            "vector_store": self.vector_store is not None,
            "database_agent": self.database_agent is not None,
            "debug_mode": self.debug_mode,
            "conversation_history_length": len(self.conversation_history),
            "shared_tools_available": SHARED_TOOLS_AVAILABLE,
            "config": self.config,
        }
        
        # Add context manager status
        if SHARED_TOOLS_AVAILABLE:
            status["browser_session"] = self.browser_session is not None
            status["rag_tool_instance"] = self.rag_tool_instance is not None
            status["model_pool_stats"] = self.get_model_pool_stats()
        
        return status

    def list_available_components(self) -> dict[str, list[str]]:
        """List available components for compatibility"""
        return {
            "agents": ["manager_agent", "research_agent", "rag_agent"],
            "models": {
                "manager": self.config["models"]["manager"]["name"],
                "research": self.config["models"]["research_agent"]["name"],
                "rag": self.config["models"]["rag_agent"]["name"],
            },
            "tools": ["web_search", "knowledge_search", "simple_chat", "sql_executor"],
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
