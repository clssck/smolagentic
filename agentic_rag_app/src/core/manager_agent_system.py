"""
Manager Agent System
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from dataclasses import dataclass, field

from smolagents import (
    ToolCallingAgent,
    Tool,
    WebSearchTool,
    VisitWebpageTool,
    LiteLLMModel,
    PromptTemplates,
    PlanningPromptTemplate,
    ManagedAgentPromptTemplate,
    FinalAnswerPromptTemplate,
    tool,
    FinalAnswerTool,
)

try:
    from smolagents import LiteLLMRouterModel

    ROUTER_MODEL_AVAILABLE = True
except ImportError:
    ROUTER_MODEL_AVAILABLE = False

try:
    from smolagents.monitoring import TreeFileLogger, JsonFileLogger

    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False

# Import additional smolagents components for optimization
try:
    from smolagents.utils import (
        AgentError,
        AgentExecutionError,
        AgentToolExecutionError,
    )
    from smolagents import ToolCollection

    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False

try:
    from src.agents.database_agent import DatabaseAgent

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

try:
    from src.core.model_pool import get_model_by_config, shared_models
    from src.core.shared_browser import (
        SharedBrowserSession,
        StatefulWebSearchTool,
        StatefulWebVisitTool,
        BrowserNavigationTool,
    )

    SHARED_TOOLS_AVAILABLE = True
except ImportError:
    SHARED_TOOLS_AVAILABLE = False

sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class PerformanceMetrics:
    """Performance monitoring for the manager agent system"""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    total_time: float = 0.0
    agent_call_counts: Dict[str, int] = field(default_factory=dict)
    tool_call_counts: Dict[str, int] = field(default_factory=dict)

    def update_tokens(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1

    def update_agent_call(self, agent_name: str):
        self.agent_call_counts[agent_name] = (
            self.agent_call_counts.get(agent_name, 0) + 1
        )

    def update_tool_call(self, tool_name: str):
        self.tool_call_counts[tool_name] = self.tool_call_counts.get(tool_name, 0) + 1

    def get_stats(self) -> Dict[str, Any]:
        avg_time = (
            self.total_time / self.total_requests if self.total_requests > 0 else 0
        )
        total_tokens = self.total_input_tokens + self.total_output_tokens

        return {
            "total_tokens": total_tokens,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_requests": self.total_requests,
            "average_time_seconds": round(avg_time, 2),
            "agent_calls": dict(self.agent_call_counts),
            "tool_calls": dict(self.tool_call_counts),
            "tokens_per_request": round(total_tokens / self.total_requests, 1)
            if self.total_requests > 0
            else 0,
        }


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

        # Advanced tool result caching
        self.tool_result_cache = {}  # Cache for expensive operations
        self.cache_hits = 0
        self.cache_misses = 0

        # Initialize logger
        import logging

        self.logger = logging.getLogger(__name__)

    def _extract_content_from_result(self, result):
        """Extract actual text content from search result, handling _node_content JSON"""
        import json
        import os

        # Try to get content from file_path first for full context
        if "payload" in result and "file_path" in result["payload"]:
            try:
                file_path = result["payload"]["file_path"]
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_content = f.read().strip()
                        if file_content:
                            # Return first 2000 characters for comprehensive content
                            return (
                                file_content[:2000] + "..."
                                if len(file_content) > 2000
                                else file_content
                            )
            except Exception as e:
                self.logger.warning(f"Failed to read file content: {e}")

        # Try different content fields in order of preference
        content = result.get("content", "").strip()
        if content:
            return content

        text = result.get("text", "").strip()
        if text:
            return text

        # Try to extract from _node_content JSON field
        if "payload" in result and "_node_content" in result["payload"]:
            try:
                node_content_str = result["payload"]["_node_content"]
                node_data = json.loads(node_content_str)
                node_text = node_data.get("text", "").strip()
                if node_text:
                    return node_text
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Failed to parse _node_content: {e}")

        # Fallback to string representation
        return str(result)[:200] + "..." if len(str(result)) > 200 else str(result)

    def _extract_citation_from_result(self, result):
        """Extract citation information from search result"""
        citation_parts = []

        if "payload" in result:
            payload = result["payload"]

            # Get filename
            file_name = payload.get("file_name", "Unknown Document")
            citation_parts.append(file_name)

            # Get file type
            file_type = payload.get("file_type", "")
            if file_type:
                citation_parts.append(f"({file_type})")

            # Get date if available
            creation_date = payload.get("creation_date", "")
            if creation_date:
                citation_parts.append(f"Created: {creation_date}")

            # Get file path for reference
            file_path = payload.get("file_path", "")
            if file_path:
                # Extract just the relative path from test_data onwards
                if "test_data" in file_path:
                    rel_path = file_path[file_path.find("test_data") :]
                    citation_parts.append(f"Source: {rel_path}")

        return " | ".join(citation_parts) if citation_parts else "Unknown Source"

    def forward(self, query: str) -> str:
        """Search the knowledge base with context awareness and caching"""
        import time
        import hashlib

        try:
            # Create cache key for the query
            cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()

            # Check cache first with improved cache management
            if cache_key in self.tool_result_cache:
                cached_result = self.tool_result_cache[cache_key]
                # Check if cache is still fresh (within 30 minutes for better performance)
                if time.time() - cached_result["timestamp"] < 1800:
                    self.cache_hits += 1
                    self.logger.info(f"🎯 Cache hit for query: {query[:50]}...")
                    # Update cache access time for LRU behavior
                    cached_result["last_accessed"] = time.time()
                    return cached_result["result"]
                else:
                    # Remove stale cache entry
                    del self.tool_result_cache[cache_key]

            # Clean cache if it gets too large (keep last 100 entries)
            if len(self.tool_result_cache) > 100:
                # Remove oldest entries
                sorted_cache = sorted(
                    self.tool_result_cache.items(),
                    key=lambda x: x[1].get("last_accessed", x[1]["timestamp"]),
                )
                for old_key, _ in sorted_cache[:20]:  # Remove 20 oldest
                    del self.tool_result_cache[old_key]

            self.cache_misses += 1

            # Add query to history
            self.search_history.append(
                {"query": query, "timestamp": time.time(), "results_found": 0}
            )

            # Get related context from previous searches
            related_context = self._get_related_context(query)

            if self.vector_store:
                # Hybrid search approach
                all_results = []
                methods_used = []

                # Method 1: Semantic search
                try:
                    from src.utils.embedding_service import embed_query

                    query_embedding = embed_query(query)
                    semantic_results = self.vector_store.search(
                        query_vector=query_embedding, top_k=3
                    )
                    if semantic_results:
                        for result in semantic_results:
                            result["search_method"] = "semantic"
                        all_results.extend(semantic_results)
                        methods_used.append("semantic")
                except Exception as e:
                    self.logger.warning(f"Semantic search failed: {e}")

                # Method 2: Text-based search
                try:
                    text_results = self.vector_store.search_by_text_filter(
                        query, limit=3
                    )
                    if text_results:
                        for result in text_results:
                            result["search_method"] = "text"
                        all_results.extend(text_results)
                        methods_used.append("text")
                except Exception as e:
                    self.logger.warning(f"Text search failed: {e}")

                # Method 3: Keyword fallback
                if not all_results:
                    try:
                        keywords = query.lower().split()
                        for keyword in keywords[:2]:  # Try top 2 keywords
                            keyword_results = self.vector_store.search_by_text_filter(
                                keyword, limit=2
                            )
                            if keyword_results:
                                for result in keyword_results:
                                    result["search_method"] = f"keyword-{keyword}"
                                all_results.extend(keyword_results)
                                methods_used.append(f"keyword")
                                break
                    except Exception as e:
                        self.logger.warning(f"Keyword search failed: {e}")

                # Deduplicate and rank results
                seen_ids = set()
                unique_results = []
                for result in all_results:
                    result_id = result.get("id")
                    if result_id not in seen_ids:
                        unique_results.append(result)
                        seen_ids.add(result_id)

                # Enhanced result ranking
                if unique_results:
                    ranked_results = self._rank_results(unique_results, query)
                    results = ranked_results[:5]  # Limit to top 5
                else:
                    results = []
                search_type = "+".join(methods_used) if methods_used else "none"

                if results:
                    # Update search history with results count
                    self.search_history[-1]["results_found"] = len(results)

                    # Store query context for future searches
                    self.query_context[query] = {
                        "results": results,
                        "timestamp": time.time(),
                        "summary": f"Found {len(results)} relevant documents using {search_type} search",
                        "search_type": search_type,
                    }

                    # Build context with citations
                    context_parts = []
                    citations = []

                    for i, result in enumerate(results):
                        content = self._extract_content_from_result(result)
                        citation = self._extract_citation_from_result(result)
                        relevance = result.get(
                            "score", result.get("relevance_score", 0)
                        )

                        context_parts.append(
                            f"Document {i + 1} (relevance: {relevance:.3f}): {content}"
                        )
                        citations.append(f"[{i + 1}] {citation}")

                    context = "\n\n".join(context_parts)
                    citations_text = "\n".join(citations)

                    response = f"Knowledge base search results for '{query}' (using {search_type} search):\n\n{context}\n\nSOURCES:\n{citations_text}"

                    # Add related context if available
                    if related_context:
                        response = f"{related_context}\n\n{response}"

                    # Cache successful results with access time
                    current_time = time.time()
                    self.tool_result_cache[cache_key] = {
                        "result": response,
                        "timestamp": current_time,
                        "last_accessed": current_time,
                        "query": query,
                        "results_count": len(results),
                    }

                    # Manage cache size (keep last 100 entries)
                    if len(self.tool_result_cache) > 100:
                        oldest_key = min(
                            self.tool_result_cache.keys(),
                            key=lambda k: self.tool_result_cache[k]["timestamp"],
                        )
                        del self.tool_result_cache[oldest_key]

                    return response
                else:
                    return f"No relevant documents found in knowledge base for '{query}' (tried {search_type} search)"
            else:
                # Fallback mode with context
                fallback_response = f"Knowledge base search for '{query}': [Vector store not available - using fallback]"
                if related_context:
                    fallback_response = f"{related_context}\n\n{fallback_response}"
                return fallback_response

        except Exception as e:
            return f"Error searching knowledge base for '{query}': {e!s}"

    def _rank_results(
        self, results: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """
        Rank results using multiple relevance signals.

        Args:
            results: List of search results
            query: Original search query

        Returns:
            Ranked list of results
        """
        query_terms = set(query.lower().split())

        for result in results:
            score = 0.0

            # Base score from search method and original score
            search_method = result.get("search_method", "unknown")
            base_score = result.get("score", 0.5)  # Default score if not provided

            # Method-based scoring
            if search_method == "semantic":
                score += base_score * 1.5  # Prefer semantic results
            elif search_method == "text":
                score += base_score * 1.2  # Text search is good
            elif "keyword" in search_method:
                score += base_score * 1.0  # Keyword is fallback
            else:
                score += base_score * 0.8

            # Content quality scoring
            content = result.get("content", result.get("text", ""))
            content_lower = content.lower()

            # Length bonus (prefer substantial content)
            content_length = len(content)
            if content_length > 1000:
                score += 0.3
            elif content_length > 500:
                score += 0.2
            elif content_length > 200:
                score += 0.1
            elif content_length < 50:
                score -= 0.2  # Penalize very short content

            # Query term matching
            content_terms = set(content_lower.split())
            query_matches = len(query_terms.intersection(content_terms))
            match_ratio = query_matches / len(query_terms) if query_terms else 0
            score += match_ratio * 0.5

            # Exact phrase matching bonus
            if query.lower() in content_lower:
                score += 0.4

            # Technical content indicators (boost for code, technical terms)
            technical_indicators = [
                "def ",
                "class ",
                "import ",
                "function",
                "algorithm",
                "method",
                "python",
                "javascript",
                "html",
                "css",
                "sql",
                "api",
                "json",
                "machine learning",
                "neural network",
                "data science",
            ]

            tech_matches = sum(
                1 for indicator in technical_indicators if indicator in content_lower
            )
            if tech_matches > 0:
                score += min(tech_matches * 0.1, 0.3)  # Cap at 0.3

            # Freshness bonus (if timestamp available)
            if "timestamp" in result:
                try:
                    import time

                    age_days = (time.time() - result["timestamp"]) / (24 * 3600)
                    if age_days < 30:  # Content less than 30 days old
                        score += 0.1
                except:
                    pass

            # Structure bonus (well-formatted content)
            if any(marker in content for marker in ["##", "```", "1.", "- "]):
                score += 0.1  # Structured content bonus

            result["relevance_score"] = round(score, 3)

        # Sort by relevance score (descending)
        ranked_results = sorted(
            results, key=lambda x: x.get("relevance_score", 0), reverse=True
        )

        # Log ranking for debugging
        self.logger.debug(f"Ranked {len(results)} results for query '{query}':")
        for i, result in enumerate(ranked_results[:3]):  # Log top 3
            method = result.get("search_method", "unknown")
            score = result.get("relevance_score", 0)
            content_preview = result.get("content", "")[:50] + "..."
            self.logger.debug(
                f"  {i + 1}. {method} (score: {score}): {content_preview}"
            )

        return ranked_results

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
            context += (
                f"- '{search['query']}' ({search['results_found']} results found)\n"
            )

        return context

    def get_search_stats(self) -> dict:
        """Get comprehensive search session statistics"""
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (
            (self.cache_hits / total_cache_requests * 100)
            if total_cache_requests > 0
            else 0
        )

        return {
            "session_id": self.session_id,
            "total_searches": len(self.search_history),
            "unique_queries": len(self.query_context),
            "recent_queries": [s["query"] for s in self.search_history[-3:]],
            "average_results": sum(s["results_found"] for s in self.search_history)
            / len(self.search_history)
            if self.search_history
            else 0,
            # Advanced caching statistics
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate_percent": round(cache_hit_rate, 1),
            "cache_size": len(self.tool_result_cache),
            "cached_queries": list(self.tool_result_cache.keys())[-5:]
            if self.tool_result_cache
            else [],
        }

    def clear_context(self):
        """Clear search context, history, and cache"""
        self.search_history = []
        self.query_context = {}
        self.tool_result_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.session_id = str(int(time.time()))


# Global reference to the manager system for tool delegation
_manager_system_instance = None


# Use @tool decorator for cleaner tool definitions
@tool
def research_agent(task: str, additional_args: dict = None) -> str:
    """Delegate research tasks to the research agent for web search and current information

    Args:
        task: The research task or query to execute
        additional_args: Additional arguments (optional)
    """
    global _manager_system_instance
    if _manager_system_instance and hasattr(
        _manager_system_instance, "_research_agent_instance"
    ):
        try:
            # Delegate to the actual research agent
            result = _manager_system_instance._research_agent_instance.run(task)
            if isinstance(result, dict) and "response" in result:
                return result["response"]
            return str(result)
        except Exception as e:
            return f"Research agent error: {str(e)}"
    return f"Research task received: {task} (agent not available)"


@tool
def rag_agent(query: str, additional_args: dict = None) -> str:
    """Delegate knowledge base queries to the RAG agent for document search and retrieval

    Args:
        query: The knowledge base query to execute
        additional_args: Additional arguments (optional)
    """
    global _manager_system_instance
    if _manager_system_instance and hasattr(
        _manager_system_instance, "_rag_agent_instance"
    ):
        try:
            # Delegate to the actual RAG agent
            result = _manager_system_instance._rag_agent_instance.run(query)
            if isinstance(result, dict) and "response" in result:
                return result["response"]
            return str(result)
        except Exception as e:
            return f"RAG agent error: {str(e)}"
    return f"Knowledge base query received: {query} (agent not available)"


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
        self.name = (
            "Manager Agent System"  # Add name attribute for Gradio UI compatibility
        )
        self.config = self._load_config(config_path)
        self.vector_store = None
        self.manager_agent = None
        self.database_agent = None
        self.conversation_history = []  # Simple conversation memory
        self.debug_mode = False

        # Shared context managers
        self.browser_session = None
        self.rag_tool_instance = None

        # Performance monitoring
        self.metrics = PerformanceMetrics()

        # Advanced logging setup
        self.loggers = []
        self._setup_advanced_logging()

        self._setup_vector_store()
        self._setup_database()
        self._setup_shared_context()
        self._setup_agents()

    def _load_config(self, config_path: str | None) -> dict[str, Any]:
        """Load configuration"""
        default_config = {
            "models": {
                "manager": {
                    "name": "openrouter/mistralai/mistral-small-3.2-24b-instruct",  # Reliable coordination via Mistral
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
            if (
                self.config["vector_store"]["enabled"]
                and VECTOR_STORE_AVAILABLE
                and UTILS_AVAILABLE
            ):
                # Try to use existing vector store
                config = Config()
                self.vector_store = QdrantVectorStore(
                    collection_name=config.QDRANT_COLLECTION_NAME, config=config
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

    def _setup_advanced_logging(self):
        """Setup advanced logging with Tree and JSON loggers"""
        try:
            if ADVANCED_LOGGING_AVAILABLE:
                # Create logs directory if it doesn't exist
                logs_dir = Path("logs")
                logs_dir.mkdir(exist_ok=True)

                # Tree logger for beautiful console output
                tree_logger = TreeFileLogger()
                self.loggers.append(tree_logger)

                # JSON logger for machine-readable logs
                json_logger = JsonFileLogger(str(logs_dir / "agent_performance.json"))
                self.loggers.append(json_logger)

                print("✅ Advanced logging enabled (Tree + JSON)")
            else:
                print("⚠️ Advanced logging not available")
        except Exception as e:
            print(f"Advanced logging setup failed: {e}")

        # Enable enhanced debug output to terminal
        print("🚀 Manager Agent System with enhanced debug output enabled")
        self.debug_mode = True

    def _create_router_model(self, model_config: dict, model_name: str):
        """Create a router model with multiple providers for reliability"""
        if not ROUTER_MODEL_AVAILABLE:
            # Fallback to regular LiteLLM model
            return LiteLLMModel(
                model_id=model_config["name"],
                temperature=model_config["temperature"],
                max_tokens=model_config["max_tokens"],
            )

        try:
            # Create model list with multiple providers for same model group
            primary_model = model_config["name"]

            # Define backup models based on primary
            model_list = []

            if "mistral" in primary_model:
                # Groq with OpenRouter backup
                model_list = [
                    {
                        "model_name": f"{model_name}-group",
                        "litellm_params": {
                            "model": primary_model,
                            "temperature": model_config["temperature"],
                            "max_tokens": model_config["max_tokens"],
                        },
                    },
                    {
                        "model_name": f"{model_name}-group",
                        "litellm_params": {
                            "model": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
                            "temperature": model_config["temperature"],
                            "max_tokens": model_config["max_tokens"],
                        },
                    },
                ]
            elif "openrouter" in primary_model:
                # OpenRouter with Groq backup
                model_list = [
                    {
                        "model_name": f"{model_name}-group",
                        "litellm_params": {
                            "model": primary_model,
                            "temperature": model_config["temperature"],
                            "max_tokens": model_config["max_tokens"],
                        },
                    },
                    {
                        "model_name": f"{model_name}-group",
                        "litellm_params": {
                            "model": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
                            "temperature": model_config["temperature"],
                            "max_tokens": model_config["max_tokens"],
                        },
                    },
                ]
            else:
                # Default model list
                model_list = [
                    {
                        "model_name": f"{model_name}-group",
                        "litellm_params": {
                            "model": primary_model,
                            "temperature": model_config["temperature"],
                            "max_tokens": model_config["max_tokens"],
                        },
                    }
                ]

            # Create router with failover and load balancing
            router_model = LiteLLMRouterModel(
                model_id=f"{model_name}-group",
                model_list=model_list,
                client_kwargs={
                    "routing_strategy": "simple-shuffle",  # Load balancing
                    "num_retries": 2,  # Automatic retry on failure
                    "timeout": 60,  # Timeout per request
                },
            )

            print(
                f"✅ Router model created for {model_name} with {len(model_list)} providers"
            )
            return router_model

        except Exception as e:
            print(f"⚠️ Router model creation failed for {model_name}: {e}")
            # Fallback to regular model
            return LiteLLMModel(
                model_id=model_config["name"],
                temperature=model_config["temperature"],
                max_tokens=model_config["max_tokens"],
            )

    def _validate_rag_answer(self, final_answer: str, memory) -> bool:
        """Validate RAG agent answers for quality and relevance"""
        try:
            # Check minimum length
            if len(final_answer.strip()) < 50:
                print("⚠️ RAG answer too short, requesting retry")
                return False

            # Check for search context indicators
            search_indicators = [
                "search results",
                "knowledge base",
                "documents found",
                "information",
                "according to",
            ]
            has_context = any(
                indicator in final_answer.lower() for indicator in search_indicators
            )

            if not has_context:
                print("⚠️ RAG answer lacks search context, requesting retry")
                return False

            # Check for error messages
            error_indicators = [
                "no results found",
                "error occurred",
                "failed to search",
                "not available",
            ]
            has_errors = any(
                error in final_answer.lower() for error in error_indicators
            )

            if has_errors:
                print("⚠️ RAG answer contains errors, requesting retry")
                return False

            return True
        except Exception:
            return True  # Don't fail on validation errors

    def _validate_research_answer(self, final_answer: str, memory) -> bool:
        """Validate research agent answers for completeness"""
        try:
            # Check for web search indicators
            web_indicators = [
                "search results",
                "found information",
                "according to",
                "website",
                "source",
            ]
            has_web_context = any(
                indicator in final_answer.lower() for indicator in web_indicators
            )

            # Check minimum substantive length
            if len(final_answer.strip()) < 100:
                print("⚠️ Research answer too brief, requesting retry")
                return False

            return True
        except Exception:
            return True

    def _validate_manager_answer(self, final_answer: str, memory) -> bool:
        """Validate manager agent final answers"""
        try:
            # Ensure final answer exists and is meaningful
            if not final_answer or len(final_answer.strip()) < 20:
                print("⚠️ Manager answer too short, requesting retry")
                return False

            # Check for delegation confirmation
            if "delegat" in final_answer.lower() or "agent" in final_answer.lower():
                return True  # Manager properly delegated

            return True
        except Exception:
            return True

    def _plan_interrupt_callback(self, memory_step, agent):
        """Callback to interrupt after planning for debugging"""
        if isinstance(memory_step, PlanningStep) and self.debug_mode:
            print(f"🧠 Agent Plan: {memory_step}")
            response = input("Continue with this plan? (y/n): ")
            if response.lower() != "y":
                agent.interrupt()

    def _performance_monitoring_callback(self, step_log, agent=None):
        """Callback to monitor performance metrics"""
        try:
            # Update agent call counts
            if agent and hasattr(agent, "name"):
                self.metrics.update_agent_call(agent.name)

            # Update token usage if available
            if hasattr(step_log, "token_usage") and step_log.token_usage:
                self.metrics.update_tokens(
                    step_log.token_usage.input_tokens,
                    step_log.token_usage.output_tokens,
                )

            # Update tool call counts if this was a tool call
            if hasattr(step_log, "tool_calls") and step_log.tool_calls:
                for tool_call in step_log.tool_calls:
                    if hasattr(tool_call, "tool_name"):
                        self.metrics.update_tool_call(tool_call.tool_name)

        except Exception as e:
            # Don't let monitoring failures break the agent
            print(f"⚠️ Performance monitoring error: {e}")

    def _get_callbacks(self):
        """Get list of callbacks based on current settings"""
        callbacks = [
            self._performance_monitoring_callback
        ]  # Always monitor performance

        # Add advanced loggers if available
        if self.loggers:
            callbacks.extend(self.loggers)

        if self.debug_mode:
            callbacks.append(self._plan_interrupt_callback)

        return callbacks

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
                research_model = get_model_by_config(
                    "research_agent", research_model_config
                )
                rag_model = get_model_by_config("rag_agent", rag_model_config)
                print("Using shared model pool")
            else:
                # Create enhanced router models for reliability
                print("Creating router models with failover capabilities...")
                manager_model = self._create_router_model(
                    manager_model_config, "manager"
                )
                research_model = self._create_router_model(
                    research_model_config, "research"
                )
                rag_model = self._create_router_model(rag_model_config, "rag")
                print("Router models created with load balancing")

            # Create specialized agents
            specialized_agents = []

            # Research Agent (optimized Mistral Small 3.2)
            if self.config["agents"]["research"]["enabled"]:
                # Use shared browser tools if available
                if SHARED_TOOLS_AVAILABLE and self.browser_session:
                    research_tools = [
                        StatefulWebSearchTool(browser_session=self.browser_session),
                        StatefulWebVisitTool(browser_session=self.browser_session),
                        BrowserNavigationTool(browser_session=self.browser_session),
                    ]
                    print("Research agent using shared browser session")
                else:
                    research_tools = [WebSearchTool(), VisitWebpageTool()]
                    print("Research agent using individual tools")

                # Add FinalAnswerTool to research tools
                research_tools.append(FinalAnswerTool())

                research_agent_instance = ToolCallingAgent(
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
                    max_tool_threads=3,  # Enable parallel tool execution
                    final_answer_checks=[
                        self._validate_research_answer
                    ],  # Quality validation
                    step_callbacks=self._get_callbacks(),
                )
                specialized_agents.append(research_agent_instance)
                # Store reference for tool delegation
                self._research_agent_instance = research_agent_instance

            # RAG Agent (optimized Mistral Small 3.2)
            if self.config["agents"]["rag"]["enabled"]:
                # Create or reuse Compound RAG tool instance for best citations and synthesis
                if not self.rag_tool_instance:
                    try:
                        from src.core.compound_rag_tool import CompoundRAGTool

                        self.rag_tool_instance = CompoundRAGTool(
                            vector_store=self.vector_store
                        )
                        print(
                            "🔗 Using Compound RAG Tool with tool composition pipeline"
                        )
                    except ImportError:
                        try:
                            from src.core.enhanced_rag_tool import EnhancedRAGTool

                            self.rag_tool_instance = EnhancedRAGTool(
                                vector_store=self.vector_store
                            )
                            print("✅ Using Enhanced RAG Tool with better citations")
                        except ImportError:
                            self.rag_tool_instance = RAGTool(
                                vector_store=self.vector_store
                            )
                            print(
                                "⚠️  Enhanced RAG Tool not available, using basic RAG Tool"
                            )

                # Add FinalAnswerTool to RAG tools
                rag_tools = [self.rag_tool_instance, FinalAnswerTool()]

                # Create simple prompt templates for RAG agent to avoid formatting issues
                rag_prompt_templates = PromptTemplates(
                    system_prompt="You are a knowledge base search agent. Your ONLY job is to search using the knowledge_search tool and return its output. The tool provides detailed source information. When using final_answer, copy the tool's output exactly as received - do not modify, summarize, or rewrite it. The tool output includes document names, file paths, and content sections that must be preserved exactly.",
                    planning=PlanningPromptTemplate(
                        initial_plan="Search knowledge base.",
                        update_plan_pre_messages="Searching...",
                        update_plan_post_messages="Continuing.",
                    ),
                    managed_agent=ManagedAgentPromptTemplate(
                        task="Searching knowledge base.", report="Search completed."
                    ),
                    final_answer=FinalAnswerPromptTemplate(
                        pre_messages="", post_messages=""
                    ),
                )

                rag_agent_instance = ToolCallingAgent(
                    tools=rag_tools,
                    model=rag_model,
                    prompt_templates=rag_prompt_templates,
                    name="rag_agent",
                    description=rag_model_config.get(
                        "description", "Agent specialized in knowledge base search"
                    ),
                    max_steps=3,  # Reduced to prevent over-processing
                    verbosity_level=0,  # Reduce verbosity to avoid formatting conflicts
                    planning_interval=1,  # Minimal planning
                    provide_run_summary=False,  # No summary to reduce complexity
                    max_tool_threads=1,  # Sequential to avoid race conditions
                    final_answer_checks=[],  # No validation to prevent loops
                    step_callbacks=[],  # No callbacks to reduce complexity
                )
                specialized_agents.append(rag_agent_instance)
                # Store reference for tool delegation
                self._rag_agent_instance = rag_agent_instance

            # Manager Agent (ultra-fast Groq Qwen3-32B)
            simple_tools = []
            if self.config["agents"]["simple"]["enabled"]:
                simple_tools.append(simple_chat)  # Use the @tool decorated function

            # Add research agent tool for proper delegation
            if self.config["agents"]["research"]["enabled"]:
                simple_tools.append(research_agent)

            # Add RAG agent tool for knowledge base queries
            if self.config["agents"]["rag"]["enabled"]:
                simple_tools.append(rag_agent)

            # Add FinalAnswerTool for proper response handling
            simple_tools.append(FinalAnswerTool())
            # Direct and action-oriented system prompt with citation emphasis
            system_prompt = """You are a Manager Agent that routes queries to specialized tools. Your job is to preserve tool outputs exactly.

TOOLS:
- rag_agent: Search knowledge base - returns detailed source documents with filenames, paths, and content
- research_agent: Web search for current news and information
- simple_chat: Basic questions, greetings, math

ROUTING RULES:
- Technical/scientific questions → rag_agent first
- Current events/news → research_agent  
- Simple questions → simple_chat

CRITICAL: When rag_agent returns output containing [PRESERVE_OUTPUT], you MUST return the tool's output exactly as-is without any modification, summary, or rewriting. This indicates detailed source information with document names, file paths, and content sections that must be preserved completely.

Look for [PRESERVE_OUTPUT] marker in tool results - if present, return the complete tool output unchanged."""

            # Create prompt templates with direct, action-oriented prompts
            prompt_templates = PromptTemplates(
                system_prompt=system_prompt,
                planning=PlanningPromptTemplate(
                    initial_plan="Step 1: Route query to appropriate tool.",
                    update_plan_pre_messages="Updating plan based on results.",
                    update_plan_post_messages="Continuing with next step.",
                ),
                managed_agent=ManagedAgentPromptTemplate(
                    task="Routing to specialized agent.", report="Agent completed task."
                ),
                final_answer=FinalAnswerPromptTemplate(
                    pre_messages="", post_messages=""
                ),
            )

            self.manager_agent = ToolCallingAgent(
                tools=simple_tools,
                model=manager_model,
                prompt_templates=prompt_templates,
                max_steps=3,  # Minimal steps for focused routing
                verbosity_level=0,  # Reduce verbosity to avoid confusion
                planning_interval=1,  # Less planning, more action
                provide_run_summary=False,  # No summary needed
                max_tool_threads=1,  # Sequential execution for clarity
                final_answer_checks=[],  # No validation to avoid loops
                step_callbacks=[],  # No callbacks to reduce complexity
            )

            print("Manager agent system initialized")

            # Set global reference for tool delegation
            global _manager_system_instance
            _manager_system_instance = self

        except Exception as e:
            print(f"Failed to setup agents: {e}")
            raise

    def run(self, query: str, **kwargs) -> str:
        """
        Run method for Gradio UI compatibility - delegates to run_query

        Args:
            query: The user query
            **kwargs: Additional arguments

        Returns:
            Response string
        """
        return self.run_query(query, **kwargs)

    def run_query(self, query: str, **kwargs) -> str:
        """
        Run a query through the manager agent system with optimized routing

        Args:
            query: The user query
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Response string
        """
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": query})

        # Quick routing for simple queries to avoid agent overhead
        if self._is_simple_query(query):
            response = self._handle_simple_query(query)
            self.conversation_history.append({"role": "assistant", "content": response})
            return response

        # ENHANCED RAG BYPASS: For knowledge base queries, use enhanced RAG tool directly
        if self._is_knowledge_query(query) and self.rag_tool_instance:
            try:
                print("🎯 Using enhanced RAG tool directly for better citations")
                response = self.rag_tool_instance.forward(query)
                self.conversation_history.append(
                    {"role": "assistant", "content": response}
                )
                return response
            except Exception as e:
                print(
                    f"⚠️ Enhanced RAG bypass failed: {e}, falling back to manager agent"
                )

        # Check for database queries first
        if self.database_agent and self._is_database_query(query):
            try:
                response = self.database_agent.query(query)
                self.conversation_history.append(
                    {"role": "assistant", "content": response}
                )
                return response
            except Exception as e:
                error_response = f"Database query error: {e}"
                self.conversation_history.append(
                    {"role": "assistant", "content": error_response}
                )
                return error_response

        if not self.manager_agent:
            return "Manager agent not available."

        try:
            # Track performance timing
            start_time = time.time()

            # Optimize context handling - only add context for complex queries
            context_query = query
            if len(self.conversation_history) > 2 and len(query.split()) > 5:
                recent_context = self.conversation_history[-4:]  # Last 2 exchanges
                context_summary = "\n".join(
                    [
                        f"{msg['role']}: {msg['content'][:100]}..."
                        for msg in recent_context
                    ]
                )
                context_query = f"Context:\n{context_summary}\n\nCurrent query: {query}"

            # Enhanced error handling with retries
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    # Run through manager agent with timeout handling
                    if ADVANCED_FEATURES_AVAILABLE:
                        result = self.manager_agent.run(context_query, reset=True)
                    else:
                        result = self.manager_agent.run(context_query)
                    break
                except (AgentExecutionError, AgentToolExecutionError) as e:
                    if attempt < max_retries - 1:
                        print(f"Retrying query due to agent error: {e}")
                        # Try with simpler query on retry
                        context_query = query
                        continue
                    else:
                        raise e
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Retrying query due to error: {e}")
                        continue
                    else:
                        raise e

            # Update performance metrics
            self.metrics.total_time += time.time() - start_time

            # Extract response from result with better handling
            if hasattr(result, "content"):
                response = result.content
            elif hasattr(result, "output"):
                response = result.output
            elif isinstance(result, str):
                response = result
            else:
                response = str(result)

            # Validate response quality
            if not response or len(response.strip()) < 5:
                response = "I encountered an issue generating a response. Could you try rephrasing your question?"

            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})

            # Keep history manageable (last 20 messages)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            return response

        except Exception as e:
            import traceback

            error_detail = str(e)
            if self.debug_mode:
                error_detail = traceback.format_exc()

            error_response = f"Error processing query: {error_detail}"
            self.conversation_history.append(
                {"role": "assistant", "content": error_response}
            )
            return error_response

    def _is_simple_query(self, query: str) -> bool:
        """Check if query can be handled with simple response to avoid agent overhead"""
        query_lower = query.lower().strip()

        # Simple greetings
        simple_patterns = [
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "thanks",
            "thank you",
            "bye",
            "goodbye",
            "see you",
        ]

        for pattern in simple_patterns:
            if pattern in query_lower:
                return True

        # Simple math (basic patterns)
        import re

        if re.search(r"\d+\s*[\+\-\*/]\s*\d+", query_lower):
            return True

        # Very short queries are likely simple
        if len(query.split()) <= 2 and len(query) <= 15:
            return True

        return False

    def _handle_simple_query(self, query: str) -> str:
        """Handle simple queries directly without agent overhead"""
        query_lower = query.lower().strip()

        # Greetings
        if any(
            greeting in query_lower
            for greeting in ["hello", "hi", "hey", "good morning", "good afternoon"]
        ):
            return "Hello! I'm your AI assistant. I can help with research, knowledge base queries, coding, and general questions. How can I assist you today?"

        # Thanks
        if any(thanks in query_lower for thanks in ["thanks", "thank you"]):
            return "You're welcome! Feel free to ask if you need anything else."

        # Goodbye
        if any(bye in query_lower for bye in ["bye", "goodbye", "see you"]):
            return "Goodbye! Have a great day!"

        # Simple math
        import re

        math_match = re.search(r"(\d+)\s*([\+\-\*/])\s*(\d+)", query_lower)
        if math_match:
            try:
                num1, op, num2 = math_match.groups()
                result = eval(f"{num1}{op}{num2}")  # Safe for simple operations
                return f"The result is: {result}"
            except:
                pass

        # Default simple response
        return (
            "I understand. Could you provide more details so I can assist you better?"
        )

    def _is_database_query(self, query: str) -> bool:
        """Check if query is database-related"""
        db_keywords = [
            "sql",
            "database",
            "table",
            "select",
            "query",
            "users",
            "orders",
            "data",
            "count",
            "total",
            "show me",
            "how many",
            "list all",
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in db_keywords)

    def _is_knowledge_query(self, query: str) -> bool:
        """Always use enhanced RAG for better citations - simple bypass"""
        # For now, always use enhanced RAG bypass for better citations
        # This can be made configurable later if needed
        return True

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
                "search_context": self.browser_session.get_search_context(),
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

    def get_performance_metrics(self) -> dict:
        """Get comprehensive performance metrics"""
        return self.metrics.get_stats()

    def clear_all_context(self):
        """Clear all shared context and history with advanced memory management"""
        self.clear_conversation_history()

        if self.browser_session:
            self.browser_session = SharedBrowserSession()
            print("🧹 Browser session reset")

        if self.rag_tool_instance:
            self.rag_tool_instance.clear_context()
            print("🧹 RAG context cleared")

        # Clear agent memory if available
        if self.manager_agent and hasattr(self.manager_agent, "memory"):
            self.manager_agent.memory.clear()
            print("🧹 Manager agent memory cleared")

        # Reset performance metrics
        self.metrics = PerformanceMetrics()
        print("🧹 Performance metrics reset")

    def optimize_memory(self):
        """Optimize memory usage by cleaning up old data"""
        # Limit conversation history to last 20 messages
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
            print("🧹 Conversation history optimized")

        # Clear old cache entries in embedding service if available
        try:
            from src.utils.embedding_service import get_embedding_service

            service = get_embedding_service()
            if hasattr(service, "_manage_cache_size"):
                service._manage_cache_size()
                print("🧹 Embedding cache optimized")
        except Exception:
            pass

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
