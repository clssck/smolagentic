"""
RAG Agent

Specialized agent for knowledge base search and document-based reasoning.
"""

import json
from typing import Dict, Any, List, Optional
from smolagents import Tool
from .base_agent import BaseAgent


class RAGTool(Tool):
    """Custom RAG tool for vector store integration"""

    name = "rag_search"
    description = "Search the knowledge base for relevant information"
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query for technical knowledge",
        },
        "top_k": {
            "type": "integer",
            "description": "Number of results to return (default: 5)",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, vector_store=None, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store

    def forward(self, query: str, top_k: int = 5) -> str:
        """Execute RAG search"""
        if not self.vector_store:
            return "Knowledge base not available"

        try:
            # Perform vector search
            results = self.vector_store.search(query, top_k=top_k)

            if not results:
                return "No relevant information found in the knowledge base."

            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                content = result.get("content", result.get("text", "No content"))
                source = result.get("source", result.get("file_name", "Unknown source"))
                score = result.get("score", 0.0)

                formatted_results.append(
                    f"Result {i} (Relevance: {score:.3f}):\n"
                    f"Source: {source}\n"
                    f"Content: {content}\n"
                )

            return "\n".join(formatted_results)

        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"


class RAGAgent(BaseAgent):
    """Agent specialized for knowledge base search and document reasoning"""

    def __init__(self, vector_store=None, **kwargs):
        """Initialize RAG agent"""

        # Set defaults
        config = {
            "name": "rag_agent",
            "model_id": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
            "max_steps": 6,
            "temperature": 0.1,
            "max_tokens": 1200,
            "description": "Knowledge base search and document-based reasoning",
        }

        # Update with provided kwargs
        config.update(kwargs)

        # Initialize RAG tool
        self.vector_store = vector_store
        rag_tool = RAGTool(vector_store)
        config["tools"] = [rag_tool]

        super().__init__(**config)

        self.rag_tool = rag_tool

    def can_handle(self, query: str, context: Dict[str, Any] = None) -> bool:
        """
        Determine if this agent can handle the query

        RAG agent handles queries that likely need document retrieval.
        Uses a simple heuristic: if it's a question or informational request.
        """
        query_lower = query.lower().strip()

        # Question patterns
        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        starts_with_question = any(
            query_lower.startswith(word) for word in question_words
        )

        # Informational request patterns
        info_patterns = ["explain", "describe", "tell me", "show me", "find", "search"]
        has_info_pattern = any(pattern in query_lower for pattern in info_patterns)

        # Ends with question mark
        is_question = query_lower.endswith("?")

        return starts_with_question or has_info_pattern or is_question

    def get_system_prompt(self) -> str:
        """Get RAG agent system prompt"""
        return """You are a RAG Agent specialized in knowledge base search and document-based reasoning.

Your capabilities:
- Search through extensive knowledge bases and document collections
- Analyze and synthesize information from multiple sources
- Provide detailed explanations based on stored knowledge
- Answer questions using authoritative, verified information
- Perform complex reasoning and analysis over document sets

Instructions:
1. Use available search tools to find relevant information
2. Try multiple search strategies if initial results are insufficient
3. Synthesize information from multiple sources when possible
4. Provide comprehensive, well-reasoned answers
5. Always cite sources and provide document references
6. Be transparent about limitations - if information isn't available, say so clearly
7. When uncertain, indicate confidence levels in your responses

Focus on delivering accurate, well-sourced, and comprehensive responses based on the available knowledge base."""

    def search_knowledge_base(
        self, query: str, top_k: int = 5, expand_search: bool = True
    ) -> Dict[str, Any]:
        """
        Search the knowledge base with optional query expansion

        Args:
            query: Search query
            top_k: Number of results to return
            expand_search: Whether to try alternative search terms

        Returns:
            Search results with metadata
        """
        try:
            # Primary search
            primary_results = self.rag_tool(query, top_k=top_k)

            results = {
                "primary_search": {
                    "query": query,
                    "results": primary_results,
                    "success": True,
                },
                "expanded_searches": [],
            }

            # Expand search if requested and primary results are limited
            if expand_search and "No relevant information found" in primary_results:
                expanded_queries = self._generate_expanded_queries(query)

                for expanded_query in expanded_queries[:2]:  # Limit to 2 expansions
                    try:
                        expanded_results = self.rag_tool(expanded_query, top_k=top_k)
                        results["expanded_searches"].append(
                            {
                                "query": expanded_query,
                                "results": expanded_results,
                                "success": True,
                            }
                        )
                    except Exception as e:
                        results["expanded_searches"].append(
                            {
                                "query": expanded_query,
                                "results": f"Search error: {str(e)}",
                                "success": False,
                            }
                        )

            return results

        except Exception as e:
            return {
                "primary_search": {
                    "query": query,
                    "results": f"Search error: {str(e)}",
                    "success": False,
                },
                "expanded_searches": [],
            }

    def _generate_expanded_queries(self, query: str) -> List[str]:
        """Generate alternative search queries using generic approaches"""
        expanded_queries = []

        # Remove question words for better search
        question_words = ["what is", "how to", "why does", "when should", "where can"]
        clean_query = query.lower()
        for qw in question_words:
            if clean_query.startswith(qw):
                clean_query = clean_query[len(qw) :].strip()
                break

        if clean_query != query.lower():
            expanded_queries.append(clean_query)

        # Split into key terms and search with individual terms
        words = query.split()
        if len(words) > 2:
            # Try with just the main nouns/terms (skip common words)
            skip_words = {
                "the",
                "a",
                "an",
                "is",
                "are",
                "how",
                "what",
                "why",
                "when",
                "where",
            }
            key_words = [w for w in words if w.lower() not in skip_words and len(w) > 2]
            if len(key_words) >= 2:
                expanded_queries.append(" ".join(key_words[:3]))

        return expanded_queries[:2]  # Limit to 2 expansions

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query using generic NLP approaches"""
        import re

        # Remove common stop words and extract meaningful terms
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "what",
            "how",
            "why",
            "when",
            "where",
            "which",
            "who",
        }

        # Extract words that are likely to be important terms
        words = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
        key_terms = [word for word in words if word not in stop_words]

        return key_terms[:5]  # Return top 5 terms

    def run(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute RAG query with enhanced search capabilities
        """
        context = context or {}

        # Check if vector store is available
        if not self.vector_store:
            return {
                "response": "Knowledge base is not available. Please ensure the vector store is properly configured.",
                "agent_name": self.name,
                "execution_time": 0.0,
                "model_used": self.model_id,
                "tools_used": [],
                "success": False,
                "error": "Vector store not available",
            }

        # Enhance query with RAG context
        enhanced_query = f"""Knowledge Base Search Task: {query}

Please search the knowledge base thoroughly to answer this query. Use multiple searches if needed and provide a comprehensive response based on the available documents."""

        # Execute the base agent logic
        result = super().run(enhanced_query, context)

        # Add RAG-specific metadata
        if result["success"]:
            result["knowledge_base_used"] = True
            result["search_type"] = "vector_similarity"

            # Extract document sources from response
            sources = self._extract_document_sources(result["response"])
            result["document_sources"] = sources

        return result

    def _extract_document_sources(self, response: str) -> List[str]:
        """Extract document sources mentioned in the response"""
        sources = []

        # Look for common source patterns
        import re

        source_patterns = [
            r"Source: ([^\n]+)",
            r"From: ([^\n]+)",
            r"Document: ([^\n]+)",
            r"File: ([^\n]+)",
        ]

        for pattern in source_patterns:
            matches = re.findall(pattern, str(response))
            sources.extend(matches)

        # Remove duplicates and limit
        unique_sources = list(set(sources))[:10]
        return unique_sources

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        if not self.vector_store:
            return {"available": False, "error": "Vector store not configured"}

        try:
            # Get collection info if available
            if hasattr(self.vector_store, "get_collection_info"):
                info = self.vector_store.get_collection_info()
                return {
                    "available": True,
                    "collection_info": info,
                    "point_count": self.vector_store.count_points()
                    if hasattr(self.vector_store, "count_points")
                    else None,
                }
            else:
                return {
                    "available": True,
                    "collection_info": "Basic vector store available",
                    "point_count": None,
                }
        except Exception as e:
            return {"available": False, "error": str(e)}
