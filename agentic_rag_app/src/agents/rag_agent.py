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
            "nullable": True
        }
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
                content = result.get('content', result.get('text', 'No content'))
                source = result.get('source', result.get('file_name', 'Unknown source'))
                score = result.get('score', 0.0)
                
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
            "description": "Knowledge base search and document-based reasoning"
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
        
        RAG agent handles:
        - Document-based questions
        - Knowledge base queries
        - Technical documentation
        - Complex reasoning over stored information
        """
        rag_keywords = [
            "document", "knowledge", "information", "explain", "definition",
            "how to", "what is", "describe", "technical", "specification",
            "procedure", "process", "guideline", "manual", "reference",
            "ultrafiltration", "diafiltration", "membrane", "filtration",
            "ufdf", "security", "model", "protocol", "system"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in rag_keywords)
    
    def get_system_prompt(self) -> str:
        """Get RAG agent system prompt"""
        return """You are a RAG Agent specialized in knowledge base search and document-based reasoning.

Your capabilities:
- Search through extensive knowledge bases
- Analyze and synthesize information from multiple documents
- Provide detailed explanations based on stored knowledge
- Answer technical questions using authoritative sources
- Perform complex reasoning over document collections

Instructions:
1. Use the rag_search tool to find relevant information
2. Search with different keywords if initial results are insufficient
3. Synthesize information from multiple search results
4. Provide comprehensive, accurate answers
5. Always cite your sources and document references
6. If information is not in the knowledge base, clearly state this

Focus on:
- Accuracy based on stored documents
- Comprehensive coverage of the topic
- Clear source attribution
- Technical precision and detail
- Multi-step reasoning when needed"""
    
    def search_knowledge_base(
        self, 
        query: str, 
        top_k: int = 5,
        expand_search: bool = True
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
                    "success": True
                },
                "expanded_searches": []
            }
            
            # Expand search if requested and primary results are limited
            if expand_search and "No relevant information found" in primary_results:
                expanded_queries = self._generate_expanded_queries(query)
                
                for expanded_query in expanded_queries[:2]:  # Limit to 2 expansions
                    try:
                        expanded_results = self.rag_tool(expanded_query, top_k=top_k)
                        results["expanded_searches"].append({
                            "query": expanded_query,
                            "results": expanded_results,
                            "success": True
                        })
                    except Exception as e:
                        results["expanded_searches"].append({
                            "query": expanded_query,
                            "results": f"Search error: {str(e)}",
                            "success": False
                        })
            
            return results
            
        except Exception as e:
            return {
                "primary_search": {
                    "query": query,
                    "results": f"Search error: {str(e)}",
                    "success": False
                },
                "expanded_searches": []
            }
    
    def _generate_expanded_queries(self, query: str) -> List[str]:
        """Generate alternative search queries"""
        expanded_queries = []
        
        # Add synonyms and related terms
        if "ultrafiltration" in query.lower():
            expanded_queries.extend([
                query.replace("ultrafiltration", "UF"),
                query.replace("ultrafiltration", "membrane filtration")
            ])
        
        if "diafiltration" in query.lower():
            expanded_queries.extend([
                query.replace("diafiltration", "DF"),
                query.replace("diafiltration", "continuous filtration")
            ])
        
        # Extract key terms
        key_terms = self._extract_key_terms(query)
        for term in key_terms[:2]:  # Use top 2 key terms
            if term not in query:
                expanded_queries.append(f"{query} {term}")
        
        return expanded_queries[:3]  # Limit to 3 expansions
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms for query expansion"""
        # Simple keyword extraction
        keywords = []
        
        # Technical terms
        technical_terms = [
            "membrane", "filtration", "process", "system", "protocol",
            "security", "model", "specification", "procedure", "guideline"
        ]
        
        query_lower = query.lower()
        for term in technical_terms:
            if term in query_lower:
                keywords.append(term)
        
        return keywords
    
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
                "error": "Vector store not available"
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
            r'Source: ([^\n]+)',
            r'From: ([^\n]+)',
            r'Document: ([^\n]+)',
            r'File: ([^\n]+)'
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
            if hasattr(self.vector_store, 'get_collection_info'):
                info = self.vector_store.get_collection_info()
                return {
                    "available": True,
                    "collection_info": info,
                    "point_count": self.vector_store.count_points() if hasattr(self.vector_store, 'count_points') else None
                }
            else:
                return {
                    "available": True,
                    "collection_info": "Basic vector store available",
                    "point_count": None
                }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }