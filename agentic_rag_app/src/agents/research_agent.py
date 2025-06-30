"""
Research Agent

Specialized agent for web research and current information gathering.
"""

import re
from typing import Dict, Any
from smolagents import WebSearchTool, VisitWebpageTool
from .base_agent import BaseAgent


class ResearchAgent(BaseAgent):
    """Agent specialized for web research and information gathering"""
    
    def __init__(self, **kwargs):
        """Initialize research agent"""
        
        # Set defaults
        config = {
            "name": "research_agent",
            "model_id": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
            "max_steps": 8,
            "temperature": 0.1,
            "max_tokens": 1200,
            "description": "Web research and current information gathering"
        }
        
        # Update with provided kwargs
        config.update(kwargs)
        
        # Initialize tools
        tools = [
            WebSearchTool(),
            VisitWebpageTool()
        ]
        config["tools"] = tools
        
        super().__init__(**config)
    
    def can_handle(self, query: str, context: Dict[str, Any] = None) -> bool:
        """
        Determine if this agent can handle the query
        
        Research agent handles:
        - Current events and news
        - Web searches
        - Real-time information
        - Market data
        - Recent developments
        """
        research_keywords = [
            "current", "latest", "recent", "news", "today", "now", 
            "search", "web", "internet", "online", "website",
            "market", "price", "stock", "trend", "update",
            "what's", "whats", "happening", "development"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in research_keywords)
    
    def get_system_prompt(self) -> str:
        """Get research agent system prompt"""
        return """You are a Research Agent specialized in web research and current information gathering.

Your capabilities:
- Web search for current information
- Visit and analyze web pages
- Gather real-time data and news
- Research market trends and updates
- Find recent developments on topics

Instructions:
1. Use web search to find current, relevant information
2. Visit relevant web pages for detailed information
3. Synthesize information from multiple sources
4. Provide accurate, up-to-date responses
5. Always cite your sources
6. If information is not current, mention the date/timeframe

Focus on:
- Accuracy and credibility of sources
- Current and relevant information
- Comprehensive research
- Clear source attribution"""
    
    def _is_web_research_query(self, query: str) -> bool:
        """Check if query requires web research"""
        web_indicators = [
            "current", "latest", "recent", "today", "now", "2024", "2025",
            "news", "market", "price", "stock", "trend", "update",
            "search for", "find", "look up", "research"
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in web_indicators)
    
    def run(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute research query
        
        Enhanced with research-specific logic
        """
        context = context or {}
        
        # Add research context to the query
        if self._is_web_research_query(query):
            enhanced_query = f"""Research Task: {query}

Please conduct thorough web research to answer this query. Use multiple sources and provide current, accurate information with proper citations."""
        else:
            enhanced_query = query
        
        # Execute the base agent logic
        result = super().run(enhanced_query, context)
        
        # Add research-specific metadata
        if result["success"]:
            result["research_type"] = "web_research"
            result["sources_used"] = self._extract_sources_from_response(result["response"])
        
        return result
    
    def _extract_sources_from_response(self, response: str) -> list:
        """Extract sources mentioned in the response"""
        # Simple URL extraction
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, str(response))
        
        # Extract domain names
        sources = []
        for url in urls:
            try:
                domain = url.split('/')[2]
                if domain not in sources:
                    sources.append(domain)
            except:
                continue
        
        return sources[:5]  # Limit to 5 sources
    
    def search_web(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Direct web search method
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Search results
        """
        try:
            search_tool = WebSearchTool()
            results = search_tool(query)
            
            return {
                "success": True,
                "results": results,
                "query": query,
                "num_results": len(results) if isinstance(results, list) else 1
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }
    
    def visit_webpage(self, url: str) -> Dict[str, Any]:
        """
        Visit and analyze a webpage
        
        Args:
            url: URL to visit
            
        Returns:
            Page content and analysis
        """
        try:
            visit_tool = VisitWebpageTool()
            content = visit_tool(url)
            
            return {
                "success": True,
                "content": content,
                "url": url,
                "content_length": len(str(content))
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "content": None
            }