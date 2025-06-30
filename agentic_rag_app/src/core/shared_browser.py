"""
Shared Browser Session for Web Tools

Provides stateful browsing session that web tools can share for context continuity.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from smolagents import Tool


class SharedBrowserSession:
    """
    Shared browser session that maintains state across multiple tool calls
    
    Features:
    - Persistent browsing history
    - Current page content and viewport
    - Search context across page navigation
    - Session state management
    """
    
    def __init__(self, viewport_size: int = 1024 * 4):
        self.viewport_size = viewport_size
        self.history: List[Tuple[str, float]] = []  # (url, timestamp)
        self.current_page_content: str = ""
        self.current_page_title: str = ""
        self.viewport_current_page = 0
        self.viewport_pages: List[Tuple[int, int]] = []
        
        # Search context
        self.last_search_query: Optional[str] = None
        self.search_results_context: List[Dict[str, Any]] = []
        
        # Session metadata
        self.session_id = str(int(time.time()))
        self.total_pages_visited = 0
        self.total_searches_performed = 0
        
    @property
    def current_url(self) -> str:
        """Get current page URL"""
        return self.history[-1][0] if self.history else "about:blank"
    
    @property
    def viewport(self) -> str:
        """Get current viewport content"""
        if not self.viewport_pages or self.viewport_current_page >= len(self.viewport_pages):
            return self.current_page_content
        
        bounds = self.viewport_pages[self.viewport_current_page]
        return self.current_page_content[bounds[0]:bounds[1]]
    
    def visit_page(self, url: str) -> str:
        """Visit a page and return its content"""
        try:
            # Add to history
            self.history.append((url, time.time()))
            self.total_pages_visited += 1
            
            # Fetch page content
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Simple content extraction (in real implementation, use proper HTML parsing)
            content = response.text
            self.current_page_content = self._extract_text_content(content)
            self.current_page_title = self._extract_title(content)
            
            # Reset viewport
            self._split_pages()
            self.viewport_current_page = 0
            
            return f"Visited: {url}\nTitle: {self.current_page_title}\nContent preview: {self.viewport[:200]}..."
            
        except Exception as e:
            error_msg = f"Error visiting {url}: {e}"
            self.current_page_content = error_msg
            return error_msg
    
    def search_web(self, query: str) -> str:
        """Perform web search and store context"""
        self.last_search_query = query
        self.total_searches_performed += 1
        
        # Simple search simulation (in real implementation, use actual search API)
        search_result = {
            "query": query,
            "timestamp": time.time(),
            "results": [
                f"Search result 1 for '{query}'",
                f"Search result 2 for '{query}'",
                f"Search result 3 for '{query}'",
            ]
        }
        
        self.search_results_context.append(search_result)
        
        # Keep only last 10 searches
        if len(self.search_results_context) > 10:
            self.search_results_context = self.search_results_context[-10:]
        
        results_text = "\n".join([f"{i+1}. {result}" for i, result in enumerate(search_result["results"])])
        return f"Search results for '{query}':\n{results_text}"
    
    def page_down(self) -> str:
        """Navigate to next page viewport"""
        if self.viewport_current_page < len(self.viewport_pages) - 1:
            self.viewport_current_page += 1
            return f"Scrolled down. Page {self.viewport_current_page + 1} of {len(self.viewport_pages)}"
        return "Already at bottom of page"
    
    def page_up(self) -> str:
        """Navigate to previous page viewport"""
        if self.viewport_current_page > 0:
            self.viewport_current_page -= 1
            return f"Scrolled up. Page {self.viewport_current_page + 1} of {len(self.viewport_pages)}"
        return "Already at top of page"
    
    def get_search_context(self) -> str:
        """Get recent search context for continuity"""
        if not self.search_results_context:
            return "No recent searches"
        
        recent_searches = self.search_results_context[-3:]  # Last 3 searches
        context = "Recent search context:\n"
        for search in recent_searches:
            context += f"- '{search['query']}' ({len(search['results'])} results)\n"
        return context
    
    def get_browse_context(self) -> str:
        """Get recent browsing context"""
        if not self.history:
            return "No browsing history"
        
        recent_pages = self.history[-3:]  # Last 3 pages
        context = "Recent browsing context:\n"
        for url, timestamp in recent_pages:
            context += f"- {url} (visited {int(time.time() - timestamp)}s ago)\n"
        return context
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            "session_id": self.session_id,
            "pages_visited": self.total_pages_visited,
            "searches_performed": self.total_searches_performed,
            "current_url": self.current_url,
            "history_length": len(self.history),
            "search_context_length": len(self.search_results_context),
        }
    
    def _extract_text_content(self, html_content: str) -> str:
        """Extract text content from HTML (simplified)"""
        # In real implementation, use BeautifulSoup or similar
        import re
        # Remove script and style elements
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html_content)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:self.viewport_size * 2]  # Limit content size
    
    def _extract_title(self, html_content: str) -> str:
        """Extract page title from HTML"""
        import re
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        return title_match.group(1).strip() if title_match else "No title"
    
    def _split_pages(self):
        """Split content into viewport-sized pages"""
        self.viewport_pages = []
        if not self.current_page_content:
            self.viewport_pages = [(0, 0)]
            return
        
        content_length = len(self.current_page_content)
        start = 0
        
        while start < content_length:
            end = min(start + self.viewport_size, content_length)
            self.viewport_pages.append((start, end))
            start = end
        
        if not self.viewport_pages:
            self.viewport_pages = [(0, 0)]


class StatefulWebSearchTool(Tool):
    """Web search tool with shared browser session"""
    
    name = "web_search"
    description = "Search the web with context awareness and session persistence"
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query",
        }
    }
    output_type = "string"
    
    def __init__(self, browser_session: SharedBrowserSession = None, **kwargs):
        super().__init__(**kwargs)
        self.browser_session = browser_session or SharedBrowserSession()
    
    def forward(self, query: str) -> str:
        """Perform web search with context"""
        # Get search context for continuity
        context = self.browser_session.get_search_context()
        
        # Perform search
        results = self.browser_session.search_web(query)
        
        # Add context if available
        if "No recent searches" not in context:
            results = f"{context}\n\nCurrent search:\n{results}"
        
        return results


class StatefulWebVisitTool(Tool):
    """Web page visit tool with shared browser session"""
    
    name = "visit_webpage"
    description = "Visit a webpage with browsing context and session persistence"
    inputs = {
        "url": {
            "type": "string", 
            "description": "URL to visit",
        }
    }
    output_type = "string"
    
    def __init__(self, browser_session: SharedBrowserSession = None, **kwargs):
        super().__init__(**kwargs)
        self.browser_session = browser_session or SharedBrowserSession()
    
    def forward(self, url: str) -> str:
        """Visit webpage with context"""
        # Get browsing context
        context = self.browser_session.get_browse_context()
        
        # Visit page
        result = self.browser_session.visit_page(url)
        
        # Add context if available
        if "No browsing history" not in context:
            result = f"{context}\n\nCurrent page:\n{result}"
        
        return result


class BrowserNavigationTool(Tool):
    """Browser navigation tool for shared session"""
    
    name = "browser_navigate"
    description = "Navigate browser pages (up/down/stats)"
    inputs = {
        "action": {
            "type": "string",
            "description": "Navigation action: 'up', 'down', 'stats', or 'context'",
        }
    }
    output_type = "string"
    
    def __init__(self, browser_session: SharedBrowserSession = None, **kwargs):
        super().__init__(**kwargs)
        self.browser_session = browser_session or SharedBrowserSession()
    
    def forward(self, action: str) -> str:
        """Navigate browser session"""
        action = action.lower().strip()
        
        if action == "up":
            return self.browser_session.page_up()
        elif action == "down":
            return self.browser_session.page_down()
        elif action == "stats":
            stats = self.browser_session.get_session_stats()
            return f"Session stats: {stats}"
        elif action == "context":
            browse_context = self.browser_session.get_browse_context()
            search_context = self.browser_session.get_search_context()
            return f"{browse_context}\n\n{search_context}"
        else:
            return f"Unknown action: {action}. Use 'up', 'down', 'stats', or 'context'"