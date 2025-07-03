"""
Enhanced Connection Pool Manager for RAG System

Provides optimized connection pooling for various services including
HTTP clients, database connections, and API clients.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, Optional, Union
from contextlib import asynccontextmanager, contextmanager

try:
    import aiohttp
    ASYNC_HTTP_AVAILABLE = True
except ImportError:
    ASYNC_HTTP_AVAILABLE = False

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    SYNC_HTTP_AVAILABLE = True
except ImportError:
    SYNC_HTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


class ConnectionPoolManager:
    """
    Centralized connection pool manager for optimized performance.
    
    Manages HTTP sessions, database connections, and API clients
    with proper lifecycle management and resource cleanup.
    """
    
    def __init__(self):
        """Initialize the connection pool manager."""
        self._async_sessions: Dict[str, aiohttp.ClientSession] = {}
        self._sync_sessions: Dict[str, requests.Session] = {}
        self._locks = {
            'async': asyncio.Lock() if ASYNC_HTTP_AVAILABLE else None,
            'sync': threading.Lock()
        }
        self._cleanup_intervals = {}
        self._last_cleanup = {}
        
        # Performance tracking
        self.pool_stats = {
            'async_sessions_created': 0,
            'sync_sessions_created': 0,
            'total_requests': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }
        
        logger.info("Connection pool manager initialized")
    
    async def get_async_session(self, 
                              pool_name: str = 'default',
                              connector_config: Optional[Dict[str, Any]] = None) -> aiohttp.ClientSession:
        """
        Get or create an async HTTP session with optimized settings.
        
        Args:
            pool_name: Name of the session pool
            connector_config: Custom connector configuration
            
        Returns:
            Configured aiohttp ClientSession
        """
        if not ASYNC_HTTP_AVAILABLE:
            raise ImportError("aiohttp is not available")
        
        async with self._locks['async']:
            if pool_name in self._async_sessions:
                session = self._async_sessions[pool_name]
                if not session.closed:
                    self.pool_stats['pool_hits'] += 1
                    return session
                else:
                    # Session is closed, remove it
                    del self._async_sessions[pool_name]
            
            # Create new optimized session
            self.pool_stats['pool_misses'] += 1
            self.pool_stats['async_sessions_created'] += 1
            
            # Default optimized connector settings (compatible with current aiohttp version)
            default_config = {
                'limit': 50,
                'limit_per_host': 10,
                'keepalive_timeout': 30,
                'enable_cleanup_closed': True,
                'use_dns_cache': True,
                'ttl_dns_cache': 60,
                'force_close': False
            }
            
            if connector_config:
                default_config.update(connector_config)
            
            connector = aiohttp.TCPConnector(**default_config)
            
            # Create session with optimized timeouts
            timeout = aiohttp.ClientTimeout(
                total=30,
                connect=10,
                sock_read=20
            )
            
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'RAG-System-Enhanced/1.0'}
            )
            
            self._async_sessions[pool_name] = session
            logger.debug(f"Created new async session pool: {pool_name}")
            
            return session
    
    def get_sync_session(self, 
                        pool_name: str = 'default',
                        session_config: Optional[Dict[str, Any]] = None) -> requests.Session:
        """
        Get or create a sync HTTP session with connection pooling.
        
        Args:
            pool_name: Name of the session pool
            session_config: Custom session configuration
            
        Returns:
            Configured requests Session
        """
        if not SYNC_HTTP_AVAILABLE:
            raise ImportError("requests is not available")
        
        with self._locks['sync']:
            if pool_name in self._sync_sessions:
                self.pool_stats['pool_hits'] += 1
                return self._sync_sessions[pool_name]
            
            # Create new optimized session
            self.pool_stats['pool_misses'] += 1
            self.pool_stats['sync_sessions_created'] += 1
            
            session = requests.Session()
            
            # Configure retry strategy (compatible with current urllib3 version)
            retry_strategy = Retry(
                total=3,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
                backoff_factor=1
            )
            
            # Configure adapters with connection pooling
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=20,
                pool_maxsize=50,
                pool_block=False
            )
            
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Set default headers
            session.headers.update({
                'User-Agent': 'RAG-System-Enhanced/1.0',
                'Connection': 'keep-alive'
            })
            
            # Apply custom configuration
            if session_config:
                for key, value in session_config.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
            
            self._sync_sessions[pool_name] = session
            logger.debug(f"Created new sync session pool: {pool_name}")
            
            return session
    
    @asynccontextmanager
    async def async_request(self, 
                           method: str, 
                           url: str, 
                           pool_name: str = 'default',
                           **kwargs):
        """
        Context manager for async HTTP requests with automatic pool management.
        
        Args:
            method: HTTP method
            url: Request URL
            pool_name: Session pool name
            **kwargs: Additional arguments for the request
        """
        session = await self.get_async_session(pool_name)
        self.pool_stats['total_requests'] += 1
        
        try:
            async with session.request(method, url, **kwargs) as response:
                yield response
        except Exception as e:
            logger.error(f"Async request failed: {e}")
            raise
    
    @contextmanager
    def sync_request(self, 
                    method: str, 
                    url: str, 
                    pool_name: str = 'default',
                    **kwargs):
        """
        Context manager for sync HTTP requests with automatic pool management.
        
        Args:
            method: HTTP method
            url: Request URL
            pool_name: Session pool name
            **kwargs: Additional arguments for the request
        """
        session = self.get_sync_session(pool_name)
        self.pool_stats['total_requests'] += 1
        
        try:
            response = session.request(method, url, **kwargs)
            yield response
        except Exception as e:
            logger.error(f"Sync request failed: {e}")
            raise
        finally:
            # Response is automatically closed by requests
            pass
    
    async def cleanup_async_sessions(self, force: bool = False):
        """
        Cleanup async sessions that are no longer needed.
        
        Args:
            force: Force cleanup of all sessions
        """
        if not ASYNC_HTTP_AVAILABLE:
            return
        
        async with self._locks['async']:
            sessions_to_remove = []
            
            for pool_name, session in self._async_sessions.items():
                if force or session.closed:
                    sessions_to_remove.append(pool_name)
                    if not session.closed:
                        await session.close()
            
            for pool_name in sessions_to_remove:
                del self._async_sessions[pool_name]
                logger.debug(f"Cleaned up async session pool: {pool_name}")
    
    def cleanup_sync_sessions(self, force: bool = False):
        """
        Cleanup sync sessions that are no longer needed.
        
        Args:
            force: Force cleanup of all sessions
        """
        if not SYNC_HTTP_AVAILABLE:
            return
        
        with self._locks['sync']:
            if force:
                for pool_name, session in self._sync_sessions.items():
                    session.close()
                    logger.debug(f"Cleaned up sync session pool: {pool_name}")
                self._sync_sessions.clear()
    
    async def close_all(self):
        """Close all connections and cleanup resources."""
        await self.cleanup_async_sessions(force=True)
        self.cleanup_sync_sessions(force=True)
        logger.info("All connection pools closed")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            **self.pool_stats,
            'active_async_sessions': len(self._async_sessions),
            'active_sync_sessions': len(self._sync_sessions),
            'pool_hit_rate_percent': (
                self.pool_stats['pool_hits'] / 
                (self.pool_stats['pool_hits'] + self.pool_stats['pool_misses']) * 100
                if (self.pool_stats['pool_hits'] + self.pool_stats['pool_misses']) > 0 else 0
            )
        }


# Global connection pool manager
_connection_pool_manager = None
_manager_lock = threading.Lock()


def get_connection_pool_manager() -> ConnectionPoolManager:
    """
    Get or create the global connection pool manager.
    
    Returns:
        ConnectionPoolManager instance
    """
    global _connection_pool_manager
    
    with _manager_lock:
        if _connection_pool_manager is None:
            _connection_pool_manager = ConnectionPoolManager()
    
    return _connection_pool_manager


async def async_http_request(method: str, url: str, pool_name: str = 'default', **kwargs):
    """
    Convenience function for async HTTP requests using connection pooling.
    
    Args:
        method: HTTP method
        url: Request URL
        pool_name: Connection pool name
        **kwargs: Additional request arguments
        
    Returns:
        Response data
    """
    manager = get_connection_pool_manager()
    
    async with manager.async_request(method, url, pool_name, **kwargs) as response:
        if response.status == 200:
            return await response.json()
        else:
            response.raise_for_status()


def sync_http_request(method: str, url: str, pool_name: str = 'default', **kwargs):
    """
    Convenience function for sync HTTP requests using connection pooling.
    
    Args:
        method: HTTP method
        url: Request URL
        pool_name: Connection pool name
        **kwargs: Additional request arguments
        
    Returns:
        Response data
    """
    manager = get_connection_pool_manager()
    
    with manager.sync_request(method, url, pool_name, **kwargs) as response:
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()


async def cleanup_connection_pools():
    """Cleanup all connection pools."""
    manager = get_connection_pool_manager()
    await manager.close_all()