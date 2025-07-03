"""
Debug Agent Wrapper for Terminal Output

This module provides debug wrappers that show agent phases and tool usage
in the terminal while the agent is running.
"""

import time
import json
from typing import Any, Dict, List
from smolagents import ToolCallingAgent


class DebugAgentWrapper:
    """Wrapper that adds debug output to agent execution."""
    
    def __init__(self, agent: ToolCallingAgent, name: str = "Agent"):
        """Initialize debug wrapper.
        
        Args:
            agent: The agent to wrap
            name: Name for debug output
        """
        self.agent = agent
        self.name = name
        self.step_count = 0
        self.start_time = None
        
    def _debug_print(self, message: str, level: str = "INFO", indent: int = 0):
        """Print formatted debug message."""
        emoji_map = {
            "PHASE": "🔄",
            "TOOL": "🛠️", 
            "RESULT": "📊",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "INFO": "ℹ️",
            "THINKING": "🤔",
            "PLANNING": "📋",
            "EXECUTING": "⚡",
            "COMPLETE": "🎉"
        }
        
        emoji = emoji_map.get(level, "•")
        indent_str = "  " * indent
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {emoji} {indent_str}{message}")
        
    def _monitor_agent_step(self, step_info: Dict[str, Any]):
        """Monitor and display agent step information."""
        step_type = step_info.get('type', 'unknown')
        
        if step_type == 'planning':
            self._debug_print(f"{self.name} - Planning Phase", "PLANNING")
            if 'content' in step_info:
                content = step_info['content'][:100] + "..." if len(step_info['content']) > 100 else step_info['content']
                self._debug_print(f"Plan: {content}", "INFO", 1)
                
        elif step_type == 'tool_call':
            tool_name = step_info.get('tool_name', 'unknown')
            self._debug_print(f"{self.name} - Using Tool: {tool_name}", "TOOL")
            
            if 'arguments' in step_info:
                args = step_info['arguments']
                if isinstance(args, dict):
                    for key, value in args.items():
                        if isinstance(value, str) and len(value) > 50:
                            value = value[:50] + "..."
                        self._debug_print(f"{key}: {value}", "INFO", 1)
                        
        elif step_type == 'tool_result':
            tool_name = step_info.get('tool_name', 'unknown')
            self._debug_print(f"{self.name} - Tool Result from {tool_name}", "RESULT")
            
            if 'result' in step_info:
                result = str(step_info['result'])
                if len(result) > 200:
                    result = result[:200] + "..."
                self._debug_print(f"Result: {result}", "INFO", 1)
                
        elif step_type == 'thinking':
            self._debug_print(f"{self.name} - Thinking...", "THINKING")
            
        elif step_type == 'final_answer':
            self._debug_print(f"{self.name} - Generating Final Answer", "COMPLETE")
            
    def run(self, query: str, **kwargs) -> Any:
        """Run agent with debug output."""
        self.start_time = time.time()
        self.step_count = 0
        
        self._debug_print(f"Starting {self.name}", "PHASE")
        self._debug_print(f"Query: {query}", "INFO", 1)
        
        try:
            # Check if agent has streaming capability
            if hasattr(self.agent, '_run_stream'):
                return self._run_with_streaming_debug(query, **kwargs)
            else:
                return self._run_with_basic_debug(query, **kwargs)
                
        except Exception as e:
            elapsed = time.time() - self.start_time
            self._debug_print(f"{self.name} failed after {elapsed:.2f}s: {e}", "ERROR")
            raise
            
    def _run_with_streaming_debug(self, query: str, **kwargs):
        """Run with streaming debug output."""
        self._debug_print(f"{self.name} - Using streaming execution", "INFO")
        
        result = None
        step_count = 0
        
        try:
            # Run the agent and capture the final result
            for event in self.agent._run_stream(query, **kwargs):
                step_count += 1
                
                # Try to extract useful information from the event
                if hasattr(event, 'type'):
                    event_type = event.type
                    
                    if event_type == 'step':
                        self._debug_print(f"Step {step_count}", "PHASE")
                        
                    elif event_type == 'tool_call':
                        if hasattr(event, 'tool_name'):
                            self._debug_print(f"Calling tool: {event.tool_name}", "TOOL")
                            
                    elif event_type == 'final_answer':
                        self._debug_print("Generating final answer", "COMPLETE")
                        if hasattr(event, 'content'):
                            result = event.content
                            
                # For other event types, just show basic info
                elif hasattr(event, '__dict__'):
                    event_info = str(event)[:100] + "..." if len(str(event)) > 100 else str(event)
                    self._debug_print(f"Event: {event_info}", "INFO", 1)
                    
            # If we didn't capture the result from streaming, try to get it normally
            if result is None:
                result = self.agent.run(query, **kwargs)
                
        except Exception as e:
            self._debug_print(f"Streaming failed, falling back to normal execution: {e}", "ERROR")
            result = self.agent.run(query, **kwargs)
            
        elapsed = time.time() - self.start_time
        self._debug_print(f"{self.name} completed in {elapsed:.2f}s after {step_count} steps", "SUCCESS")
        
        return result
        
    def _run_with_basic_debug(self, query: str, **kwargs):
        """Run with basic debug output."""
        self._debug_print(f"{self.name} - Using standard execution", "INFO")
        
        result = self.agent.run(query, **kwargs)
        
        elapsed = time.time() - self.start_time
        self._debug_print(f"{self.name} completed in {elapsed:.2f}s", "SUCCESS")
        
        return result


class DebugManagerSystem:
    """Debug wrapper for the manager agent system."""
    
    def __init__(self, manager_system):
        """Initialize debug manager.
        
        Args:
            manager_system: The manager system to wrap
        """
        self.manager_system = manager_system
        self.query_count = 0
        
    def _debug_print(self, message: str, level: str = "INFO"):
        """Print debug message with formatting."""
        emoji_map = {
            "START": "🚀",
            "ROUTING": "🔀", 
            "AGENT": "🤖",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "INFO": "ℹ️"
        }
        
        emoji = emoji_map.get(level, "•")
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {emoji} MANAGER: {message}")
        
    def run_query(self, query: str, agent_name: str = None, **kwargs):
        """Run query with debug output."""
        self.query_count += 1
        start_time = time.time()
        
        self._debug_print(f"Query #{self.query_count}: {query}", "START")
        
        if agent_name:
            self._debug_print(f"Routing to specific agent: {agent_name}", "ROUTING")
        else:
            self._debug_print("Using intelligent routing", "ROUTING")
            
        try:
            # If the manager system has a debug-capable agent, wrap it
            if hasattr(self.manager_system, 'manager_agent'):
                original_agent = self.manager_system.manager_agent
                debug_agent = DebugAgentWrapper(original_agent, "Manager Agent")
                
                # Temporarily replace the agent
                self.manager_system.manager_agent = debug_agent
                result = self.manager_system.run_query(query, agent_name=agent_name, **kwargs)
                
                # Restore original agent
                self.manager_system.manager_agent = original_agent
            else:
                result = self.manager_system.run_query(query, agent_name=agent_name, **kwargs)
                
            elapsed = time.time() - start_time
            self._debug_print(f"Query completed in {elapsed:.2f}s", "SUCCESS")
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            self._debug_print(f"Query failed after {elapsed:.2f}s: {e}", "ERROR")
            raise
            
    def __getattr__(self, name):
        """Delegate other methods to the wrapped manager system."""
        return getattr(self.manager_system, name)


def create_debug_manager(manager_system):
    """Create a debug-enabled manager system."""
    return DebugManagerSystem(manager_system)