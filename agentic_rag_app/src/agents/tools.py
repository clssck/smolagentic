"""
Shared Tools for Agents

Collection of tools that can be used across different agents.
"""

import json
import requests
import time
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Base class for all tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool"""
        pass
    
    def get_info(self) -> Dict[str, str]:
        """Get tool information"""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__
        }


class SimpleChatTool(BaseTool):
    """Simple chat tool for basic interactions"""
    
    def __init__(self):
        super().__init__(
            name="simple_chat",
            description="Handle simple greetings, basic math, and straightforward questions"
        )
    
    def __call__(self, query: str) -> str:
        """Handle simple chat queries"""
        query_lower = query.lower().strip()
        
        # Greetings
        if any(greeting in query_lower for greeting in ['hello', 'hi', 'hey']):
            return "Hello! How can I help you today?"
        
        # Math operations
        try:
            # Simple math evaluation (be careful with eval in production)
            if any(op in query for op in ['+', '-', '*', '/']):
                # Basic validation
                if all(char in '0123456789+-*/.() ' for char in query):
                    result = eval(query)
                    return f"The answer is {result}"
        except:
            pass
        
        # Thanks
        if any(thanks in query_lower for thanks in ['thanks', 'thank you']):
            return "You're welcome! Is there anything else I can help you with?"
        
        # Default response
        return f"I can help with simple questions. You asked: '{query}'"


class FileReaderTool(BaseTool):
    """Tool for reading files"""
    
    def __init__(self):
        super().__init__(
            name="file_reader",
            description="Read content from files"
        )
    
    def __call__(self, file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Read file content"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            return {
                "success": True,
                "content": content,
                "file_path": file_path,
                "size": len(content),
                "lines": len(content.splitlines()) if content else 0
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path,
                "content": None
            }


class FileWriterTool(BaseTool):
    """Tool for writing files"""
    
    def __init__(self):
        super().__init__(
            name="file_writer",
            description="Write content to files"
        )
    
    def __call__(self, file_path: str, content: str, encoding: str = "utf-8", mode: str = "w") -> Dict[str, Any]:
        """Write content to file"""
        try:
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)
            
            return {
                "success": True,
                "file_path": file_path,
                "bytes_written": len(content.encode(encoding)),
                "mode": mode
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }


class HTTPRequestTool(BaseTool):
    """Tool for making HTTP requests"""
    
    def __init__(self):
        super().__init__(
            name="http_request",
            description="Make HTTP requests to web APIs"
        )
    
    def __call__(
        self, 
        url: str, 
        method: str = "GET", 
        headers: Dict[str, str] = None,
        data: Union[str, Dict] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Make HTTP request"""
        try:
            headers = headers or {}
            
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method.upper() == "POST":
                if isinstance(data, dict):
                    response = requests.post(url, json=data, headers=headers, timeout=timeout)
                else:
                    response = requests.post(url, data=data, headers=headers, timeout=timeout)
            elif method.upper() == "PUT":
                if isinstance(data, dict):
                    response = requests.put(url, json=data, headers=headers, timeout=timeout)
                else:
                    response = requests.put(url, data=data, headers=headers, timeout=timeout)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, timeout=timeout)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported HTTP method: {method}",
                    "url": url
                }
            
            # Try to parse JSON response
            try:
                json_content = response.json()
            except:
                json_content = None
            
            return {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text,
                "json": json_content,
                "url": url,
                "method": method.upper()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "method": method.upper()
            }


class TimerTool(BaseTool):
    """Tool for timing operations"""
    
    def __init__(self):
        super().__init__(
            name="timer",
            description="Time operations and provide delays"
        )
        self.start_times = {}
    
    def start_timer(self, timer_name: str = "default") -> Dict[str, Any]:
        """Start a timer"""
        start_time = time.time()
        self.start_times[timer_name] = start_time
        
        return {
            "timer_name": timer_name,
            "started_at": start_time,
            "action": "started"
        }
    
    def stop_timer(self, timer_name: str = "default") -> Dict[str, Any]:
        """Stop a timer and get elapsed time"""
        if timer_name not in self.start_times:
            return {
                "error": f"Timer '{timer_name}' was not started",
                "timer_name": timer_name
            }
        
        start_time = self.start_times[timer_name]
        end_time = time.time()
        elapsed = end_time - start_time
        
        del self.start_times[timer_name]
        
        return {
            "timer_name": timer_name,
            "started_at": start_time,
            "ended_at": end_time,
            "elapsed_seconds": elapsed,
            "elapsed_formatted": f"{elapsed:.3f}s",
            "action": "stopped"
        }
    
    def __call__(self, action: str, timer_name: str = "default") -> Dict[str, Any]:
        """Timer tool main interface"""
        if action == "start":
            return self.start_timer(timer_name)
        elif action == "stop":
            return self.stop_timer(timer_name)
        elif action == "delay":
            # Simple delay functionality
            delay_time = float(timer_name) if timer_name.replace('.', '').isdigit() else 1.0
            time.sleep(delay_time)
            return {
                "action": "delay",
                "delay_seconds": delay_time,
                "message": f"Delayed for {delay_time} seconds"
            }
        else:
            return {
                "error": f"Unknown timer action: {action}",
                "available_actions": ["start", "stop", "delay"]
            }


class JSONProcessorTool(BaseTool):
    """Tool for processing JSON data"""
    
    def __init__(self):
        super().__init__(
            name="json_processor",
            description="Parse, validate, and manipulate JSON data"
        )
    
    def __call__(self, data: Union[str, dict], action: str = "parse") -> Dict[str, Any]:
        """Process JSON data"""
        try:
            if action == "parse":
                if isinstance(data, str):
                    parsed = json.loads(data)
                    return {
                        "success": True,
                        "parsed_data": parsed,
                        "data_type": type(parsed).__name__,
                        "action": "parse"
                    }
                else:
                    return {
                        "success": True,
                        "parsed_data": data,
                        "data_type": type(data).__name__,
                        "action": "parse",
                        "note": "Data was already parsed"
                    }
            
            elif action == "stringify":
                if isinstance(data, str):
                    # Try to parse first to validate
                    json.loads(data)
                    json_string = data
                else:
                    json_string = json.dumps(data, indent=2)
                
                return {
                    "success": True,
                    "json_string": json_string,
                    "length": len(json_string),
                    "action": "stringify"
                }
            
            elif action == "validate":
                if isinstance(data, str):
                    json.loads(data)  # This will raise an exception if invalid
                    return {
                        "success": True,
                        "valid": True,
                        "action": "validate"
                    }
                else:
                    json.dumps(data)  # This will raise an exception if not serializable
                    return {
                        "success": True,
                        "valid": True,
                        "action": "validate"
                    }
            
            elif action == "keys":
                if isinstance(data, str):
                    parsed_data = json.loads(data)
                else:
                    parsed_data = data
                
                if isinstance(parsed_data, dict):
                    return {
                        "success": True,
                        "keys": list(parsed_data.keys()),
                        "key_count": len(parsed_data),
                        "action": "keys"
                    }
                else:
                    return {
                        "success": False,
                        "error": "Data is not a JSON object (dictionary)",
                        "data_type": type(parsed_data).__name__,
                        "action": "keys"
                    }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "available_actions": ["parse", "stringify", "validate", "keys"]
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": action
            }


class TextProcessorTool(BaseTool):
    """Tool for text processing operations"""
    
    def __init__(self):
        super().__init__(
            name="text_processor",
            description="Process and analyze text content"
        )
    
    def __call__(self, text: str, action: str = "analyze") -> Dict[str, Any]:
        """Process text"""
        try:
            if action == "analyze":
                words = text.split()
                lines = text.splitlines()
                
                return {
                    "success": True,
                    "character_count": len(text),
                    "word_count": len(words),
                    "line_count": len(lines),
                    "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
                    "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
                    "action": "analyze"
                }
            
            elif action == "uppercase":
                return {
                    "success": True,
                    "result": text.upper(),
                    "action": "uppercase"
                }
            
            elif action == "lowercase":
                return {
                    "success": True,
                    "result": text.lower(),
                    "action": "lowercase"
                }
            
            elif action == "title":
                return {
                    "success": True,
                    "result": text.title(),
                    "action": "title"
                }
            
            elif action == "strip":
                return {
                    "success": True,
                    "result": text.strip(),
                    "original_length": len(text),
                    "new_length": len(text.strip()),
                    "action": "strip"
                }
            
            elif action == "reverse":
                return {
                    "success": True,
                    "result": text[::-1],
                    "action": "reverse"
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "available_actions": ["analyze", "uppercase", "lowercase", "title", "strip", "reverse"]
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": action
            }


# Tool registry for easy access
AVAILABLE_TOOLS = {
    "simple_chat": SimpleChatTool,
    "file_reader": FileReaderTool,
    "file_writer": FileWriterTool,
    "http_request": HTTPRequestTool,
    "timer": TimerTool,
    "json_processor": JSONProcessorTool,
    "text_processor": TextProcessorTool
}


def get_tool(tool_name: str) -> Optional[BaseTool]:
    """Get a tool instance by name"""
    if tool_name in AVAILABLE_TOOLS:
        return AVAILABLE_TOOLS[tool_name]()
    return None


def list_available_tools() -> List[str]:
    """List all available tools"""
    return list(AVAILABLE_TOOLS.keys())


def get_tool_info(tool_name: str) -> Optional[Dict[str, str]]:
    """Get information about a specific tool"""
    tool = get_tool(tool_name)
    return tool.get_info() if tool else None