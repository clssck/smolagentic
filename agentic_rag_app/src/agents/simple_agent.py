"""
Simple Agent

Basic conversational agent for simple queries, greetings, and general assistance.
"""

import re
import math
from typing import Dict, Any
from .base_agent import BaseAgent


class SimpleAgent(BaseAgent):
    """Agent for simple queries, greetings, and basic interactions"""
    
    def __init__(self, **kwargs):
        """Initialize simple agent"""
        
        # Set defaults
        config = {
            "name": "simple_agent",
            "model_id": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
            "tools": [],
            "max_steps": 3,
            "temperature": 0.1,
            "max_tokens": 800,
            "description": "Simple queries, greetings, math, and basic assistance"
        }
        
        # Update with provided kwargs
        config.update(kwargs)
        
        super().__init__(**config)
    
    def can_handle(self, query: str, context: Dict[str, Any] = None) -> bool:
        """
        Determine if this agent can handle the query
        
        Simple agent handles:
        - Greetings and pleasantries
        - Simple math calculations
        - Basic questions
        - General assistance requests
        - Small talk
        """
        simple_patterns = [
            # Greetings
            r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
            r'\b(how are you|what\'s up|whats up)\b',
            
            # Math expressions (simple patterns)
            r'\d+\s*[\+\-\*\/]\s*\d+',
            r'\b(calculate|compute|solve|what is|whats)\s+\d+',
            
            # Simple questions
            r'\b(thank you|thanks|bye|goodbye)\b',
            r'\b(help|assist|support)\b',
            r'\b(who are you|what can you do)\b',
            
            # Basic conversational
            r'\b(yes|no|okay|ok|sure|maybe)\b$',
        ]
        
        query_lower = query.lower().strip()
        
        # Check patterns
        for pattern in simple_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Check for very short queries (likely simple)
        if len(query.split()) <= 3 and not any(complex_word in query_lower for complex_word in [
            "implement", "analyze", "research", "explain", "describe", "technical"
        ]):
            return True
        
        return False
    
    def get_system_prompt(self) -> str:
        """Get simple agent system prompt"""
        return """You are a Simple Agent for basic interactions and simple queries.

Your role:
- Handle greetings and pleasantries warmly
- Perform simple math calculations
- Provide brief, helpful responses to basic questions
- Assist with general inquiries
- Maintain a friendly, conversational tone

Instructions:
1. Keep responses concise and friendly
2. For math problems, show the calculation clearly
3. For greetings, respond appropriately and ask how you can help
4. For simple questions, provide direct answers
5. If a query seems complex, suggest they might need a more specialized agent

Focus on:
- Brevity and clarity
- Friendly, helpful tone
- Quick, direct responses
- Simple problem solving"""
    
    def run(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute simple query with basic processing
        """
        import time
        start_time = time.time()
        context = context or {}
        
        try:
            # Try to handle math queries directly
            math_result = self._handle_math_query(query)
            if math_result:
                execution_time = time.time() - start_time
                return {
                    "response": math_result,
                    "agent_name": self.name,
                    "execution_time": execution_time,
                    "model_used": "built-in-calculator",
                    "tools_used": ["math_calculator"],
                    "success": True,
                    "error": None,
                    "query_type": "math"
                }
            
            # Handle with LLM for other simple queries
            simple_query = f"""Simple Query: {query}

Please provide a brief, friendly response. Keep it concise and helpful."""
            
            result = super().run(simple_query, context)
            
            # Add simple agent metadata
            if result["success"]:
                result["query_type"] = self._classify_query_type(query)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            self.logger.error(f"Error in simple agent: {error_msg}")
            
            return {
                "response": f"I'm sorry, I encountered an error: {error_msg}",
                "agent_name": self.name,
                "execution_time": execution_time,
                "model_used": self.model_id,
                "tools_used": [],
                "success": False,
                "error": error_msg,
                "query_type": "error"
            }
    
    def _handle_math_query(self, query: str) -> str:
        """Handle simple math calculations directly"""
        try:
            # Look for simple math expressions
            math_patterns = [
                r'(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)',  # Addition
                r'(\d+(?:\.\d+)?)\s*\-\s*(\d+(?:\.\d+)?)',  # Subtraction
                r'(\d+(?:\.\d+)?)\s*\*\s*(\d+(?:\.\d+)?)',  # Multiplication
                r'(\d+(?:\.\d+)?)\s*\/\s*(\d+(?:\.\d+)?)',  # Division
                r'(\d+(?:\.\d+)?)\s*\^\s*(\d+(?:\.\d+)?)',  # Exponentiation
            ]
            
            for pattern in math_patterns:
                match = re.search(pattern, query)
                if match:
                    num1 = float(match.group(1))
                    num2 = float(match.group(2))
                    
                    if '+' in query:
                        result = num1 + num2
                        operation = "addition"
                    elif '-' in query:
                        result = num1 - num2
                        operation = "subtraction"
                    elif '*' in query or '×' in query:
                        result = num1 * num2
                        operation = "multiplication"
                    elif '/' in query or '÷' in query:
                        if num2 == 0:
                            return "I can't divide by zero! Please try a different calculation."
                        result = num1 / num2
                        operation = "division"
                    elif '^' in query or '**' in query:
                        result = num1 ** num2
                        operation = "exponentiation"
                    else:
                        continue
                    
                    # Format result nicely
                    if result.is_integer():
                        result = int(result)
                    else:
                        result = round(result, 6)
                    
                    return f"The {operation} of {num1} and {num2} is {result}."
            
            # Handle specific math requests
            if re.search(r'what is|whats\s+\d+', query.lower()):
                # Extract numbers and try to evaluate
                numbers = re.findall(r'\d+(?:\.\d+)?', query)
                if len(numbers) >= 2:
                    if '+' in query or 'plus' in query.lower():
                        result = sum(float(n) for n in numbers)
                        return f"The sum is {result}."
                    elif '*' in query or 'times' in query.lower():
                        result = 1
                        for n in numbers:
                            result *= float(n)
                        return f"The product is {result}."
            
            return None
            
        except Exception:
            return None
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of simple query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return "greeting"
        elif any(word in query_lower for word in ['thank', 'thanks', 'bye', 'goodbye']):
            return "closing"
        elif any(op in query for op in ['+', '-', '*', '/', '^']) or 'calculate' in query_lower:
            return "math"
        elif any(word in query_lower for word in ['help', 'assist', 'support']):
            return "help_request"
        elif query.endswith('?'):
            return "question"
        else:
            return "general"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get simple agent capabilities"""
        return {
            "math_operations": [
                "Addition (+)",
                "Subtraction (-)", 
                "Multiplication (*)",
                "Division (/)",
                "Exponentiation (^)"
            ],
            "interaction_types": [
                "Greetings and farewells",
                "Simple questions and answers",
                "Basic assistance requests",
                "Casual conversation"
            ],
            "response_style": "Brief, friendly, and direct",
            "max_complexity": "Simple calculations and basic interactions"
        }