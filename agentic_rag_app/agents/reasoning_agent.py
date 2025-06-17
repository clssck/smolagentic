import logging
import re
from typing import Dict, Any, Optional, Tuple
from llama_index.core.llms import ChatMessage, MessageRole
from models.factory import get_model_factory
from utils.config_loader import get_config_loader

logger = logging.getLogger(__name__)

class ReasoningAgent:
    """
    Agent that uses Qwen 3's thinking mode for complex reasoning tasks.
    Extracts and provides both reasoning steps and final answers.
    """
    
    def __init__(self, model_name: str = "qwen3-32b"):
        self.config = get_config_loader()
        self.model_factory = get_model_factory()
        self.model_name = model_name
        self.llm = self._setup_reasoning_model()
        
    def _setup_reasoning_model(self):
        """Setup LLM with reasoning-optimized parameters"""
        # Get base model from factory
        base_model = self.model_factory.get_chat_model(self.model_name)
        
        # Override with reasoning-specific parameters (if supported)
        try:
            base_model.temperature = 0.6
        except:
            pass
        
        try:
            base_model.max_tokens = 8000
        except:
            pass
        
        # Store parameters for reference
        self._reasoning_params = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0,
            "max_tokens": 8000
        }
        
        return base_model
    
    def think(self, query: str, enable_thinking: bool = True) -> Dict[str, Any]:
        """
        Process a query using thinking mode and extract reasoning steps.
        
        Args:
            query: The question or problem to solve
            enable_thinking: Whether to enable thinking mode (default True)
            
        Returns:
            Dict containing reasoning steps and final answer
        """
        try:
            # Construct system prompt for reasoning
            system_prompt = self._get_reasoning_system_prompt(enable_thinking)
            
            # Create messages
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=query)
            ]
            
            # Get response from model
            response = self.llm.chat(messages)
            response_text = str(response)
            
            # Parse thinking and final answer
            thinking_content, final_answer = self._parse_response(response_text)
            
            return {
                "query": query,
                "thinking_enabled": enable_thinking,
                "thinking_steps": thinking_content,
                "final_answer": final_answer,
                "full_response": response_text,
                "model_used": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error in reasoning: {e}")
            return {
                "query": query,
                "thinking_enabled": enable_thinking,
                "thinking_steps": None,
                "final_answer": f"Error occurred during reasoning: {str(e)}",
                "full_response": None,
                "model_used": self.model_name
            }
    
    def _get_reasoning_system_prompt(self, enable_thinking: bool) -> str:
        """Generate system prompt for reasoning mode"""
        base_prompt = """You are an intelligent reasoning assistant. Your task is to think through problems step by step and provide well-reasoned answers."""
        
        if enable_thinking:
            thinking_prompt = """

When solving complex problems:
1. Use <think> tags to show your reasoning process
2. Break down the problem into smaller steps
3. Consider multiple approaches and evaluate them
4. Show your work and thought process clearly
5. After thinking, provide a clear, concise final answer

Example format:
<think>
Let me break this down step by step...
First, I need to understand what's being asked...
Then I should consider...
Actually, let me think about this differently...
So the key insight is...
Therefore, the answer should be...
</think>

Based on my analysis, the answer is: [clear final answer]
"""
            return base_prompt + thinking_prompt
        else:
            return base_prompt + "\n\nProvide direct, concise answers without showing intermediate reasoning steps."
    
    def _parse_response(self, response: str) -> Tuple[Optional[str], str]:
        """
        Extract thinking content and final answer from response.
        
        Returns:
            Tuple of (thinking_content, final_answer)
        """
        # Look for <think>...</think> blocks
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, response, re.DOTALL)
        
        if think_matches:
            # Extract thinking content (join multiple think blocks if any)
            thinking_content = '\n\n'.join(match.strip() for match in think_matches)
            
            # Remove think blocks to get final answer
            final_answer = re.sub(think_pattern, '', response, flags=re.DOTALL).strip()
            
            # Clean up final answer
            final_answer = re.sub(r'\n\s*\n', '\n\n', final_answer)  # Remove extra blank lines
            final_answer = final_answer.strip()
            
        else:
            # No thinking blocks found, entire response is final answer
            thinking_content = None
            final_answer = response.strip()
        
        return thinking_content, final_answer
    
    def solve_problem(self, problem: str, domain: str = "general") -> Dict[str, Any]:
        """
        Solve a specific problem with domain-aware reasoning.
        
        Args:
            problem: The problem to solve
            domain: Problem domain (math, coding, logic, etc.)
        """
        domain_prompts = {
            "math": "This is a mathematical problem. Show all calculations and reasoning steps.",
            "coding": "This is a programming problem. Consider algorithms, complexity, and edge cases.",
            "logic": "This is a logical reasoning problem. Examine premises and conclusions carefully.",
            "analysis": "This requires analytical thinking. Consider multiple perspectives and evidence.",
            "general": "Think through this problem systematically."
        }
        
        enhanced_query = f"{domain_prompts.get(domain, domain_prompts['general'])}\n\nProblem: {problem}"
        
        result = self.think(enhanced_query, enable_thinking=True)
        result["domain"] = domain
        
        return result
    
    def compare_approaches(self, problem: str, approaches: list) -> Dict[str, Any]:
        """
        Compare different approaches to solving a problem.
        """
        comparison_query = f"""
Problem: {problem}

Consider these approaches:
{chr(10).join(f"{i+1}. {approach}" for i, approach in enumerate(approaches))}

Analyze each approach, compare their strengths and weaknesses, and recommend the best one.
"""
        
        result = self.think(comparison_query, enable_thinking=True)
        result["approaches_compared"] = approaches
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current reasoning model"""
        info = {
            "model_name": self.model_name,
            "supports_thinking": True
        }
        
        # Add reasoning parameters
        if hasattr(self, '_reasoning_params'):
            info.update(self._reasoning_params)
        
        # Add actual model attributes if available
        for attr in ['temperature', 'max_tokens']:
            if hasattr(self.llm, attr):
                info[f"actual_{attr}"] = getattr(self.llm, attr)
        
        return info

# Global instance
_reasoning_agent = None

def get_reasoning_agent(model_name: str = "qwen3-32b") -> ReasoningAgent:
    """Get global reasoning agent instance"""
    global _reasoning_agent
    if _reasoning_agent is None or _reasoning_agent.model_name != model_name:
        _reasoning_agent = ReasoningAgent(model_name)
    return _reasoning_agent