from typing import Dict, Any
from llama_index.core.tools import BaseTool, ToolMetadata, ToolOutput
from agents.reasoning_agent import get_reasoning_agent

class ReasoningTool(BaseTool):
    """
    Tool that provides access to Qwen 3's thinking mode for complex reasoning tasks.
    Can be integrated into the main RAG agent for advanced problem solving.
    """
    
    def __init__(self, model_name: str = "qwen3-32b-reasoning"):
        self.reasoning_agent = get_reasoning_agent(model_name)
        self._metadata = ToolMetadata(
            name="deep_reasoning",
            description="""Use this tool for complex reasoning tasks that require step-by-step thinking.
            
Perfect for:
- Mathematical problems and calculations
- Logical puzzles and reasoning
- Complex analysis requiring multiple steps
- Comparing different approaches or solutions
- Problems that benefit from showing work/thinking process

Input should be a clear problem statement or question that needs deep analysis."""
        )
    
    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata
    
    def call(self, query: str, domain: str = "general") -> str:
        """
        Execute reasoning on the given query.
        
        Args:
            query: The problem or question to reason through
            domain: Optional domain hint (math, coding, logic, analysis, general)
        """
        try:
            result = self.reasoning_agent.solve_problem(query, domain)
            
            # Format response for the agent
            if result["thinking_steps"]:
                response = f"""**Reasoning Process:**
{result['thinking_steps']}

**Final Answer:**
{result['final_answer']}"""
            else:
                response = result['final_answer']
            
            return response
            
        except Exception as e:
            return f"Error in reasoning: {str(e)}"
    
    def __call__(self, input: Any) -> ToolOutput:
        """Required by BaseTool interface"""
        if isinstance(input, str):
            result = self.call(input)
        elif isinstance(input, dict):
            query = input.get("query", str(input))
            domain = input.get("domain", "general")
            result = self.call(query, domain)
        else:
            result = self.call(str(input))
        
        return ToolOutput(
            content=result,
            tool_name=self.metadata.name,
            raw_input=input,
            raw_output=result
        )

class ComparisonTool(BaseTool):
    """
    Tool for comparing different approaches or solutions to a problem.
    """
    
    def __init__(self, model_name: str = "qwen3-32b-reasoning"):
        self.reasoning_agent = get_reasoning_agent(model_name)
        self._metadata = ToolMetadata(
            name="compare_approaches",
            description="""Compare different approaches, methods, or solutions to a problem.
            
Use when you need to:
- Evaluate multiple solution strategies
- Compare pros and cons of different approaches
- Analyze trade-offs between options
- Make recommendations based on analysis

Provide the problem and a list of approaches to compare."""
        )
    
    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata
    
    def call(self, problem: str, approaches: str) -> str:
        """
        Compare different approaches to solving a problem.
        
        Args:
            problem: The problem statement
            approaches: Comma-separated list of approaches to compare
        """
        try:
            # Parse approaches
            approach_list = [approach.strip() for approach in approaches.split(',')]
            
            result = self.reasoning_agent.compare_approaches(problem, approach_list)
            
            response = f"""**Problem Analysis:**
{result['thinking_steps'] if result['thinking_steps'] else 'Direct comparison performed'}

**Comparison Result:**
{result['final_answer']}"""
            
            return response
            
        except Exception as e:
            return f"Error in comparison: {str(e)}"
    
    def __call__(self, input: Any) -> ToolOutput:
        """Required by BaseTool interface"""
        if isinstance(input, str):
            # Assume single string is the problem, no approaches specified
            result = f"Error: Please provide both problem and approaches. Got: {input}"
        elif isinstance(input, dict):
            problem = input.get("problem", "")
            approaches = input.get("approaches", "")
            if problem and approaches:
                result = self.call(problem, approaches)
            else:
                result = f"Error: Missing problem or approaches in input: {input}"
        else:
            result = f"Error: Invalid input format: {input}"
        
        return ToolOutput(
            content=result,
            tool_name=self.metadata.name,
            raw_input=input,
            raw_output=result
        )

def get_reasoning_tools(model_name: str = "qwen3-32b-reasoning") -> list:
    """Get list of reasoning tools for integration into agents"""
    return [
        ReasoningTool(model_name),
        ComparisonTool(model_name)
    ]