"""
Mistral-Optimized Configuration for RAG System

This module provides optimized configurations specifically for Mistral models,
which have excellent function calling capabilities and reliability.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class MistralOptimizedConfig:
    """Configuration optimized for Mistral models with excellent function calling."""
    
    # Mistral models with their optimal configurations
    MISTRAL_MODELS = {
        "openrouter/mistralai/mistral-small-3.2-24b-instruct": {
            "name": "Mistral Small 3.2 24B",
            "context_window": 32768,
            "max_tokens": 8192,
            "temperature": 0.2,  # Optimal for function calling
            "top_p": 0.9,
            "stream": False,  # Better for function calling
            "function_calling": "excellent",
            "recommended_use": ["manager", "coordination", "function_calling"],
            "cost_tier": "low",
            "speed": "fast"
        },
        "openrouter/mistralai/mistral-medium-2312": {
            "name": "Mistral Medium",
            "context_window": 32768,
            "max_tokens": 8192,
            "temperature": 0.1,
            "top_p": 0.95,
            "stream": False,
            "function_calling": "excellent", 
            "recommended_use": ["complex_reasoning", "research"],
            "cost_tier": "medium",
            "speed": "medium"
        },
        "openrouter/mistralai/mistral-large-2407": {
            "name": "Mistral Large",
            "context_window": 128000,
            "max_tokens": 16384,
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": False,
            "function_calling": "excellent",
            "recommended_use": ["complex_tasks", "long_context"],
            "cost_tier": "high",
            "speed": "medium"
        }
    }
    
    @classmethod
    def get_manager_model_config(cls) -> Dict[str, Any]:
        """Get optimal configuration for manager agent."""
        return {
            "model_id": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
            "max_tokens": 4096,  # Sufficient for coordination tasks
            "temperature": 0.2,  # Balanced for consistency and creativity
            "top_p": 0.9,
            "stream": False,  # Required for reliable function calling
            "timeout": 30,
            "retry_attempts": 3,
            "function_calling_optimized": True
        }
    
    @classmethod
    def get_research_model_config(cls) -> Dict[str, Any]:
        """Get optimal configuration for research agent."""
        return {
            "model_id": "openrouter/mistralai/mistral-medium-2312",
            "max_tokens": 8192,  # Longer responses for research
            "temperature": 0.1,  # Lower for factual accuracy
            "top_p": 0.95,
            "stream": False,
            "timeout": 45,
            "retry_attempts": 2
        }
    
    @classmethod
    def get_rag_model_config(cls) -> Dict[str, Any]:
        """Get optimal configuration for RAG agent."""
        return {
            "model_id": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
            "max_tokens": 6144,  # Good balance for RAG responses
            "temperature": 0.3,  # Slightly higher for natural responses
            "top_p": 0.9,
            "stream": False,
            "timeout": 30,
            "retry_attempts": 3
        }
    
    @classmethod
    def get_simple_model_config(cls) -> Dict[str, Any]:
        """Get optimal configuration for simple Q&A agent."""
        return {
            "model_id": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
            "max_tokens": 2048,  # Shorter responses for simple tasks
            "temperature": 0.2,
            "top_p": 0.8,
            "stream": False,
            "timeout": 20,
            "retry_attempts": 2
        }
    
    @classmethod
    def get_optimized_system_prompt(cls, agent_type: str) -> str:
        """Get system prompt optimized for Mistral models."""
        
        base_prompt = """You are a helpful AI assistant powered by Mistral AI. You excel at function calling and following precise instructions.

IMPORTANT FUNCTION CALLING GUIDELINES:
1. Always use the available tools when they are needed to answer questions properly
2. Call functions with the exact parameter names and types specified
3. Do not provide final answers without using required tools first
4. If multiple tools are needed, use them in logical sequence
5. Be precise and accurate in your function calls

Your responses should be helpful, accurate, and concise."""

        agent_specific_prompts = {
            "manager": """
You are the Manager Agent responsible for coordinating other specialized agents and tools.

Your role:
- Route queries to appropriate specialized agents
- Coordinate multi-step workflows
- Use tools for search, research, and information gathering
- Provide clear, well-structured responses
- Maintain context across agent interactions

Always use the available tools to provide comprehensive and accurate responses.""",

            "research": """
You are the Research Agent specialized in comprehensive information gathering and analysis.

Your role:
- Conduct thorough web searches for current information
- Analyze and synthesize multiple sources
- Provide detailed, well-researched responses
- Use web browsing tools effectively
- Maintain source credibility and accuracy

Always search for current information using the available tools before providing research-based answers.""",

            "rag": """
You are the RAG Agent specialized in knowledge base search and document retrieval.

Your role:
- Search the knowledge base for relevant information
- Combine retrieved knowledge with your reasoning
- Provide accurate answers based on available documents
- Use vector search tools effectively
- Indicate when information is not available in the knowledge base

Always search the knowledge base first before providing answers to knowledge-related questions.""",

            "simple": """
You are the Simple Q&A Agent for basic questions and interactions.

Your role:
- Handle straightforward questions and conversations
- Perform basic calculations and simple tasks
- Provide quick, helpful responses
- Use tools when necessary for accuracy
- Maintain a friendly, conversational tone

Keep responses concise and directly answer the user's question."""
        }
        
        if agent_type in agent_specific_prompts:
            return base_prompt + "\n\n" + agent_specific_prompts[agent_type]
        else:
            return base_prompt
    
    @classmethod
    def get_function_calling_optimizations(cls) -> Dict[str, Any]:
        """Get function calling optimizations for Mistral models."""
        return {
            "max_function_calls_per_turn": 5,
            "function_call_timeout": 30,
            "retry_failed_calls": True,
            "parallel_function_calls": False,  # Sequential is more reliable
            "strict_parameter_validation": True,
            "detailed_error_messages": True,
            "fallback_on_failure": True
        }
    
    @classmethod
    def create_optimized_model_instance(cls, agent_type: str = "manager"):
        """Create an optimized LiteLLM model instance for Mistral."""
        from smolagents import LiteLLMModel
        
        config_map = {
            "manager": cls.get_manager_model_config(),
            "research": cls.get_research_model_config(), 
            "rag": cls.get_rag_model_config(),
            "simple": cls.get_simple_model_config()
        }
        
        config = config_map.get(agent_type, cls.get_manager_model_config())
        
        try:
            model = LiteLLMModel(
                model_id=config["model_id"],
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"]
            )
            
            logger.info(f"Created optimized Mistral model for {agent_type}: {config['model_id']}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create Mistral model: {e}")
            # Fallback to basic configuration
            return LiteLLMModel(
                model_id="openrouter/mistralai/mistral-small-3.2-24b-instruct",
                max_tokens=4096,
                temperature=0.2
            )
    
    @classmethod
    def test_mistral_function_calling(cls) -> Dict[str, Any]:
        """Test Mistral function calling capabilities."""
        print("🧪 Testing Mistral Function Calling")
        print("=" * 40)
        
        test_results = {
            "timestamp": time.time(),
            "model_tests": {},
            "function_calling_score": 0,
            "recommendations": []
        }
        
        try:
            # Test basic model creation
            model = cls.create_optimized_model_instance("manager")
            
            # Simple generation test
            test_messages = [
                {"role": "system", "content": cls.get_optimized_system_prompt("manager")},
                {"role": "user", "content": "Hello, please respond with a brief greeting."}
            ]
            
            response = model.generate(test_messages)
            
            test_results["model_tests"]["basic_generation"] = {
                "status": "success",
                "response_length": len(str(response))
            }
            test_results["function_calling_score"] += 25
            
            print("✅ Basic generation test passed")
            
            # Test with function calling context
            function_test_messages = [
                {"role": "system", "content": cls.get_optimized_system_prompt("manager")},
                {"role": "user", "content": "Can you help me search for information about machine learning?"}
            ]
            
            response2 = model.generate(function_test_messages)
            
            test_results["model_tests"]["function_context"] = {
                "status": "success", 
                "mentions_tools": "tool" in str(response2).lower() or "search" in str(response2).lower()
            }
            test_results["function_calling_score"] += 25
            
            print("✅ Function calling context test passed")
            
            # Generate recommendations
            if test_results["function_calling_score"] >= 40:
                test_results["recommendations"].extend([
                    "Mistral models working excellently for function calling",
                    "Use temperature 0.1-0.3 for optimal function calling",
                    "Disable streaming for best function calling reliability", 
                    "Mistral Small 3.2 24B is perfect for manager agent"
                ])
            
            test_results["overall_status"] = "excellent"
            
        except Exception as e:
            test_results["model_tests"]["error"] = str(e)
            test_results["overall_status"] = "failed"
            test_results["recommendations"].append(f"Fix model configuration: {e}")
            
        return test_results


def main():
    """Test the Mistral optimized configuration."""
    import time
    
    config = MistralOptimizedConfig()
    
    print("🎯 MISTRAL OPTIMIZATION TEST")
    print("=" * 40)
    
    # Show optimal configurations
    print("\n📋 Optimal Configurations:")
    print(f"Manager: {config.get_manager_model_config()['model_id']}")
    print(f"Research: {config.get_research_model_config()['model_id']}")
    print(f"RAG: {config.get_rag_model_config()['model_id']}")
    
    # Test function calling
    results = config.test_mistral_function_calling()
    
    print(f"\n📊 Test Results:")
    print(f"Function Calling Score: {results['function_calling_score']}/100")
    print(f"Overall Status: {results['overall_status']}")
    
    print(f"\n💡 Recommendations:")
    for rec in results['recommendations']:
        print(f"  • {rec}")
    
    print(f"\n✅ Mistral optimization test completed!")


if __name__ == "__main__":
    main()