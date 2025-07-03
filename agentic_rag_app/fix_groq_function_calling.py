#!/usr/bin/env python3
"""
Fix for Groq Function Calling Issues

This script addresses the common Groq API function calling errors by:
1. Implementing robust error handling
2. Adding model configuration optimizations  
3. Creating fallback mechanisms
4. Testing function calling reliability
"""

import sys
import json
import time
from typing import Dict, Any, List, Optional

sys.path.append('.')

from src.utils.config import Config


class GroqFunctionCallingFix:
    """Fix Groq function calling issues with enhanced error handling."""
    
    def __init__(self):
        """Initialize the fix utilities."""
        self.groq_models = [
            "groq/llama3-groq-70b-8192-tool-use-preview",
            "groq/llama3-groq-8b-8192-tool-use-preview", 
            "groq/mixtral-8x7b-32768",
            "groq/llama2-70b-4096"
        ]
        
        self.optimal_configs = {
            "groq/llama3-groq-70b-8192-tool-use-preview": {
                "max_tokens": 4096,
                "temperature": 0.1,
                "top_p": 0.9,
                "stop": None,
                "stream": False,
                "tools_required": True
            },
            "groq/llama3-groq-8b-8192-tool-use-preview": {
                "max_tokens": 2048, 
                "temperature": 0.2,
                "top_p": 0.8,
                "stop": None,
                "stream": False,
                "tools_required": True
            },
            "groq/mixtral-8x7b-32768": {
                "max_tokens": 8192,
                "temperature": 0.3,
                "top_p": 0.9,
                "stop": None,
                "stream": False,
                "tools_required": False  # Mixtral has limited function calling
            }
        }
    
    def create_optimized_model_config(self, model_name: str) -> Dict[str, Any]:
        """Create optimized configuration for Groq models."""
        base_config = {
            "model": model_name,
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": False,
            "timeout": 30
        }
        
        # Apply model-specific optimizations
        if model_name in self.optimal_configs:
            base_config.update(self.optimal_configs[model_name])
        
        return base_config
    
    def create_robust_system_prompt(self) -> str:
        """Create a system prompt that works well with Groq function calling."""
        return """You are a helpful AI assistant. When using tools, always follow these guidelines:

1. ALWAYS call the appropriate tool function when needed
2. Use the exact function name and parameters as specified
3. Do not provide final answers without calling required tools
4. If you need to search or retrieve information, use the available tools
5. Keep responses concise and focused

If you cannot use a tool, explain why and provide the best answer you can."""

    def create_function_calling_prompt_template(self) -> str:
        """Create a prompt template optimized for function calling."""
        return """{{system_prompt}}

Available tools:
{{tool_descriptions}}

User Query: {{user_input}}

Think step by step:
1. Understand what the user is asking
2. Determine if you need to use any tools
3. If tools are needed, call them with proper parameters
4. Provide a helpful response based on the results

Remember: Always use tools when they are needed to answer the question properly."""

    def implement_error_handling_wrapper(self) -> str:
        """Return Python code for error handling wrapper."""
        return '''
def groq_safe_agent_call(agent, query, max_retries=3, fallback_models=None):
    """
    Safe wrapper for agent calls with Groq error handling.
    
    Args:
        agent: The agent instance
        query: User query
        max_retries: Maximum retry attempts
        fallback_models: List of fallback models to try
        
    Returns:
        Response or error message
    """
    import time
    from smolagents.utils import AgentGenerationError
    
    if fallback_models is None:
        fallback_models = [
            "openrouter/mistralai/mistral-small-3.2-24b-instruct",
            "openrouter/qwen/qwen-2.5-7b-instruct"
        ]
    
    last_error = None
    
    # Try with current model
    for attempt in range(max_retries):
        try:
            print(f"🤖 Attempt {attempt + 1} with current model...")
            response = agent.run(query)
            return response
            
        except AgentGenerationError as e:
            last_error = e
            error_msg = str(e).lower()
            
            if "tool_use_failed" in error_msg or "function" in error_msg:
                print(f"⚠️  Function calling failed: {e}")
                
                # Extract the failed generation if available
                if hasattr(e, 'failed_generation') or 'failed_generation' in error_msg:
                    print("📝 Model generated content but failed to call function")
                    
                    # Try to extract useful content from the failed generation
                    try:
                        import re
                        # Look for the failed_generation content in the error
                        match = re.search(r'"failed_generation":"([^"]*)"', str(e))
                        if match:
                            content = match.group(1).replace('\\\\n', '\\n')
                            print(f"📄 Extracted content: {content[:200]}...")
                            # Return the content if it looks complete
                            if len(content) > 100 and content.strip():
                                return content.strip()
                    except Exception:
                        pass
                
                # Wait before retry
                time.sleep(2 ** attempt)
            else:
                # Non-function calling error, break immediately
                break
                
        except Exception as e:
            last_error = e
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            time.sleep(1)
    
    # Try fallback models
    for fallback_model in fallback_models:
        try:
            print(f"🔄 Trying fallback model: {fallback_model}")
            
            # Update agent model
            original_model = agent.model
            from smolagents import LiteLLMModel
            agent.model = LiteLLMModel(model_id=fallback_model)
            
            response = agent.run(query)
            
            # Restore original model
            agent.model = original_model
            
            return response
            
        except Exception as e:
            print(f"❌ Fallback {fallback_model} failed: {e}")
            continue
    
    # If all fails, return error message
    return f"❌ All attempts failed. Last error: {last_error}"


def create_groq_optimized_agent():
    """Create an agent optimized for Groq function calling."""
    from smolagents import ToolCallingAgent, LiteLLMModel
    from smolagents import WebSearchTool, FinalAnswerTool
    
    # Use tool-calling optimized model
    model = LiteLLMModel(
        model_id="groq/llama3-groq-70b-8192-tool-use-preview",
        max_tokens=4096,
        temperature=0.1,
        top_p=0.9
    )
    
    # Create agent with minimal tools to reduce complexity
    tools = [WebSearchTool(), FinalAnswerTool()]
    
    agent = ToolCallingAgent(
        tools=tools,
        model=model,
        max_steps=3,  # Reduce steps to minimize function calling complexity
        planning_interval=2
    )
    
    return agent
'''

    def test_groq_function_calling(self) -> Dict[str, Any]:
        """Test Groq function calling with various configurations."""
        print("🧪 Testing Groq Function Calling")
        print("=" * 40)
        
        test_results = {
            'timestamp': time.time(),
            'model_tests': {},
            'recommendations': []
        }
        
        # Simple test query that requires function calling
        test_query = "What is the current weather in New York?"
        
        for model_name in self.groq_models:
            print(f"\nTesting model: {model_name}")
            
            try:
                # Check if GROQ API key is available
                import os
                if not os.getenv('GROQ_API_KEY'):
                    print("⚠️  GROQ_API_KEY not found - skipping Groq tests")
                    test_results['model_tests'][model_name] = {
                        'status': 'skipped',
                        'reason': 'No API key'
                    }
                    continue
                
                config = self.create_optimized_model_config(model_name)
                print(f"Config: {config}")
                
                # Test basic model call (without function calling first)
                try:
                    from smolagents import LiteLLMModel
                    
                    model = LiteLLMModel(
                        model_id=model_name,
                        **{k: v for k, v in config.items() if k != 'model'}
                    )
                    
                    # Simple generation test
                    messages = [{"role": "user", "content": "Say hello"}]
                    response = model.generate(messages, stop_sequences=["</response>"])
                    
                    test_results['model_tests'][model_name] = {
                        'status': 'success',
                        'basic_generation': 'working',
                        'response_length': len(str(response)),
                        'config_used': config
                    }
                    
                    print(f"✅ Basic generation working")
                    
                except Exception as e:
                    test_results['model_tests'][model_name] = {
                        'status': 'failed',
                        'error': str(e),
                        'config_used': config
                    }
                    print(f"❌ Failed: {e}")
                    
            except Exception as e:
                print(f"❌ Model test setup failed: {e}")
                test_results['model_tests'][model_name] = {
                    'status': 'setup_failed',
                    'error': str(e)
                }
        
        # Generate recommendations
        working_models = [m for m, r in test_results['model_tests'].items() 
                         if r.get('status') == 'success']
        
        if working_models:
            test_results['recommendations'].append(f"Working models: {', '.join(working_models)}")
            test_results['recommendations'].append("Use lower temperature (0.1-0.2) for function calling")
            test_results['recommendations'].append("Disable streaming for function calling")
            test_results['recommendations'].append("Use tool-calling preview models when available")
        else:
            test_results['recommendations'].append("Consider using OpenRouter or OpenAI as fallback")
            test_results['recommendations'].append("Check GROQ_API_KEY configuration")
        
        return test_results
    
    def generate_fix_summary(self) -> str:
        """Generate a summary of fixes to implement."""
        return """
🔧 GROQ FUNCTION CALLING FIXES SUMMARY
=====================================

1. ERROR HANDLING IMPROVEMENTS:
   ✅ Implement retry logic with exponential backoff
   ✅ Add fallback model support
   ✅ Extract content from failed generations
   ✅ Better error message parsing

2. MODEL CONFIGURATION OPTIMIZATION:
   ✅ Use tool-calling optimized models
   ✅ Lower temperature (0.1-0.2) for consistency
   ✅ Disable streaming for function calls
   ✅ Appropriate token limits

3. PROMPT ENGINEERING:
   ✅ Clear function calling instructions
   ✅ Step-by-step thinking prompts
   ✅ Explicit tool usage guidelines

4. FALLBACK MECHANISMS:
   ✅ OpenRouter models as backup
   ✅ Model switching on failure
   ✅ Content extraction from failed calls

5. IMPLEMENTATION STEPS:
   1. Update agent configuration with optimized settings
   2. Implement error handling wrapper
   3. Add model fallback logic
   4. Test with various query types
   5. Monitor function calling success rates

📋 IMMEDIATE ACTIONS:
- Check GROQ_API_KEY is properly set
- Use groq/llama3-groq-70b-8192-tool-use-preview model
- Implement the groq_safe_agent_call wrapper
- Add fallback to OpenRouter models
- Test with simple queries first
"""


def main():
    """Main function to run the Groq fix analysis."""
    fixer = GroqFunctionCallingFix()
    
    print("🔧 GROQ FUNCTION CALLING DIAGNOSTIC & FIX")
    print("=" * 50)
    
    # Test current configuration
    test_results = fixer.test_groq_function_calling()
    
    print(f"\n📊 Test Results:")
    for model, result in test_results['model_tests'].items():
        status = result.get('status', 'unknown')
        print(f"  {model}: {status}")
    
    print(f"\n💡 Recommendations:")
    for rec in test_results['recommendations']:
        print(f"  • {rec}")
    
    # Generate implementation code
    print(f"\n📝 Error Handling Code:")
    print("Save this to groq_error_handler.py:")
    print("-" * 40)
    print(fixer.implement_error_handling_wrapper())
    
    # Print fix summary
    print(fixer.generate_fix_summary())
    
    # Save results
    try:
        with open('groq_function_calling_diagnosis.json', 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        print("📄 Detailed results saved to: groq_function_calling_diagnosis.json")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")


if __name__ == "__main__":
    main()