#!/usr/bin/env python3
"""Debug the tool interface to see what's being passed"""

from tools.enhanced_search_tool import EnhancedSearchTool
from llama_index.core.tools import ToolOutput

def test_tool_interface():
    tool = EnhancedSearchTool()
    
    # Test different input formats that ReAct agent might use
    test_inputs = [
        "ufdf",
        {"input": "ufdf"},
        {"query": "ufdf"},
        {"tool_input": "ufdf"}
    ]
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n=== Test {i}: {type(test_input).__name__} - {test_input} ===")
        try:
            result = tool(test_input)
            print(f"Result type: {type(result)}")
            if isinstance(result, ToolOutput):
                print(f"Content preview: {result.content[:200]}...")
            else:
                print(f"Content preview: {str(result)[:200]}...")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_tool_interface()