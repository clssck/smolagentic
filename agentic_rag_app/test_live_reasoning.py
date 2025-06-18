#!/usr/bin/env python3
"""Live testing script for reasoning agent with actual model calls.
Tests the Qwen 3 thinking mode and reasoning capabilities.
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.rag_agent import AgenticRAG
from agents.reasoning_agent import ReasoningAgent
from tools.reasoning_tool import ComparisonTool, ReasoningTool


def test_basic_reasoning():
    """Test basic reasoning functionality with thinking mode"""
    print("🧠 Testing Basic Reasoning Agent...")

    try:
        agent = ReasoningAgent("qwen3-32b-reasoning")

        # Test mathematical reasoning
        print("\n📊 Testing mathematical reasoning:")
        result = agent.think("What is 15% of 240? Show your work step by step.")

        print(f"Query: {result['query']}")
        print(f"Thinking enabled: {result['thinking_enabled']}")
        print(f"Model used: {result['model_used']}")

        if result["thinking_steps"]:
            print(f"\n🤔 Thinking process:\n{result['thinking_steps']}")

        print(f"\n✅ Final answer:\n{result['final_answer']}")

        return True

    except Exception as e:
        print(f"❌ Error in basic reasoning test: {e}")
        return False


def test_problem_solving():
    """Test domain-specific problem solving"""
    print("\n🔢 Testing Problem Solving...")

    try:
        agent = ReasoningAgent("qwen3-32b-reasoning")

        # Test logical reasoning
        result = agent.solve_problem(
            "If all roses are flowers, and some flowers are red, can we conclude that some roses are red?",
            "logic",
        )

        print(f"Domain: {result['domain']}")
        if result["thinking_steps"]:
            print(f"\n🤔 Logical reasoning:\n{result['thinking_steps']}")

        print(f"\n✅ Conclusion:\n{result['final_answer']}")

        return True

    except Exception as e:
        print(f"❌ Error in problem solving test: {e}")
        return False


def test_approach_comparison():
    """Test comparison of different approaches"""
    print("\n⚖️ Testing Approach Comparison...")

    try:
        agent = ReasoningAgent("qwen3-32b-reasoning")

        problem = "Sort a list of 1000 numbers"
        approaches = ["Bubble Sort", "Quick Sort", "Merge Sort"]

        result = agent.compare_approaches(problem, approaches)

        print(f"Problem: {problem}")
        print(f"Approaches compared: {result['approaches_compared']}")

        if result["thinking_steps"]:
            print(f"\n🤔 Analysis process:\n{result['thinking_steps']}")

        print(f"\n✅ Recommendation:\n{result['final_answer']}")

        return True

    except Exception as e:
        print(f"❌ Error in comparison test: {e}")
        return False


def test_reasoning_tools():
    """Test reasoning tools directly"""
    print("\n🛠️ Testing Reasoning Tools...")

    try:
        # Test ReasoningTool
        reasoning_tool = ReasoningTool("qwen3-32b-reasoning")

        print("\n🔧 Testing ReasoningTool:")
        result = reasoning_tool.call(
            "Calculate the compound interest on $1000 at 5% annual rate for 3 years",
            "math",
        )

        print(f"Tool result:\n{result}")

        # Test ComparisonTool
        comparison_tool = ComparisonTool("qwen3-32b-reasoning")

        print("\n🔧 Testing ComparisonTool:")
        result = comparison_tool.call(
            "Choose the best web framework for a small startup",
            "React, Vue.js, Angular",
        )

        print(f"Comparison result:\n{result}")

        return True

    except Exception as e:
        print(f"❌ Error in tools test: {e}")
        return False


def test_rag_with_reasoning():
    """Test RAG agent with reasoning capabilities"""
    print("\n🤖 Testing RAG Agent with Reasoning...")

    try:
        # Create RAG agent with reasoning enabled
        rag = AgenticRAG(
            chat_model_name="qwen3-32b",
            enable_reasoning=True,
            # qdrant_url will use environment variable
        )

        print("\n📊 System Status:")
        status = rag.get_system_status()
        print(f"Chat model: {status.get('current_chat_model', 'unknown')}")
        print("Reasoning enabled: Tools include reasoning capabilities")

        # Test reasoning integration
        print("\n💬 Testing reasoning integration:")
        response = rag.chat(
            "I need to solve this step by step: If a train travels 120 km in 1.5 hours, "
            "what's its speed in km/h and how long would it take to travel 300 km?",
        )

        print(f"RAG + Reasoning response:\n{response}")

        return True

    except Exception as e:
        print(f"❌ Error in RAG reasoning test: {e}")
        return False


def test_model_info():
    """Test model information and configuration"""
    print("\n📋 Testing Model Configuration...")

    try:
        agent = ReasoningAgent("qwen3-32b-reasoning")
        info = agent.get_model_info()

        print("Model Configuration:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ Error in model info test: {e}")
        return False


def main():
    """Run all live tests"""
    print("🚀 Starting Live Reasoning Tests with Qwen 3 Models")
    print("=" * 60)

    # Check for API keys
    required_keys = ["OPENROUTER_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        print(f"❌ Missing required environment variables: {missing_keys}")
        print("Please set your API keys before running tests.")
        return False

    tests = [
        ("Model Configuration", test_model_info),
        ("Basic Reasoning", test_basic_reasoning),
        ("Problem Solving", test_problem_solving),
        ("Approach Comparison", test_approach_comparison),
        ("Reasoning Tools", test_reasoning_tools),
        ("RAG with Reasoning", test_rag_with_reasoning),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 Running: {test_name}")
        print("=" * 60)

        try:
            success = test_func()
            results.append((test_name, success))

            if success:
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")

        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Reasoning agent is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
