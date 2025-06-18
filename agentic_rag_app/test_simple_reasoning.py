#!/usr/bin/env python3
"""Simple live test for reasoning agent without Qdrant dependency.
Tests core Qwen 3 thinking mode functionality.
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.reasoning_agent import ReasoningAgent


def test_basic_thinking():
    """Test basic thinking functionality"""
    print("🧠 Testing Basic Thinking Mode...")

    try:
        agent = ReasoningAgent("qwen3-32b")

        # Simple math problem
        result = agent.think("What is 25% of 80? Show your calculation step by step.")

        print(f"Query: {result['query']}")
        print(f"Model: {result['model_used']}")
        print(f"Thinking enabled: {result['thinking_enabled']}")

        if result["thinking_steps"]:
            print(f"\n🤔 Thinking Process:\n{result['thinking_steps']}")
        else:
            print("\n⚠️ No thinking steps detected")

        print(f"\n✅ Final Answer:\n{result['final_answer']}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reasoning_model():
    """Test reasoning-specific model"""
    print("\n🔧 Testing Reasoning Model Configuration...")

    try:
        agent = ReasoningAgent("qwen3-32b-reasoning")

        info = agent.get_model_info()
        print("Model Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Test with thinking enabled
        result = agent.think("Solve: If x + 5 = 12, what is x?", enable_thinking=True)

        print("\n📝 Problem solving result:")
        print(f"Thinking enabled: {result['thinking_enabled']}")

        if result["thinking_steps"]:
            print(f"\n🤔 Reasoning:\n{result['thinking_steps']}")

        print(f"\n✅ Answer:\n{result['final_answer']}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison():
    """Test approach comparison"""
    print("\n⚖️ Testing Approach Comparison...")

    try:
        agent = ReasoningAgent("qwen3-32b")

        result = agent.compare_approaches(
            "Choose a programming language for web backend",
            ["Python", "JavaScript (Node.js)", "Go"],
        )

        print("Problem: Choose a programming language for web backend")
        print(f"Approaches: {result['approaches_compared']}")

        if result["thinking_steps"]:
            print(f"\n🤔 Analysis:\n{result['thinking_steps']}")

        print(f"\n✅ Recommendation:\n{result['final_answer']}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run simple reasoning tests"""
    print("🚀 Simple Reasoning Tests with Qwen 3")
    print("=" * 50)

    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not found!")
        print("Please set your OpenRouter API key.")
        return False

    tests = [
        ("Basic Thinking", test_basic_thinking),
        ("Reasoning Model", test_reasoning_model),
        ("Approach Comparison", test_comparison),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 {test_name}")
        print("=" * 50)

        success = test_func()
        results.append((test_name, success))

        print(f"\n{'✅ PASSED' if success else '❌ FAILED'}")

    # Summary
    print(f"\n{'='*50}")
    print("📊 SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")

    print(f"\nResult: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
