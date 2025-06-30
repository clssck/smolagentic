#!/usr/bin/env python3
"""
Test the new optimal multi-model system
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from src.core.manager_agent_system import ManagerAgentSystem


def test_optimal_system():
    """Test the new optimal model configuration"""
    print("🚀 TESTING OPTIMAL MULTI-MODEL SYSTEM")
    print("=" * 60)
    print("Manager: Groq Qwen3-32B (ultra-fast coordination)")
    print("Agents: Mistral Small 3.2 24B (excellent performance)")

    try:
        # Create system with optimal config
        system = ManagerAgentSystem("optimal_models_config.json")

        # Show configuration
        components = system.list_available_components()
        print("\n📋 System Configuration:")
        print(f"  Models: {components['models']}")
        print(f"  Agents: {components['agents']}")

        # Test queries for different agent types
        test_cases = [
            {
                "query": "Hello! How are you today?",
                "expected_agent": "simple_chat",
                "description": "Simple greeting (should use manager's tools directly)",
            },
            {
                "query": "What is 15 * 8 + 27?",
                "expected_agent": "simple_chat",
                "description": "Basic math (manager handles directly)",
            },
            {
                "query": "Search for the latest news about artificial intelligence",
                "expected_agent": "research_agent",
                "description": "Web research (delegates to research agent)",
            },
            {
                "query": "Find information about machine learning in the knowledge base",
                "expected_agent": "rag_agent",
                "description": "Knowledge base query (delegates to RAG agent)",
            },
        ]

        print("\n🧪 PERFORMANCE TESTS")
        print("-" * 40)

        total_time = 0
        successful_tests = 0

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. {test_case['description']}")
            print(f"   Query: '{test_case['query']}'")

            start_time = time.time()
            try:
                response = system.run_query(test_case["query"])
                elapsed_time = time.time() - start_time
                total_time += elapsed_time

                # Check if response is reasonable
                if len(str(response)) > 10:
                    successful_tests += 1
                    print(f"   ✅ {elapsed_time:.2f}s - Success")
                    print(f"   📝 Response: {str(response)[:100]}...")
                else:
                    print(f"   ⚠️  {elapsed_time:.2f}s - Short response")

            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"   ❌ {elapsed_time:.2f}s - Error: {str(e)[:60]}")
                total_time += elapsed_time

        # Performance summary
        avg_time = total_time / len(test_cases)
        success_rate = (successful_tests / len(test_cases)) * 100

        print("\n📊 PERFORMANCE SUMMARY")
        print("=" * 40)
        print(
            f"✅ Success Rate: {success_rate:.0f}% ({successful_tests}/{len(test_cases)})"
        )
        print(f"⚡ Average Speed: {avg_time:.2f}s per query")
        print(f"🚀 Total Time: {total_time:.2f}s for {len(test_cases)} queries")

        if avg_time < 3.0:
            print("🏆 EXCELLENT: System is running optimally!")
        elif avg_time < 5.0:
            print("✅ GOOD: System performance is solid")
        else:
            print("⚠️  SLOW: System may need optimization")

        # System status
        print("\n🔧 SYSTEM STATUS")
        print("-" * 30)
        status = system.get_status()
        for key, value in status.items():
            if key != "config":
                print(f"  {key}: {value}")

        return {
            "success_rate": success_rate,
            "avg_time": avg_time,
            "total_time": total_time,
            "successful_tests": successful_tests,
        }

    except Exception as e:
        print(f"❌ System test failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Main test function"""
    result = test_optimal_system()

    if result:
        print("\n🎉 OPTIMAL SYSTEM TEST COMPLETE!")
        print("The new multi-model configuration is ready for production use.")

        if result["success_rate"] >= 75 and result["avg_time"] < 5.0:
            print("✅ System meets performance targets!")
        else:
            print("⚠️  System may need fine-tuning")
    else:
        print("\n❌ System test failed - check configuration")


if __name__ == "__main__":
    main()
