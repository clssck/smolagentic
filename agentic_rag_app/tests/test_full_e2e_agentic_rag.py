#!/usr/bin/env python3
"""
Comprehensive End-to-End Test for Agentic RAG System with Live APIs

This test suite validates the complete agentic RAG system including:
- Environment setup and API key validation
- Individual component testing (models, embedders, retrievers, routers)
- Agent functionality testing (RAG, Simple QA, Research, Code agents)
- Intelligent routing system
- Full workflow integration
- Performance metrics and reporting

Usage:
    python test_full_e2e_agentic_rag.py
    python test_full_e2e_agentic_rag.py --quick    # Run quick tests only
    python test_full_e2e_agentic_rag.py --agent rag_agent  # Test specific agent
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any

# Add project root to path
sys.path.append(str(os.path.dirname(os.path.abspath(__file__))))

from src.core.modular_system import ModularRAGSystem
from src.utils.config import Config


class E2ETestSuite:
    """Comprehensive end-to-end test suite for the agentic RAG system."""

    def __init__(self, quick_mode: bool = False, specific_agent: str | None = None):
        """Initialize the test suite.

        Args:
            quick_mode: If True, run only essential tests
            specific_agent: If provided, test only this agent
        """
        self.quick_mode = quick_mode
        self.specific_agent = specific_agent
        self.results = {
            "test_start_time": datetime.now().isoformat(),
            "environment": {},
            "components": {},
            "agents": {},
            "routing": {},
            "workflows": {},
            "performance": {},
            "errors": [],
            "summary": {},
        }
        self.system = None

    def print_section(self, title: str, level: int = 1):
        """Print a formatted section header."""
        if level == 1:
            print(f"\n{'=' * 80}")
            print(f" {title}")
            print(f"{'=' * 80}\n")
        else:
            print(f"\n{'-' * 60}")
            print(f" {title}")
            print(f"{'-' * 60}\n")

    def log_result(
        self,
        category: str,
        test_name: str,
        status: str,
        details: dict[str, Any] = None,
        error: str = None,
    ):
        """Log test result."""
        result = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
        }
        if error:
            result["error"] = error
            self.results["errors"].append(
                {
                    "category": category,
                    "test": test_name,
                    "error": error,
                    "timestamp": result["timestamp"],
                }
            )

        if category not in self.results:
            self.results[category] = {}
        self.results[category][test_name] = result

    def test_environment_setup(self) -> bool:
        """Test environment setup and API key validation."""
        self.print_section("1. Environment Setup & API Key Validation")

        try:
            # Validate configuration
            Config.validate()
            print("✅ Configuration validation passed")

            # Check individual API keys
            api_keys = {
                "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
                "DEEPINFRA_API_KEY": os.getenv("DEEPINFRA_API_KEY"),
                "QDRANT_URL": os.getenv("QDRANT_URL"),
                "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),  # Optional
            }

            available_keys = []
            missing_keys = []

            for key, value in api_keys.items():
                if value:
                    available_keys.append(key)
                    print(f"✅ {key}: Available")
                else:
                    missing_keys.append(key)
                    if key in [
                        "OPENROUTER_API_KEY",
                        "DEEPINFRA_API_KEY",
                        "QDRANT_URL",
                        "QDRANT_API_KEY",
                    ]:
                        print(f"❌ {key}: Missing (Required)")
                    else:
                        print(f"⚠️  {key}: Missing (Optional)")

            self.results["environment"] = {
                "available_keys": available_keys,
                "missing_keys": missing_keys,
                "config_valid": True,
            }

            if (
                len(
                    [
                        k
                        for k in missing_keys
                        if k
                        in [
                            "OPENROUTER_API_KEY",
                            "DEEPINFRA_API_KEY",
                            "QDRANT_URL",
                            "QDRANT_API_KEY",
                        ]
                    ]
                )
                > 0
            ):
                print("\n❌ Some required API keys are missing!")
                return False

            print("\n✅ Environment setup complete!")
            return True

        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            self.log_result("environment", "setup", "failed", error=str(e))
            return False

    async def test_individual_components(self):
        """Test individual components (models, embedders, retrievers)."""
        self.print_section("2. Individual Component Testing")

        try:
            self.system = ModularRAGSystem()

            # Test Models
            await self._test_models()

            # Test Embedders
            await self._test_embedders()

            # Test Retrievers
            await self._test_retrievers()

            # Test Routers
            await self._test_routers()

        except Exception as e:
            print(f"❌ Component testing failed: {e}")
            self.log_result("components", "overall", "failed", error=str(e))

    async def _test_models(self):
        """Test model functionality."""
        self.print_section("Testing Models", level=2)

        # Get available models based on API keys
        test_models = []
        if os.getenv("OPENROUTER_API_KEY"):
            test_models.extend(["openrouter_qwen3_32b", "openrouter_qwen3_14b"])
        if os.getenv("DEEPINFRA_API_KEY"):
            test_models.extend(["deepinfra_qwen_32b"])

        model_results = {}

        for model_name in test_models:
            print(f"\nTesting model: {model_name}")
            try:
                start_time = time.time()

                # Get model
                model = self.system.get_model(model_name)
                print("  ✅ Model created successfully")

                # Test simple generation
                response = self.system.generate_response(
                    [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. Be very brief.",
                        },
                        {
                            "role": "user",
                            "content": "What is 2+2? Answer with just the number.",
                        },
                    ],
                    model_name=model_name,
                )

                response_time = time.time() - start_time
                print(f"  ✅ Generation successful ({response_time:.2f}s)")
                print(f"  📝 Response: {str(response)[:100]}...")

                model_results[model_name] = {
                    "status": "success",
                    "response_time": response_time,
                    "response_preview": str(response)[:100],
                }

            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}...")
                model_results[model_name] = {"status": "failed", "error": str(e)[:100]}

        self.log_result("components", "models", "completed", details=model_results)

    async def _test_embedders(self):
        """Test embedder functionality."""
        self.print_section("Testing Embedders", level=2)

        test_embedders = []
        if os.getenv("DEEPINFRA_API_KEY"):
            test_embedders.append("deepinfra_qwen_4b")

        # Always test local embedders if available
        test_embedders.extend(["local_minilm", "local_bge_large"])

        embedder_results = {}
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming artificial intelligence.",
        ]

        for embedder_name in test_embedders:
            print(f"\nTesting embedder: {embedder_name}")
            try:
                # Skip local models if dependencies not available
                if embedder_name.startswith("local_"):
                    try:
                        import sentence_transformers
                    except ImportError:
                        print("  ⚠️  Skipping - sentence-transformers not installed")
                        continue

                start_time = time.time()

                # Get embedder
                embedder = self.system.get_embedder(embedder_name)
                print("  ✅ Embedder created successfully")

                # Test embedding
                embeddings = self.system.embed_text(
                    test_texts[0], embedder_name=embedder_name
                )

                response_time = time.time() - start_time
                print(f"  ✅ Embedding successful ({response_time:.2f}s)")
                print(f"  📏 Embedding dimension: {len(embeddings)}")

                embedder_results[embedder_name] = {
                    "status": "success",
                    "response_time": response_time,
                    "dimension": len(embeddings),
                }

            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}...")
                embedder_results[embedder_name] = {
                    "status": "failed",
                    "error": str(e)[:100],
                }

        self.log_result(
            "components", "embedders", "completed", details=embedder_results
        )

    async def _test_retrievers(self):
        """Test retriever functionality."""
        self.print_section("Testing Retrievers", level=2)

        retriever_results = {}

        try:
            print("Testing Qdrant retriever...")

            # Test retrieval
            start_time = time.time()
            documents = self.system.retrieve_documents(
                "machine learning artificial intelligence", retriever_name="qdrant_main"
            )
            response_time = time.time() - start_time

            print(f"  ✅ Retrieval successful ({response_time:.2f}s)")
            print(f"  📄 Retrieved {len(documents)} documents")

            retriever_results["qdrant_main"] = {
                "status": "success",
                "response_time": response_time,
                "documents_count": len(documents),
            }

        except Exception as e:
            print(f"  ❌ Error: {str(e)[:100]}...")
            retriever_results["qdrant_main"] = {
                "status": "failed",
                "error": str(e)[:100],
            }

        self.log_result(
            "components", "retrievers", "completed", details=retriever_results
        )

    async def _test_routers(self):
        """Test router functionality."""
        self.print_section("Testing Routers", level=2)

        router_results = {}

        try:
            print("Testing intelligent router...")

            # Get router
            router = self.system.get_router("intelligent_router")
            print("  ✅ Router created successfully")

            router_results["intelligent_router"] = {
                "status": "success",
                "type": type(router).__name__,
            }

        except Exception as e:
            print(f"  ❌ Error: {str(e)[:100]}...")
            router_results["intelligent_router"] = {
                "status": "failed",
                "error": str(e)[:100],
            }

        self.log_result("components", "routers", "completed", details=router_results)

    async def test_agent_functionality(self):
        """Test each agent type with real queries."""
        self.print_section("3. Agent Functionality Testing")

        # Define test queries for each agent type
        test_queries = {
            "rag_agent": [
                "What information do you have about machine learning in your knowledge base?",
                "Search for documents about artificial intelligence applications.",
            ],
            "simple_qa": [
                "What is the capital of France?",
                "Define machine learning in simple terms.",
            ],
            "research_agent": [
                "Research the latest trends in artificial intelligence and provide a comprehensive analysis.",
                "Compare the pros and cons of different machine learning approaches.",
            ],
            "code_agent": [
                "Write a Python function to calculate fibonacci numbers.",
                "Create a simple REST API endpoint using FastAPI.",
            ],
        }

        # Test specific agent if requested
        if self.specific_agent:
            if self.specific_agent in test_queries:
                test_queries = {self.specific_agent: test_queries[self.specific_agent]}
            else:
                print(f"❌ Unknown agent: {self.specific_agent}")
                return

        agent_results = {}

        for agent_name, queries in test_queries.items():
            print(f"\nTesting agent: {agent_name}")

            try:
                # Get agent
                agent = self.system.get_agent(agent_name)
                print("  ✅ Agent created successfully")

                query_results = []

                for i, query in enumerate(queries):
                    if self.quick_mode and i > 0:  # Only test first query in quick mode
                        break

                    print(f"  🔍 Testing query {i + 1}: {query[:50]}...")

                    try:
                        start_time = time.time()
                        response = self.system.run_query(query, agent_name=agent_name)
                        response_time = time.time() - start_time

                        print(f"    ✅ Response received ({response_time:.2f}s)")
                        print(f"    📝 Preview: {str(response)[:100]}...")

                        query_results.append(
                            {
                                "query": query,
                                "status": "success",
                                "response_time": response_time,
                                "response_preview": str(response)[:200],
                            }
                        )

                    except Exception as e:
                        print(f"    ❌ Query failed: {str(e)[:100]}...")
                        query_results.append(
                            {
                                "query": query,
                                "status": "failed",
                                "error": str(e)[:100],
                            }
                        )

                agent_results[agent_name] = {
                    "status": "completed",
                    "queries_tested": len(query_results),
                    "successful_queries": len(
                        [q for q in query_results if q["status"] == "success"]
                    ),
                    "query_results": query_results,
                }

            except Exception as e:
                print(f"  ❌ Agent creation failed: {str(e)[:100]}...")
                agent_results[agent_name] = {"status": "failed", "error": str(e)[:100]}

        self.log_result("agents", "functionality", "completed", details=agent_results)

    async def test_intelligent_routing(self):
        """Test the intelligent routing system."""
        self.print_section("4. Intelligent Routing System Testing")

        # Test queries that should route to different agents
        routing_test_queries = [
            ("What is the capital of Japan?", "simple_qa"),
            ("Search for documents about neural networks", "rag_agent"),
            ("Research the latest developments in quantum computing", "research_agent"),
            ("Write a Python function to sort a list", "code_agent"),
        ]

        routing_results = []

        try:
            from src.core.main_router import SmartAgentRouter

            router = SmartAgentRouter()

            for query, expected_agent in routing_test_queries:
                print(f"\nTesting routing for: {query[:50]}...")

                try:
                    start_time = time.time()

                    # Test routing decision
                    # Note: This depends on the actual router implementation
                    response = router.run_query(query)

                    routing_time = time.time() - start_time
                    print(f"  ✅ Routing completed ({routing_time:.2f}s)")
                    print(f"  📝 Response preview: {str(response)[:100]}...")

                    routing_results.append(
                        {
                            "query": query,
                            "expected_agent": expected_agent,
                            "status": "success",
                            "routing_time": routing_time,
                            "response_preview": str(response)[:200],
                        }
                    )

                except Exception as e:
                    print(f"  ❌ Routing failed: {str(e)[:100]}...")
                    routing_results.append(
                        {
                            "query": query,
                            "expected_agent": expected_agent,
                            "status": "failed",
                            "error": str(e)[:100],
                        }
                    )

        except Exception as e:
            print(f"❌ Router initialization failed: {str(e)[:100]}...")
            routing_results.append(
                {"error": f"Router initialization failed: {str(e)[:100]}"}
            )

        self.log_result(
            "routing", "intelligent_routing", "completed", details=routing_results
        )

    async def test_full_workflows(self):
        """Test complete end-to-end workflows."""
        self.print_section("5. Full Workflow Integration Testing")

        workflows = [
            {
                "name": "Knowledge Base Query Workflow",
                "description": "Query knowledge base and generate comprehensive response",
                "query": "What are the main components of a RAG system and how do they work together?",
                "expected_agent": "rag_agent",
            },
            {
                "name": "Research and Analysis Workflow",
                "description": "Comprehensive research with multiple sources",
                "query": "Analyze the current state of large language models and their applications",
                "expected_agent": "research_agent",
            },
            {
                "name": "Code Generation Workflow",
                "description": "Generate and explain code solutions",
                "query": "Create a Python class for a simple chatbot with memory",
                "expected_agent": "code_agent",
            },
        ]

        if self.quick_mode:
            workflows = workflows[:1]  # Only test first workflow in quick mode

        workflow_results = []

        for workflow in workflows:
            print(f"\nTesting workflow: {workflow['name']}")
            print(f"Description: {workflow['description']}")

            try:
                start_time = time.time()

                # Run the complete workflow
                response = self.system.run_query(workflow["query"])

                workflow_time = time.time() - start_time
                print(f"  ✅ Workflow completed ({workflow_time:.2f}s)")
                print(f"  📝 Response length: {len(str(response))} characters")
                print(f"  📝 Preview: {str(response)[:150]}...")

                workflow_results.append(
                    {
                        "name": workflow["name"],
                        "query": workflow["query"],
                        "status": "success",
                        "workflow_time": workflow_time,
                        "response_length": len(str(response)),
                        "response_preview": str(response)[:300],
                    }
                )

            except Exception as e:
                print(f"  ❌ Workflow failed: {str(e)[:100]}...")
                workflow_results.append(
                    {
                        "name": workflow["name"],
                        "query": workflow["query"],
                        "status": "failed",
                        "error": str(e)[:100],
                    }
                )

        self.log_result(
            "workflows", "integration", "completed", details=workflow_results
        )

    async def test_web_ui_functionality(self):
        """Test the Gradio web UI functionality."""
        self.print_section("6. Web UI Functionality Testing")

        if self.quick_mode:
            print("⚠️  Skipping web UI tests in quick mode")
            return

        try:
            from src.ui.web_ui import create_rag_agent

            print("Testing Web UI agent creation...")
            agent = create_rag_agent()
            print("  ✅ RAG agent created successfully")

            # Test agent functionality
            print("Testing agent functionality...")
            test_message = "What is machine learning?"

            try:
                response = agent.run(test_message)
                print("  ✅ Agent run successful")
                print(f"  📝 Response preview: {str(response)[:100]}...")

                ui_results = {
                    "initialization": "success",
                    "agent_run": "success",
                    "test_message": test_message,
                    "response_preview": str(response)[:200],
                }

            except Exception as e:
                print(f"  ❌ Agent run failed: {str(e)[:100]}...")
                ui_results = {
                    "initialization": "success",
                    "agent_run": "failed",
                    "error": str(e)[:100],
                }

        except Exception as e:
            print(f"❌ Web UI agent creation failed: {str(e)[:100]}...")
            ui_results = {"initialization": "failed", "error": str(e)[:100]}

        self.log_result("web_ui", "functionality", "completed", details=ui_results)

    def calculate_performance_metrics(self):
        """Calculate overall performance metrics."""
        self.print_section("7. Performance Metrics Calculation")

        metrics = {
            "total_tests": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "average_response_times": {},
            "component_success_rates": {},
            "overall_success_rate": 0.0,
        }

        # Analyze component results
        for category, tests in self.results.items():
            if category in [
                "test_start_time",
                "environment",
                "errors",
                "summary",
                "performance",
            ]:
                continue

            for test_name, test_data in tests.items():
                if isinstance(test_data, dict) and "status" in test_data:
                    metrics["total_tests"] += 1

                    if test_data["status"] in ["success", "completed"]:
                        metrics["successful_tests"] += 1
                    else:
                        metrics["failed_tests"] += 1

        # Calculate success rate
        if metrics["total_tests"] > 0:
            metrics["overall_success_rate"] = (
                metrics["successful_tests"] / metrics["total_tests"]
            )

        # Extract response times
        response_times = []
        for category, tests in self.results.items():
            if isinstance(tests, dict):
                for test_name, test_data in tests.items():
                    if isinstance(test_data, dict) and "details" in test_data:
                        details = test_data["details"]
                        if isinstance(details, dict):
                            for item_name, item_data in details.items():
                                if (
                                    isinstance(item_data, dict)
                                    and "response_time" in item_data
                                ):
                                    response_times.append(item_data["response_time"])

        if response_times:
            metrics["average_response_time"] = sum(response_times) / len(response_times)
            metrics["min_response_time"] = min(response_times)
            metrics["max_response_time"] = max(response_times)

        self.results["performance"] = metrics

        print(f"📊 Total tests: {metrics['total_tests']}")
        print(f"✅ Successful: {metrics['successful_tests']}")
        print(f"❌ Failed: {metrics['failed_tests']}")
        print(f"📈 Success rate: {metrics['overall_success_rate']:.1%}")

        if response_times:
            print(f"⏱️  Average response time: {metrics['average_response_time']:.2f}s")
            print(f"⚡ Fastest response: {metrics['min_response_time']:.2f}s")
            print(f"🐌 Slowest response: {metrics['max_response_time']:.2f}s")

    def generate_test_report(self):
        """Generate comprehensive test report."""
        self.print_section("8. Test Report Generation")

        # Add test completion time
        self.results["test_end_time"] = datetime.now().isoformat()

        # Create summary
        summary = {
            "test_duration": "N/A",
            "total_components_tested": len(
                [
                    k
                    for k in self.results
                    if k
                    not in [
                        "test_start_time",
                        "test_end_time",
                        "environment",
                        "errors",
                        "summary",
                        "performance",
                    ]
                ]
            ),
            "total_errors": len(self.results.get("errors", [])),
            "quick_mode": self.quick_mode,
            "specific_agent": self.specific_agent,
        }

        # Calculate test duration
        try:
            start_time = datetime.fromisoformat(self.results["test_start_time"])
            end_time = datetime.fromisoformat(self.results["test_end_time"])
            duration = end_time - start_time
            summary["test_duration"] = str(duration)
        except:
            pass

        self.results["summary"] = summary

        # Save detailed report
        report_filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(report_filename, "w") as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"📄 Detailed report saved: {report_filename}")
        except Exception as e:
            print(f"⚠️  Could not save report: {e}")

        # Print summary
        print("\n📋 Test Summary:")
        print(f"   Duration: {summary['test_duration']}")
        print(f"   Components tested: {summary['total_components_tested']}")
        print(f"   Total errors: {summary['total_errors']}")
        print(f"   Quick mode: {summary['quick_mode']}")
        if summary["specific_agent"]:
            print(f"   Specific agent: {summary['specific_agent']}")

        # Print errors if any
        if self.results.get("errors"):
            print("\n❌ Errors encountered:")
            for error in self.results["errors"][:5]:  # Show first 5 errors
                print(
                    f"   {error['category']}.{error['test']}: {error['error'][:100]}..."
                )
            if len(self.results["errors"]) > 5:
                print(f"   ... and {len(self.results['errors']) - 5} more errors")

    async def run_all_tests(self):
        """Run the complete test suite."""
        print("🚀 Starting Comprehensive Agentic RAG System E2E Tests")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if self.quick_mode:
            print("⚡ Running in QUICK mode - limited tests")
        if self.specific_agent:
            print(f"🎯 Testing specific agent: {self.specific_agent}")

        try:
            # 1. Environment setup
            if not self.test_environment_setup():
                print("\n❌ Environment setup failed - cannot continue")
                return False

            # 2. Component testing
            await self.test_individual_components()

            # 3. Agent functionality
            await self.test_agent_functionality()

            # 4. Intelligent routing
            if not self.specific_agent:  # Skip routing tests if testing specific agent
                await self.test_intelligent_routing()

            # 5. Full workflows
            await self.test_full_workflows()

            # 6. Web UI (if not quick mode)
            await self.test_web_ui_functionality()

            # 7. Performance metrics
            self.calculate_performance_metrics()

            # 8. Generate report
            self.generate_test_report()

            print("\n🎉 All tests completed!")
            return True

        except Exception as e:
            print(f"\n💥 Fatal error during testing: {e}")
            traceback.print_exc()
            return False


async def main():
    """Main function to run the E2E test suite."""
    parser = argparse.ArgumentParser(
        description="Comprehensive End-to-End Test Suite for Agentic RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_full_e2e_agentic_rag.py                    # Run full test suite
  python test_full_e2e_agentic_rag.py --quick            # Run quick tests only
  python test_full_e2e_agentic_rag.py --agent rag_agent  # Test specific agent
  python test_full_e2e_agentic_rag.py --help             # Show this help

Test Categories:
  1. Environment Setup & API Key Validation
  2. Individual Component Testing (Models, Embedders, Retrievers, Routers)
  3. Agent Functionality Testing (RAG, Simple QA, Research, Code agents)
  4. Intelligent Routing System Testing
  5. Full Workflow Integration Testing
  6. Web UI Functionality Testing
  7. Performance Metrics Calculation
  8. Test Report Generation

Requirements:
  - OPENROUTER_API_KEY environment variable
  - DEEPINFRA_API_KEY environment variable
  - QDRANT_URL environment variable
  - QDRANT_API_KEY environment variable
        """,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (skip comprehensive workflows and UI tests)",
    )

    parser.add_argument(
        "--agent",
        type=str,
        choices=["rag_agent", "simple_qa", "research_agent", "code_agent"],
        help="Test only a specific agent type",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (currently not implemented)",
    )

    args = parser.parse_args()

    # Create and run test suite
    test_suite = E2ETestSuite(quick_mode=args.quick, specific_agent=args.agent)

    success = await test_suite.run_all_tests()

    if success:
        print("\n🎉 Test suite completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test suite failed!")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
