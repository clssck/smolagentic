#!/usr/bin/env python3
"""
Model Benchmarker - Test different OpenRouter models for agent capabilities
"""

import json
import time
from dataclasses import dataclass
from typing import Any

try:
    from smolagents import (
        CodeAgent,
        LiteLLMModel,
        ToolCallingAgent,
        WebSearchTool,
        tool,
    )

    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False
    print("⚠️  smolagents not available")


@dataclass
class BenchmarkResult:
    """Results from a model benchmark test"""

    model_name: str
    test_name: str
    success: bool
    response_time: float
    response_quality: int  # 1-5 scale
    tool_usage_correct: bool
    error_message: str = ""
    response_text: str = ""
    token_usage: dict = None


@tool
def simple_math(expression: str) -> str:
    """Evaluate a simple mathematical expression

    Args:
        expression: Mathematical expression like "2+2" or "10*5"
    """
    try:
        # Safe evaluation for basic math
        allowed_chars = set("0123456789+-*/.() ")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"The result of {expression} is {result}"
        else:
            return "Invalid expression - only basic math operations allowed"
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def text_analyzer(text: str) -> str:
    """Analyze text and return basic statistics

    Args:
        text: Text to analyze
    """
    words = text.split()
    sentences = text.count(".") + text.count("!") + text.count("?")

    return f"Text analysis:\n- Words: {len(words)}\n- Characters: {len(text)}\n- Sentences: {sentences}\n- Average word length: {sum(len(w) for w in words) / len(words):.1f}"


@tool
def format_response(content: str, format_type: str = "markdown") -> str:
    """Format content in specified format

    Args:
        content: Content to format
        format_type: Format type (markdown, json, xml)
    """
    if format_type == "markdown":
        return f"# Formatted Response\n\n{content}\n\n*Formatted as markdown*"
    elif format_type == "json":
        return json.dumps({"response": content, "format": "json"}, indent=2)
    elif format_type == "xml":
        return f"<response>\n  <content>{content}</content>\n  <format>xml</format>\n</response>"
    else:
        return f"Unknown format: {format_type}"


class ModelBenchmarker:
    """Benchmark different models for agent capabilities"""

    def __init__(self):
        self.test_tools = [simple_math, text_analyzer, format_response]
        self.results = []

    def create_test_agent(self, model_name: str, agent_type: str = "code") -> Any:
        """Create a test agent with the specified model"""
        if not SMOLAGENTS_AVAILABLE:
            raise ImportError("smolagents not available")

        model = LiteLLMModel(
            model_id=f"openrouter/{model_name}", temperature=0.1, max_tokens=1000
        )

        if agent_type == "code":
            return CodeAgent(
                tools=self.test_tools,
                model=model,
                max_steps=3,
                verbosity_level=0,
                return_full_result=True,
            )
        else:  # tool_calling
            return ToolCallingAgent(
                tools=self.test_tools, model=model, max_steps=3, verbosity_level=0
            )

    def run_benchmark_test(self, model_name: str, test_config: dict) -> BenchmarkResult:
        """Run a single benchmark test on a model"""
        test_name = test_config["name"]
        query = test_config["query"]
        expected_tool = test_config.get("expected_tool")
        agent_type = test_config.get("agent_type", "code")

        print(f"  🧪 {test_name}")

        try:
            # Create agent
            agent = self.create_test_agent(model_name, agent_type)

            # Run test
            start_time = time.time()
            result = agent.run(query)
            response_time = time.time() - start_time

            # Extract response
            if hasattr(result, "output"):
                response_text = str(result.output)
            else:
                response_text = str(result)

            # Check tool usage (basic heuristic)
            tool_usage_correct = True
            if expected_tool:
                tool_usage_correct = expected_tool.lower() in response_text.lower()

            # Rate response quality (1-5 scale)
            quality_score = self._rate_response_quality(
                query, response_text, test_config
            )

            return BenchmarkResult(
                model_name=model_name,
                test_name=test_name,
                success=True,
                response_time=response_time,
                response_quality=quality_score,
                tool_usage_correct=tool_usage_correct,
                response_text=response_text[:200] + "..."
                if len(response_text) > 200
                else response_text,
            )

        except Exception as e:
            return BenchmarkResult(
                model_name=model_name,
                test_name=test_name,
                success=False,
                response_time=0,
                response_quality=1,
                tool_usage_correct=False,
                error_message=str(e)[:100],
            )

    def _rate_response_quality(
        self, query: str, response: str, test_config: dict
    ) -> int:
        """Rate response quality on 1-5 scale"""
        score = 3  # baseline

        # Check if response addresses the query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = (
            len(query_words & response_words) / len(query_words) if query_words else 0
        )

        if overlap > 0.5:
            score += 1
        if overlap < 0.2:
            score -= 1

        # Check response length (not too short or too long)
        if 50 <= len(response) <= 500:
            score += 1
        elif len(response) < 20:
            score -= 2

        # Check for expected keywords
        expected_keywords = test_config.get("expected_keywords", [])
        if expected_keywords:
            keyword_matches = sum(
                1 for kw in expected_keywords if kw.lower() in response.lower()
            )
            if keyword_matches >= len(expected_keywords) * 0.7:
                score += 1

        return max(1, min(5, score))

    def benchmark_model(self, model_name: str, test_suite: list) -> dict:
        """Run full benchmark suite on a model"""
        print(f"\n🚀 Benchmarking {model_name}")
        print("-" * 60)

        model_results = []
        total_time = 0
        successful_tests = 0

        for test_config in test_suite:
            result = self.run_benchmark_test(model_name, test_config)
            model_results.append(result)

            if result.success:
                total_time += result.response_time
                successful_tests += 1
                status = "✅" if result.tool_usage_correct else "⚠️"
                print(
                    f"    {status} {result.response_time:.2f}s - Quality: {result.response_quality}/5"
                )
            else:
                print(f"    ❌ Failed - {result.error_message}")

        # Calculate metrics
        avg_time = total_time / successful_tests if successful_tests > 0 else 0
        success_rate = successful_tests / len(test_suite) * 100
        avg_quality = sum(r.response_quality for r in model_results) / len(
            model_results
        )
        tool_accuracy = (
            sum(1 for r in model_results if r.tool_usage_correct)
            / len(model_results)
            * 100
        )

        summary = {
            "model": model_name,
            "success_rate": success_rate,
            "avg_response_time": avg_time,
            "avg_quality_score": avg_quality,
            "tool_usage_accuracy": tool_accuracy,
            "total_tests": len(test_suite),
            "successful_tests": successful_tests,
            "detailed_results": model_results,
        }

        print(
            f"📊 Summary: {success_rate:.0f}% success, {avg_time:.2f}s avg, {avg_quality:.1f}/5 quality"
        )

        return summary

    def create_test_suite(self) -> list:
        """Create comprehensive test suite"""
        return [
            {
                "name": "Simple Math",
                "query": "Calculate 15 * 7 + 23",
                "expected_tool": "simple_math",
                "expected_keywords": ["128", "result"],
                "agent_type": "code",
            },
            {
                "name": "Text Analysis",
                "query": "Analyze this text: 'Hello world! How are you today? I hope you're doing well.'",
                "expected_tool": "text_analyzer",
                "expected_keywords": ["words", "characters", "sentences"],
                "agent_type": "code",
            },
            {
                "name": "Format as JSON",
                "query": "Format the response 'AI is amazing' as JSON",
                "expected_tool": "format_response",
                "expected_keywords": ["json", "response"],
                "agent_type": "code",
            },
            {
                "name": "Multi-step Task",
                "query": "Calculate 5*5, then analyze the result as text, then format it as markdown",
                "expected_tool": "multiple",
                "expected_keywords": ["25", "analysis", "markdown"],
                "agent_type": "code",
            },
            {
                "name": "Tool Selection",
                "query": "I need to format some data as XML",
                "expected_tool": "format_response",
                "expected_keywords": ["xml"],
                "agent_type": "tool_calling",
            },
            {
                "name": "Complex Math",
                "query": "What's the result of (10 + 5) * 3 - 8?",
                "expected_tool": "simple_math",
                "expected_keywords": ["37", "result"],
                "agent_type": "code",
            },
        ]

    def run_comprehensive_benchmark(self, models: list) -> dict:
        """Run comprehensive benchmark across all models"""
        print("🔬 MODEL BENCHMARKING SUITE")
        print("=" * 80)
        print("Testing capabilities:")
        print("  🧮 Mathematical operations")
        print("  📝 Text analysis")
        print("  🎨 Response formatting")
        print("  🔀 Multi-step tasks")
        print("  ⚡ Speed and accuracy")

        test_suite = self.create_test_suite()
        all_results = {}

        for model in models:
            try:
                result = self.benchmark_model(model, test_suite)
                all_results[model] = result
                self.results.append(result)
            except Exception as e:
                print(f"❌ Failed to benchmark {model}: {e}")
                all_results[model] = {"error": str(e)}

            time.sleep(1)  # Brief pause between models

        # Generate summary report
        self._generate_summary_report(all_results)

        return all_results

    def _generate_summary_report(self, results: dict):
        """Generate summary report comparing all models"""
        print("\n🏆 BENCHMARK RESULTS SUMMARY")
        print("=" * 80)

        valid_results = {k: v for k, v in results.items() if "error" not in v}

        if not valid_results:
            print("❌ No successful benchmarks to compare")
            return

        # Sort by overall score (combination of success rate, speed, quality)
        def calculate_score(result):
            success_weight = 0.4
            speed_weight = 0.3  # Lower is better, so invert
            quality_weight = 0.3

            success_score = result["success_rate"] / 100
            speed_score = max(
                0, 1 - (result["avg_response_time"] / 10)
            )  # Normalize speed
            quality_score = result["avg_quality_score"] / 5

            return (
                success_weight * success_score
                + speed_weight * speed_score
                + quality_weight * quality_score
            )

        sorted_results = sorted(
            valid_results.items(), key=lambda x: calculate_score(x[1]), reverse=True
        )

        print(
            f"{'Rank':<4} {'Model':<35} {'Success':<8} {'Speed':<8} {'Quality':<8} {'Tools':<8}"
        )
        print("-" * 80)

        for i, (model, result) in enumerate(sorted_results):
            rank = (
                "🥇"
                if i == 0
                else "🥈"
                if i == 1
                else "🥉"
                if i == 2
                else f"{i + 1:2d}"
            )
            model_short = model.replace("mistralai/", "").replace("qwen/", "")[:30]

            print(
                f"{rank:<4} {model_short:<35} "
                f"{result['success_rate']:.0f}%     "
                f"{result['avg_response_time']:.2f}s    "
                f"{result['avg_quality_score']:.1f}/5    "
                f"{result['tool_usage_accuracy']:.0f}%"
            )

        # Speed champions
        print("\n⚡ SPEED CHAMPIONS:")
        speed_sorted = sorted(
            valid_results.items(), key=lambda x: x[1]["avg_response_time"]
        )
        for i, (model, result) in enumerate(speed_sorted[:3]):
            model_short = model.replace("mistralai/", "").replace("qwen/", "")[:25]
            print(f"  {i + 1}. {model_short:<25} - {result['avg_response_time']:.2f}s")

        # Quality champions
        print("\n🎯 QUALITY CHAMPIONS:")
        quality_sorted = sorted(
            valid_results.items(), key=lambda x: x[1]["avg_quality_score"], reverse=True
        )
        for i, (model, result) in enumerate(quality_sorted[:3]):
            model_short = model.replace("mistralai/", "").replace("qwen/", "")[:25]
            print(f"  {i + 1}. {model_short:<25} - {result['avg_quality_score']:.1f}/5")

        # Tool usage champions
        print("\n🔧 TOOL USAGE CHAMPIONS:")
        tool_sorted = sorted(
            valid_results.items(),
            key=lambda x: x[1]["tool_usage_accuracy"],
            reverse=True,
        )
        for i, (model, result) in enumerate(tool_sorted[:3]):
            model_short = model.replace("mistralai/", "").replace("qwen/", "")[:25]
            print(
                f"  {i + 1}. {model_short:<25} - {result['tool_usage_accuracy']:.0f}%"
            )

    def save_results(self, filename: str = "model_benchmark_results.json"):
        """Save benchmark results to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = f"benchmark_results_{timestamp}.json"

        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n💾 Results saved to {filepath}")


def main():
    """Run the model benchmarking suite"""

    # OpenRouter models to test
    models_to_test = [
        "mistralai/mistral-small-3.2-24b-instruct:free",  # Free Mistral
        "mistralai/mistral-small-3.2-24b-instruct",  # Paid Mistral
        "qwen/qwen3-235b-a22b",  # Large Qwen
        "qwen/qwen2.5-vl-72b-instruct",  # VLM Qwen
        "qwen/qwen3-32b",  # Mid Qwen
        "qwen/qwen3-32b:free",  # Free mid Qwen
        "qwen/qwen3-30b-a3b",  # Current default
        "qwen/qwen3-235b-a22b:free",  # Free large Qwen
        "qwen/qwen3-14b:free",  # Free small Qwen
    ]

    benchmarker = ModelBenchmarker()
    results = benchmarker.run_comprehensive_benchmark(models_to_test)

    # Save results
    benchmarker.save_results()

    print(f"\n🎉 Benchmarking complete! Tested {len(models_to_test)} models.")
    print("💡 Use these results to assign optimal models to different agent types.")


if __name__ == "__main__":
    main()
