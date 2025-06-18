"""Tests for reasoning tools functionality.

This module contains tests for ReasoningTool and ComparisonTool classes
that provide advanced reasoning capabilities for the RAG system.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index.core.tools import ToolMetadata
from tools.reasoning_tool import ComparisonTool, ReasoningTool, get_reasoning_tools


class TestReasoningTool(unittest.TestCase):
    """Test cases for the ReasoningTool class functionality."""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_reasoning_agent = Mock()

        # Patch the get_reasoning_agent function
        self.agent_patcher = patch("tools.reasoning_tool.get_reasoning_agent")
        self.mock_get_agent = self.agent_patcher.start()
        self.mock_get_agent.return_value = self.mock_reasoning_agent

    def tearDown(self):
        """Clean up patches"""
        self.agent_patcher.stop()

    def test_init_default_model(self):
        """Test ReasoningTool initialization with default model"""
        tool = ReasoningTool()

        self.mock_get_agent.assert_called_with("qwen3-32b-reasoning")
        self.assertEqual(tool.reasoning_agent, self.mock_reasoning_agent)

        # Check metadata
        self.assertIsInstance(tool.metadata, ToolMetadata)
        self.assertEqual(tool.metadata.name, "deep_reasoning")
        self.assertIn("complex reasoning tasks", tool.metadata.description)
        self.assertIn("step-by-step thinking", tool.metadata.description)

    def test_init_custom_model(self):
        """Test ReasoningTool initialization with custom model"""
        tool = ReasoningTool("custom-model")

        self.mock_get_agent.assert_called_with("custom-model")

    def test_call_success_with_thinking(self):
        """Test successful tool call with thinking steps"""
        tool = ReasoningTool()

        # Mock reasoning agent response
        mock_result = {
            "thinking_steps": "Step 1: Analyze the problem\nStep 2: Consider solutions",
            "final_answer": "The solution is X",
            "thinking_enabled": True,
            "domain": "math",
            "model_used": "qwen3-32b-reasoning",
        }
        self.mock_reasoning_agent.solve_problem.return_value = mock_result

        result = tool.call("What is 2 + 2?", "math")

        # Verify agent was called correctly
        self.mock_reasoning_agent.solve_problem.assert_called_once_with("What is 2 + 2?", "math")

        # Verify result is a string with formatted response
        self.assertIsInstance(result, str)
        self.assertIn("**Reasoning Process:**", result)
        self.assertIn("Step 1: Analyze", result)
        self.assertIn("**Final Answer:**", result)
        self.assertIn("The solution is X", result)

    def test_call_success_without_thinking(self):
        """Test successful tool call without thinking steps"""
        tool = ReasoningTool()

        # Mock reasoning agent response without thinking
        mock_result = {
            "thinking_steps": None,
            "final_answer": "Direct answer",
            "thinking_enabled": False,
            "domain": "general",
            "model_used": "qwen3-32b-reasoning",
        }
        self.mock_reasoning_agent.solve_problem.return_value = mock_result

        result = tool.call("Simple question")

        # Verify agent was called with default domain
        self.mock_reasoning_agent.solve_problem.assert_called_once_with("Simple question", "general")

        # Check response is just the final answer
        self.assertEqual(result, "Direct answer")

    def test_call_error_handling(self):
        """Test error handling in tool call"""
        tool = ReasoningTool()

        # Mock reasoning agent to raise exception
        self.mock_reasoning_agent.solve_problem.side_effect = Exception("Agent error")

        result = tool.call("Test query")

        # Verify error is handled gracefully
        self.assertIn("Error in reasoning", result)

    def test_metadata_content(self):
        """Test tool metadata content"""
        tool = ReasoningTool()

        metadata = tool.metadata
        self.assertEqual(metadata.name, "deep_reasoning")

        description = metadata.description
        expected_keywords = [
            "complex reasoning",
            "step-by-step thinking",
            "Mathematical problems",
            "Logical puzzles",
            "Complex analysis",
        ]

        for keyword in expected_keywords:
            self.assertIn(keyword, description)


class TestComparisonTool(unittest.TestCase):
    """Test cases for the ComparisonTool class functionality."""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_reasoning_agent = Mock()

        # Patch the get_reasoning_agent function
        self.agent_patcher = patch("tools.reasoning_tool.get_reasoning_agent")
        self.mock_get_agent = self.agent_patcher.start()
        self.mock_get_agent.return_value = self.mock_reasoning_agent

    def tearDown(self):
        """Clean up patches"""
        self.agent_patcher.stop()

    def test_init_default_model(self):
        """Test ComparisonTool initialization with default model"""
        tool = ComparisonTool()

        self.mock_get_agent.assert_called_with("qwen3-32b-reasoning")
        self.assertEqual(tool.reasoning_agent, self.mock_reasoning_agent)

        # Check metadata
        self.assertIsInstance(tool.metadata, ToolMetadata)
        self.assertEqual(tool.metadata.name, "compare_approaches")
        self.assertIn("Compare different approaches", tool.metadata.description)

    def test_call_success(self):
        """Test successful comparison tool call"""
        tool = ComparisonTool()

        # Mock reasoning agent response
        mock_result = {
            "thinking_steps": "Comparing the approaches...",
            "final_answer": "Method B is the best choice",
            "approaches_compared": ["Method A", "Method B", "Method C"],
            "model_used": "qwen3-32b-reasoning",
        }
        self.mock_reasoning_agent.compare_approaches.return_value = mock_result

        result = tool.call("Choose the best sorting algorithm", "Method A, Method B, Method C")

        # Verify agent was called correctly
        expected_approaches = ["Method A", "Method B", "Method C"]
        self.mock_reasoning_agent.compare_approaches.assert_called_once_with(
            "Choose the best sorting algorithm",
            expected_approaches,
        )

        # Verify result is a string with formatted response
        self.assertIsInstance(result, str)
        self.assertIn("**Problem Analysis:**", result)
        self.assertIn("**Comparison Result:**", result)
        self.assertIn("Method B is the best choice", result)

    def test_call_no_thinking_steps(self):
        """Test comparison tool call without thinking steps"""
        tool = ComparisonTool()

        # Mock reasoning agent response without thinking
        mock_result = {
            "thinking_steps": None,
            "final_answer": "Direct comparison result",
            "approaches_compared": ["A", "B"],
            "model_used": "qwen3-32b-reasoning",
        }
        self.mock_reasoning_agent.compare_approaches.return_value = mock_result

        result = tool.call("Test problem", "A, B")

        # Check response uses fallback text for analysis
        self.assertIn("Direct comparison performed", result)
        self.assertIn("Direct comparison result", result)

    def test_call_whitespace_handling(self):
        """Test approach parsing with whitespace"""
        tool = ComparisonTool()

        mock_result = {
            "thinking_steps": "Analysis",
            "final_answer": "Result",
            "approaches_compared": ["Method A", "Method B", "Method C"],
            "model_used": "qwen3-32b-reasoning",
        }
        self.mock_reasoning_agent.compare_approaches.return_value = mock_result

        # Test with extra whitespace
        result = tool.call("Problem", " Method A , Method B,Method C ")

        # Verify whitespace is stripped
        expected_approaches = ["Method A", "Method B", "Method C"]
        self.mock_reasoning_agent.compare_approaches.assert_called_once_with(
            "Problem",
            expected_approaches,
        )

    def test_call_error_handling(self):
        """Test error handling in comparison tool"""
        tool = ComparisonTool()

        # Mock reasoning agent to raise exception
        self.mock_reasoning_agent.compare_approaches.side_effect = Exception("Comparison error")

        result = tool.call("Problem", "A, B")

        # Verify error is handled gracefully
        self.assertIn("Error in comparison", result)

    def test_metadata_content(self):
        """Test comparison tool metadata content"""
        tool = ComparisonTool()

        metadata = tool.metadata
        self.assertEqual(metadata.name, "compare_approaches")

        description = metadata.description
        expected_keywords = [
            "Compare different approaches",
            "Evaluate multiple solution strategies",
            "pros and cons",
            "trade-offs",
            "recommendations",
        ]

        for keyword in expected_keywords:
            self.assertIn(keyword, description)


class TestReasoningToolsIntegration(unittest.TestCase):
    """Integration test cases for reasoning tools functionality."""

    @patch("tools.reasoning_tool.ReasoningTool")
    @patch("tools.reasoning_tool.ComparisonTool")
    def test_get_reasoning_tools_default(self, mock_comparison_tool, mock_reasoning_tool):
        """Test get_reasoning_tools with default model"""
        mock_reasoning_instance = Mock()
        mock_comparison_instance = Mock()

        mock_reasoning_tool.return_value = mock_reasoning_instance
        mock_comparison_tool.return_value = mock_comparison_instance

        tools = get_reasoning_tools()

        # Verify both tools are created with default model
        mock_reasoning_tool.assert_called_once_with("qwen3-32b-reasoning")
        mock_comparison_tool.assert_called_once_with("qwen3-32b-reasoning")

        # Verify return list
        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0], mock_reasoning_instance)
        self.assertEqual(tools[1], mock_comparison_instance)

    @patch("tools.reasoning_tool.ReasoningTool")
    @patch("tools.reasoning_tool.ComparisonTool")
    def test_get_reasoning_tools_custom_model(self, mock_comparison_tool, mock_reasoning_tool):
        """Test get_reasoning_tools with custom model"""
        custom_model = "custom-reasoning-model"

        tools = get_reasoning_tools(custom_model)

        # Verify both tools are created with custom model
        mock_reasoning_tool.assert_called_once_with(custom_model)
        mock_comparison_tool.assert_called_once_with(custom_model)


if __name__ == "__main__":
    unittest.main()
