"""Tests for the reasoning agent functionality.

This module contains tests for the ReasoningAgent class and its
advanced reasoning capabilities with step-by-step analysis.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.reasoning_agent import ReasoningAgent, get_reasoning_agent


class TestReasoningAgent(unittest.TestCase):
    """Test cases for the ReasoningAgent class functionality."""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = Mock()
        self.mock_factory = Mock()
        self.mock_llm = Mock()

        # Mock the global functions
        self.config_patcher = patch("agents.reasoning_agent.get_config_loader")
        self.factory_patcher = patch("agents.reasoning_agent.get_model_factory")

        self.mock_get_config = self.config_patcher.start()
        self.mock_get_factory = self.factory_patcher.start()

        self.mock_get_config.return_value = self.mock_config
        self.mock_get_factory.return_value = self.mock_factory
        self.mock_factory.get_chat_model.return_value = self.mock_llm

        # Set up mock LLM attributes
        self.mock_llm.temperature = 0.7
        self.mock_llm.top_p = 0.9
        self.mock_llm.top_k = 10
        self.mock_llm.min_p = 0.1
        self.mock_llm.max_tokens = 4000

    def tearDown(self):
        """Clean up patches"""
        self.config_patcher.stop()
        self.factory_patcher.stop()

    def test_init_default_model(self):
        """Test ReasoningAgent initialization with default model"""
        agent = ReasoningAgent()

        self.mock_factory.get_chat_model.assert_called_with("qwen3-32b")
        self.assertEqual(agent.model_name, "qwen3-32b")
        self.assertEqual(agent.llm, self.mock_llm)

        # Check reasoning parameters are set
        self.assertEqual(self.mock_llm.temperature, 0.6)
        self.assertEqual(self.mock_llm.top_p, 0.95)
        self.assertEqual(self.mock_llm.top_k, 20)
        self.assertEqual(self.mock_llm.min_p, 0)
        self.assertEqual(self.mock_llm.max_tokens, 8000)

    def test_init_custom_model(self):
        """Test ReasoningAgent initialization with custom model"""
        agent = ReasoningAgent("qwen3-30b-reasoning")

        self.mock_factory.get_chat_model.assert_called_with("qwen3-30b-reasoning")
        self.assertEqual(agent.model_name, "qwen3-30b-reasoning")

    def test_get_reasoning_system_prompt_thinking_enabled(self):
        """Test system prompt generation with thinking enabled"""
        agent = ReasoningAgent()
        prompt = agent._get_reasoning_system_prompt(True)

        self.assertIn("intelligent reasoning assistant", prompt)
        self.assertIn("<think>", prompt)
        self.assertIn("step by step", prompt)
        self.assertIn("reasoning process", prompt)

    def test_get_reasoning_system_prompt_thinking_disabled(self):
        """Test system prompt generation with thinking disabled"""
        agent = ReasoningAgent()
        prompt = agent._get_reasoning_system_prompt(False)

        self.assertIn("intelligent reasoning assistant", prompt)
        self.assertNotIn("<think>", prompt)
        self.assertIn("direct, concise answers", prompt)

    def test_parse_response_with_thinking(self):
        """Test parsing response with thinking blocks"""
        agent = ReasoningAgent()
        response = """<think>
Let me think about this step by step.
First, I need to understand the problem.
Then I should consider the options.
</think>

Based on my analysis, the answer is 42."""

        thinking, answer = agent._parse_response(response)

        self.assertIsNotNone(thinking)
        self.assertIn("step by step", thinking)
        self.assertIn("understand the problem", thinking)
        self.assertEqual(answer.strip(), "Based on my analysis, the answer is 42.")

    def test_parse_response_without_thinking(self):
        """Test parsing response without thinking blocks"""
        agent = ReasoningAgent()
        response = "The answer is 42."

        thinking, answer = agent._parse_response(response)

        self.assertIsNone(thinking)
        self.assertEqual(answer, "The answer is 42.")

    def test_parse_response_multiple_thinking_blocks(self):
        """Test parsing response with multiple thinking blocks"""
        agent = ReasoningAgent()
        response = """<think>First thought</think>
Some text.
<think>Second thought</think>
Final answer."""

        thinking, answer = agent._parse_response(response)

        self.assertIn("First thought", thinking)
        self.assertIn("Second thought", thinking)
        self.assertEqual(answer.strip(), "Some text.\n\nFinal answer.")

    def test_think_success(self):
        """Test successful thinking operation"""
        agent = ReasoningAgent()

        # Mock LLM response
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="""<think>
This is a math problem. Let me solve it step by step.
2 + 2 = 4
</think>

The answer is 4.""")

        self.mock_llm.chat.return_value = mock_response

        result = agent.think("What is 2 + 2?")

        # Verify LLM was called with correct messages
        self.mock_llm.chat.assert_called_once()
        call_args = self.mock_llm.chat.call_args[0][0]
        self.assertEqual(len(call_args), 2)  # system + user message
        self.assertEqual(call_args[1].content, "What is 2 + 2?")

        # Verify result structure
        self.assertEqual(result["query"], "What is 2 + 2?")
        self.assertTrue(result["thinking_enabled"])
        self.assertIn("step by step", result["thinking_steps"])
        self.assertEqual(result["final_answer"], "The answer is 4.")
        self.assertEqual(result["model_used"], "qwen3-32b")

    def test_think_error_handling(self):
        """Test error handling in think method"""
        agent = ReasoningAgent()

        # Mock LLM to raise exception
        self.mock_llm.chat.side_effect = Exception("Model error")

        result = agent.think("Test query")

        self.assertEqual(result["query"], "Test query")
        self.assertIsNone(result["thinking_steps"])
        self.assertIn("Error occurred during reasoning", result["final_answer"])
        self.assertIsNone(result["full_response"])

    def test_solve_problem_math_domain(self):
        """Test solve_problem with math domain"""
        agent = ReasoningAgent()

        # Mock successful think call
        with patch.object(agent, "think") as mock_think:
            mock_think.return_value = {
                "query": "Enhanced query",
                "thinking_enabled": True,
                "thinking_steps": "Math reasoning",
                "final_answer": "42",
                "model_used": "qwen3-32b",
            }

            result = agent.solve_problem("2 + 2", "math")

            # Verify enhanced query was used
            call_args = mock_think.call_args[0]
            self.assertIn("mathematical problem", call_args[0])
            self.assertIn("2 + 2", call_args[0])

            # Verify domain is added
            self.assertEqual(result["domain"], "math")

    def test_compare_approaches(self):
        """Test compare_approaches method"""
        agent = ReasoningAgent()

        approaches = ["Method A", "Method B", "Method C"]

        with patch.object(agent, "think") as mock_think:
            mock_think.return_value = {
                "query": "Comparison query",
                "thinking_enabled": True,
                "thinking_steps": "Comparison reasoning",
                "final_answer": "Method B is best",
                "model_used": "qwen3-32b",
            }

            result = agent.compare_approaches("Test problem", approaches)

            # Verify comparison query structure
            call_args = mock_think.call_args[0]
            query = call_args[0]
            self.assertIn("Test problem", query)
            self.assertIn("Method A", query)
            self.assertIn("Method B", query)
            self.assertIn("Method C", query)

            # Verify approaches are included in result
            self.assertEqual(result["approaches_compared"], approaches)

    def test_get_model_info(self):
        """Test get_model_info method"""
        agent = ReasoningAgent()

        info = agent.get_model_info()

        expected_keys = ["model_name", "temperature", "top_p", "max_tokens", "supports_thinking"]
        for key in expected_keys:
            self.assertIn(key, info)

        self.assertEqual(info["model_name"], "qwen3-32b")
        self.assertTrue(info["supports_thinking"])


class TestReasoningAgentGlobal(unittest.TestCase):
    """Test cases for global reasoning agent instance management."""

    def setUp(self):
        """Set up test fixtures"""
        # Reset global instance
        import agents.reasoning_agent
        agents.reasoning_agent._reasoning_agent = None

    @patch("agents.reasoning_agent.ReasoningAgent")
    def test_get_reasoning_agent_singleton(self, mock_reasoning_agent_class):
        """Test global reasoning agent singleton behavior"""
        mock_agent = Mock()
        mock_agent.model_name = "qwen3-32b"
        mock_reasoning_agent_class.return_value = mock_agent

        # First call should create instance
        agent1 = get_reasoning_agent()
        mock_reasoning_agent_class.assert_called_once_with("qwen3-32b")

        # Second call should return same instance
        agent2 = get_reasoning_agent()
        self.assertEqual(mock_reasoning_agent_class.call_count, 1)  # Still only called once
        self.assertEqual(agent1, agent2)

    @patch("agents.reasoning_agent.ReasoningAgent")
    def test_get_reasoning_agent_different_model(self, mock_reasoning_agent_class):
        """Test global reasoning agent with different model"""
        mock_agent1 = Mock()
        mock_agent1.model_name = "qwen3-32b"
        mock_agent2 = Mock()
        mock_agent2.model_name = "qwen3-30b"

        mock_reasoning_agent_class.side_effect = [mock_agent1, mock_agent2]

        # First call with default model
        agent1 = get_reasoning_agent()

        # Second call with different model should create new instance
        agent2 = get_reasoning_agent("qwen3-30b")

        self.assertEqual(mock_reasoning_agent_class.call_count, 2)
        self.assertNotEqual(agent1, agent2)


if __name__ == "__main__":
    unittest.main()
