"""Integration tests for the RAG agent functionality.

This module contains comprehensive tests for the AgenticRAG class,
testing integration between components and core functionality.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.rag_agent import AgenticRAG, get_agentic_rag
from llama_index.core.tools import QueryEngineTool


class TestRAGReasoningIntegration(unittest.TestCase):
    """Test cases for RAG agent integration with reasoning capabilities."""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = Mock()
        self.mock_factory = Mock()
        self.mock_qdrant = Mock()
        self.mock_chat_model = Mock()
        self.mock_embed_model = Mock()
        self.mock_query_engine = Mock()
        self.mock_reasoning_tools = [Mock(), Mock()]

        # Patch all dependencies
        self.config_patcher = patch("agents.rag_agent.get_config_loader")
        self.factory_patcher = patch("agents.rag_agent.get_model_factory")
        self.qdrant_patcher = patch("agents.rag_agent.get_qdrant_store")
        self.tools_patcher = patch("agents.rag_agent.get_reasoning_tools")
        self.agent_patcher = patch("agents.rag_agent.ReActAgent")
        self.memory_patcher = patch("agents.rag_agent.ChatMemoryBuffer")

        self.mock_get_config = self.config_patcher.start()
        self.mock_get_factory = self.factory_patcher.start()
        self.mock_get_qdrant = self.qdrant_patcher.start()
        self.mock_get_tools = self.tools_patcher.start()
        self.mock_react_agent = self.agent_patcher.start()
        self.mock_memory_buffer = self.memory_patcher.start()

        # Configure mocks
        self.mock_get_config.return_value = self.mock_config
        self.mock_get_factory.return_value = self.mock_factory
        self.mock_get_qdrant.return_value = self.mock_qdrant
        self.mock_get_tools.return_value = self.mock_reasoning_tools

        self.mock_factory.get_chat_model.return_value = self.mock_chat_model
        self.mock_factory.get_embedding_model.return_value = self.mock_embed_model
        self.mock_qdrant.get_query_engine.return_value = self.mock_query_engine

        # Configure agent and memory
        self.mock_agent_instance = Mock()
        self.mock_react_agent.from_tools.return_value = self.mock_agent_instance
        self.mock_memory_instance = Mock()
        self.mock_memory_buffer.from_defaults.return_value = self.mock_memory_instance

    def tearDown(self):
        """Clean up patches"""
        self.config_patcher.stop()
        self.factory_patcher.stop()
        self.qdrant_patcher.stop()
        self.tools_patcher.stop()
        self.agent_patcher.stop()
        self.memory_patcher.stop()

    @patch.dict(os.environ, {"DEFAULT_CHAT_MODEL": "qwen3-32b", "DEFAULT_EMBEDDING_MODEL": "qwen3-embed"})
    def test_init_with_reasoning_enabled(self):
        """Test AgenticRAG initialization with reasoning enabled"""
        rag = AgenticRAG(enable_reasoning=True)

        # Verify models were loaded
        self.mock_factory.get_chat_model.assert_called_with("qwen3-32b")
        self.mock_factory.get_embedding_model.assert_called_with("qwen3-embed")

        # Verify reasoning tools were requested
        self.mock_get_tools.assert_called_once_with("qwen3-32b-reasoning")

        # Verify agent was created with correct tools
        self.mock_react_agent.from_tools.assert_called_once()
        call_args = self.mock_react_agent.from_tools.call_args
        tools_arg = call_args[1]["tools"]

        # Should have RAG tool + 2 reasoning tools = 3 tools
        self.assertEqual(len(tools_arg), 3)

        # First tool should be QueryEngineTool (RAG)
        self.assertIsInstance(tools_arg[0], QueryEngineTool)

        # Next two should be reasoning tools
        self.assertEqual(tools_arg[1], self.mock_reasoning_tools[0])
        self.assertEqual(tools_arg[2], self.mock_reasoning_tools[1])

        # Verify system prompt includes reasoning capabilities
        system_prompt = call_args[1]["system_prompt"]
        self.assertIn("advanced reasoning capabilities", system_prompt)
        self.assertIn("deep_reasoning tool", system_prompt)
        self.assertIn("compare_approaches tool", system_prompt)

    @patch.dict(os.environ, {"DEFAULT_CHAT_MODEL": "qwen3-32b", "DEFAULT_EMBEDDING_MODEL": "qwen3-embed"})
    def test_init_with_reasoning_disabled(self):
        """Test AgenticRAG initialization with reasoning disabled"""
        rag = AgenticRAG(enable_reasoning=False)

        # Verify reasoning tools were NOT requested
        self.mock_get_tools.assert_not_called()

        # Verify agent was created with only RAG tool
        self.mock_react_agent.from_tools.assert_called_once()
        call_args = self.mock_react_agent.from_tools.call_args
        tools_arg = call_args[1]["tools"]

        # Should have only 1 tool (RAG)
        self.assertEqual(len(tools_arg), 1)
        self.assertIsInstance(tools_arg[0], QueryEngineTool)

        # Verify system prompt does NOT include reasoning capabilities
        system_prompt = call_args[1]["system_prompt"]
        self.assertNotIn("advanced reasoning capabilities", system_prompt)
        self.assertNotIn("deep_reasoning tool", system_prompt)
        self.assertNotIn("compare_approaches tool", system_prompt)

    @patch.dict(os.environ, {"DEFAULT_CHAT_MODEL": "qwen3-32b", "DEFAULT_EMBEDDING_MODEL": "qwen3-embed"})
    def test_chat_method(self):
        """Test chat method functionality"""
        rag = AgenticRAG(enable_reasoning=True)

        # Mock agent response
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="Test response")
        self.mock_agent_instance.chat.return_value = mock_response

        result = rag.chat("Test message")

        # Verify agent was called
        self.mock_agent_instance.chat.assert_called_once_with("Test message")
        self.assertEqual(result, "Test response")

    @patch.dict(os.environ, {"DEFAULT_CHAT_MODEL": "qwen3-32b", "DEFAULT_EMBEDDING_MODEL": "qwen3-embed"})
    def test_chat_error_handling(self):
        """Test chat method error handling"""
        rag = AgenticRAG(enable_reasoning=True)

        # Mock agent to raise exception
        self.mock_agent_instance.chat.side_effect = Exception("Chat error")

        result = rag.chat("Test message")

        self.assertIn("encountered an error", result)
        self.assertIn("Chat error", result)

    @patch.dict(os.environ, {"DEFAULT_CHAT_MODEL": "qwen3-32b", "DEFAULT_EMBEDDING_MODEL": "qwen3-embed"})
    def test_get_system_status(self):
        """Test get_system_status method"""
        rag = AgenticRAG(enable_reasoning=True)

        # Mock factory and qdrant responses
        self.mock_factory.list_available_models.side_effect = lambda model_type: ["model1", "model2"]
        self.mock_qdrant.get_collection_info.return_value = {"collection": "test"}
        self.mock_memory_instance.get_all.return_value = ["msg1", "msg2"]

        # Mock model attributes
        self.mock_chat_model.model = "qwen3-32b"
        self.mock_embed_model.model_name = "qwen3-embed"

        status = rag.get_system_status()

        # Verify status contains expected keys
        expected_keys = [
            "current_chat_model", "current_embedding_model",
            "available_chat_models", "available_embedding_models",
            "qdrant_collection", "chat_history_length",
        ]
        for key in expected_keys:
            self.assertIn(key, status)

        # Verify values
        self.assertEqual(status["current_chat_model"], "qwen3-32b")
        self.assertEqual(status["current_embedding_model"], "qwen3-embed")
        self.assertEqual(status["chat_history_length"], 2)

    @patch.dict(os.environ, {"DEFAULT_CHAT_MODEL": "qwen3-32b", "DEFAULT_EMBEDDING_MODEL": "qwen3-embed"})
    def test_switch_chat_model(self):
        """Test switch_chat_model method"""
        rag = AgenticRAG(enable_reasoning=True)

        # Mock new model
        new_model = Mock()
        self.mock_factory.get_chat_model.return_value = new_model

        # Store initial call count
        initial_calls = self.mock_react_agent.from_tools.call_count

        rag.switch_chat_model("new-model")

        # Verify new model was loaded (should be called twice now - initial + new model)
        self.assertEqual(self.mock_factory.get_chat_model.call_count, 2)

        # Verify agent was recreated
        self.assertEqual(self.mock_react_agent.from_tools.call_count, initial_calls + 1)

    @patch.dict(os.environ, {"DEFAULT_CHAT_MODEL": "qwen3-32b", "DEFAULT_EMBEDDING_MODEL": "qwen3-embed"})
    def test_clear_history(self):
        """Test clear_history method"""
        rag = AgenticRAG(enable_reasoning=True)

        rag.clear_history()

        self.mock_memory_instance.reset.assert_called_once()

    @patch.dict(os.environ, {"DEFAULT_CHAT_MODEL": "qwen3-32b", "DEFAULT_EMBEDDING_MODEL": "qwen3-embed"})
    def test_search_knowledge_base(self):
        """Test search_knowledge_base method"""
        rag = AgenticRAG(enable_reasoning=True)

        # Mock search results
        mock_results = [{"content": "result1"}, {"content": "result2"}]
        self.mock_qdrant.search.return_value = mock_results

        results = rag.search_knowledge_base("test query", top_k=3)

        self.mock_qdrant.search.assert_called_once_with("test query", 3)
        self.assertEqual(results, mock_results)


class TestGlobalRAGInstance(unittest.TestCase):
    """Test cases for the global RAG agent instance management."""

    def setUp(self):
        """Reset global instance before each test"""
        import agents.rag_agent
        agents.rag_agent._agentic_rag = None

    @patch("agents.rag_agent.AgenticRAG")
    def test_get_agentic_rag_singleton(self, mock_agentic_rag_class):
        """Test global agentic RAG singleton behavior"""
        mock_rag = Mock()
        mock_agentic_rag_class.return_value = mock_rag

        # First call should create instance
        rag1 = get_agentic_rag()
        mock_agentic_rag_class.assert_called_once_with(
            # qdrant_url will use environment variable
            enable_reasoning=True,
        )

        # Second call should return same instance
        rag2 = get_agentic_rag()
        self.assertEqual(mock_agentic_rag_class.call_count, 1)  # Still only called once
        self.assertEqual(rag1, rag2)

    @patch("agents.rag_agent.AgenticRAG")
    def test_get_agentic_rag_custom_params(self, mock_agentic_rag_class):
        """Test global agentic RAG with custom parameters"""
        mock_rag = Mock()
        mock_agentic_rag_class.return_value = mock_rag

        rag = get_agentic_rag(
            qdrant_url="http://custom:6333",
            enable_reasoning=False,
        )

        mock_agentic_rag_class.assert_called_once_with(
            qdrant_url="http://custom:6333",
            enable_reasoning=False,
        )


class TestToolIntegration(unittest.TestCase):
    """Test specific tool integration scenarios"""

    def setUp(self):
        """Set up mocks for tool integration tests"""
        self.mock_reasoning_tool = Mock()
        self.mock_comparison_tool = Mock()

        # Set up tool metadata
        self.mock_reasoning_tool.metadata.name = "deep_reasoning"
        self.mock_comparison_tool.metadata.name = "compare_approaches"

        self.tools_patcher = patch("agents.rag_agent.get_reasoning_tools")
        self.mock_get_tools = self.tools_patcher.start()
        self.mock_get_tools.return_value = [self.mock_reasoning_tool, self.mock_comparison_tool]

    def tearDown(self):
        """Clean up patches"""
        self.tools_patcher.stop()

    @patch("agents.rag_agent.get_config_loader")
    @patch("agents.rag_agent.get_model_factory")
    @patch("agents.rag_agent.get_qdrant_store")
    @patch("agents.rag_agent.ReActAgent")
    @patch("agents.rag_agent.ChatMemoryBuffer")
    @patch.dict(os.environ, {"DEFAULT_CHAT_MODEL": "qwen3-32b", "DEFAULT_EMBEDDING_MODEL": "qwen3-embed"})
    def test_reasoning_tools_added_to_agent(self, mock_memory, mock_agent, mock_qdrant,
                                          mock_factory, mock_config):
        """Test that reasoning tools are properly added to the agent"""
        # Set up mocks
        mock_config.return_value = Mock()
        mock_factory.return_value = Mock()
        mock_qdrant.return_value = Mock()

        mock_factory.return_value.get_chat_model.return_value = Mock()
        mock_factory.return_value.get_embedding_model.return_value = Mock()
        mock_qdrant.return_value.get_query_engine.return_value = Mock()

        mock_memory.from_defaults.return_value = Mock()
        mock_agent.from_tools.return_value = Mock()

        # Create RAG instance with reasoning enabled
        rag = AgenticRAG(enable_reasoning=True)

        # Verify tools were requested with correct model
        self.mock_get_tools.assert_called_once_with("qwen3-32b-reasoning")

        # Verify agent was created with all tools
        mock_agent.from_tools.assert_called_once()
        call_args = mock_agent.from_tools.call_args[1]
        tools = call_args["tools"]

        # Should have 3 tools: RAG + 2 reasoning tools
        self.assertEqual(len(tools), 3)

        # Check that reasoning tools are included
        tool_objects = tools[1:]  # Skip RAG tool
        self.assertIn(self.mock_reasoning_tool, tool_objects)
        self.assertIn(self.mock_comparison_tool, tool_objects)


if __name__ == "__main__":
    unittest.main()
