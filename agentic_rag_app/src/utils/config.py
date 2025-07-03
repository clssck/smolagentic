"""Configuration module for the Agentic RAG application."""

import os
from typing import ClassVar

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration class for environment variables and settings."""

    # Qdrant configuration
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "agentic_rag")

    # OpenRouter configuration
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    # Model configurations - Mistral prioritized for function calling excellence
    MODELS: ClassVar[dict] = {
        "openrouter/mistralai/mistral-small-3.2-24b-instruct": {
            "priority": 1,  # Highest priority - excellent function calling
            "max_tokens": 8192,
            "context_window": 32768,
            "performance_score": 0.95,
            "function_calling": "excellent",
            "cost_efficiency": "high",
            "recommended_use": ["manager", "coordination", "rag"]
        },
        "openrouter/mistralai/mistral-medium-2312": {
            "priority": 2,
            "max_tokens": 8192,
            "context_window": 32768,
            "performance_score": 0.92,
            "function_calling": "excellent",
            "recommended_use": ["research", "complex_reasoning"]
        },
        "openrouter/mistralai/mistral-large-2407": {
            "priority": 3,
            "max_tokens": 16384,
            "context_window": 128000,
            "performance_score": 0.98,
            "function_calling": "excellent",
            "recommended_use": ["complex_tasks", "long_context"]
        },
        "qwen/qwen-2.5-7b-instruct": {
            "priority": 4,  # Fallback option
            "max_tokens": 4000,
            "context_window": 8000,
            "performance_score": 0.75,
            "function_calling": "good",
            "recommended_use": ["simple_tasks", "fallback"]
        },
    }

    # DeepInfra configuration
    DEEPINFRA_TOKEN = os.getenv("DEEPINFRA_API_KEY")
    DEEPINFRA_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

    # Agent configuration
    MAX_AGENT_STEPS = int(
        os.getenv("MAX_AGENT_STEPS", "5")
    )  # Optimized for efficiency - most queries need 1-3 steps
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Streaming configuration - disabled by default due to compatibility issues
    ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "false").lower() == "true"
    STREAM_VERBOSITY = int(
        os.getenv("STREAM_VERBOSITY", "1")
    )  # 0=minimal, 1=normal, 2=verbose

    # UI configuration
    GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    @classmethod
    def validate(cls) -> bool:
        """Validate that at least one API key is available."""
        # Check for any available API keys
        available_keys = []

        if cls.OPENROUTER_API_KEY:
            available_keys.append("OPENROUTER_API_KEY")
        if cls.DEEPINFRA_TOKEN:
            available_keys.append("DEEPINFRA_API_KEY")
        if os.getenv("OPENAI_API_KEY"):
            available_keys.append("OPENAI_API_KEY")
        if os.getenv("GROQ_API_KEY"):
            available_keys.append("GROQ_API_KEY")

        # Require at least one API key for LLM
        if not available_keys:
            raise ValueError(
                "No API keys found. Please set at least one of: "
                "OPENROUTER_API_KEY, DEEPINFRA_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY"
            )

        # Qdrant is optional - can use local embeddings
        if not cls.QDRANT_URL or not cls.QDRANT_API_KEY:
            print("⚠️  Qdrant not configured - will use local embeddings")

        print(f"✅ Found API keys: {', '.join(available_keys)}")
        return True
