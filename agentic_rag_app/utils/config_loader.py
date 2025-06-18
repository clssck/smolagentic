"""Configuration loader for model and provider settings.

This module provides centralized configuration management for language models,
embedding models, and API providers from YAML configuration files.
"""

from enum import Enum
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Enumeration for different types of models."""
    CHAT = "chat"
    EMBEDDING = "embedding"

class ConfigLoader:
    """Loader for model and provider configuration from YAML files."""
    
    def __init__(self, config_dir: str = "config") -> None:
        """Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files.
        """
        self.config_dir = Path(config_dir)
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = self.config_dir / "models.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file) as f:
            return yaml.safe_load(f)

    def get_model_config(self, model_name: str, model_type: ModelType) -> dict[str, Any]:
        """Get configuration for a specific model"""
        type_key = model_type.value
        models = self.config["models"][type_key]

        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found in {type_key} models")

        return models[model_name]

    def get_provider_config(self, provider_name: str) -> dict[str, Any]:
        """Get configuration for a provider"""
        providers = self.config["providers"]

        if provider_name not in providers:
            raise ValueError(f"Provider '{provider_name}' not found")

        return providers[provider_name]

    def get_api_key(self, provider_name: str) -> str:
        """Get API key for provider from environment"""
        provider_config = self.get_provider_config(provider_name)
        api_key_env = provider_config["api_key_env"]
        api_key = os.getenv(api_key_env)

        if not api_key:
            raise ValueError(f"API key not found in environment variable: {api_key_env}")

        return api_key

    def list_models(self, model_type: ModelType) -> list[str]:
        """List available models of given type"""
        return list(self.config["models"][model_type.value].keys())

# Global instance
_config_loader = None

def get_config_loader() -> ConfigLoader:
    """Get or create a singleton ConfigLoader instance.
    
    Returns:
        The ConfigLoader singleton instance.
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader
