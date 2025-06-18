"""Provider Management for Dynamic Model Loading.

This module provides provider-specific configurations and abstractions
for different AI model providers, integrating with LiteLLM for unified access.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
import time
from typing import Any

# Third-party imports
import litellm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Local imports
from utils.config_loader import ProviderConfig, get_api_key, get_config_loader

# Set up logging
logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Provider status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


@dataclass
class ProviderMetrics:
    """Metrics for provider performance tracking."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: float | None = None
    rate_limit_hits: int = 0
    error_count: int = 0
    last_error: str | None = None


class BaseProvider(ABC):
    """Abstract base class for all providers."""

    def __init__(self, provider_config: ProviderConfig) -> None:
        """Initialize the base provider with configuration.

        Args:
            provider_config: Configuration object for the provider.
        """
        self.config = provider_config
        self.status = ProviderStatus.INACTIVE
        self.metrics = ProviderMetrics()
        self.api_key = None
        self._session = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the provider."""
        try:
            self.api_key = get_api_key(self.config.provider_id)
            self._setup_session()
            self.status = ProviderStatus.ACTIVE
            logger.info("Initialized provider: %s", self.config.name)
        except Exception as e:
            self.status = ProviderStatus.ERROR
            self.metrics.last_error = str(e)
            logger.exception("Failed to initialize provider %s: %s", self.config.name, e)

    def _setup_session(self) -> None:
        """Set up HTTP session with retry configuration."""
        self._session = requests.Session()

        retry_strategy = Retry(
            total=self.config.retry_config.get("max_retries", 3),
            backoff_factor=self.config.retry_config.get("backoff_factor", 2.0),
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        # Set default headers
        self._session.headers.update(self.config.headers)

        # Set authentication
        auth_config = self.config.authentication
        if auth_config["type"] == "bearer_token" or auth_config["type"] == "api_key" or auth_config["type"] == "custom_header":
            self._session.headers[auth_config["header"]] = f"{auth_config['prefix']}{self.api_key}"

    def _update_metrics(self, success: bool, response_time: float, error: str | None = None) -> None:
        """Update provider metrics."""
        self.metrics.total_requests += 1
        self.metrics.last_request_time = time.time()

        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
            self.metrics.error_count += 1
            if error:
                self.metrics.last_error = error

        # Update average response time
        total_successful = self.metrics.successful_requests
        if total_successful > 0:
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (total_successful - 1) + response_time)
                / total_successful
            )

    def _check_rate_limits(self) -> bool:
        """Check if rate limits allow the request."""
        # Simple rate limiting check - can be enhanced
        if not self.config.rate_limits:
            return True

        self.config.rate_limits.get("requests_per_minute", float("inf"))
        time.time()

        # For simplicity, we'll just log rate limit hits
        # In production, implement proper rate limiting logic
        return True

    @abstractmethod
    def supports_model_type(self, model_type: str) -> bool:
        """Check if provider supports a model type."""

    @abstractmethod
    def get_litellm_model_name(self, model_id: str) -> str:
        """Convert model ID to LiteLLM format."""


class OpenRouterProvider(BaseProvider):
    """OpenRouter provider implementation."""

    def supports_model_type(self, model_type: str) -> bool:
        """Check if OpenRouter supports the specified model type.

        Args:
            model_type: The type of model to check support for.

        Returns:
            True if the model type is supported, False otherwise.
        """
        return model_type in self.config.supported_model_types

    def get_litellm_model_name(self, model_id: str) -> str:
        """Convert model ID to LiteLLM format for OpenRouter.

        Args:
            model_id: The model identifier.

        Returns:
            The LiteLLM-formatted model name.
        """
        # OpenRouter uses the format: openrouter/model_id
        return f"openrouter/{model_id}"

    def setup_litellm_config(self) -> dict[str, Any]:
        """Setup LiteLLM configuration for OpenRouter.

        Returns:
            Dictionary containing LiteLLM configuration parameters.
        """
        return {
            "api_base": self.config.base_url,
            "api_key": self.api_key,
            "headers": {
                **self.config.headers,
                "Authorization": f"Bearer {self.api_key}",
            },
            "timeout": self.config.timeout,
            "max_retries": self.config.retry_config.get("max_retries", 3),
        }


class DeepInfraProvider(BaseProvider):
    """DeepInfra provider implementation."""

    def supports_model_type(self, model_type: str) -> bool:
        """Check if DeepInfra supports the specified model type.

        Args:
            model_type: The type of model to check support for.

        Returns:
            True if the model type is supported, False otherwise.
        """
        return model_type in self.config.supported_model_types

    def get_litellm_model_name(self, model_id: str) -> str:
        """Convert model ID to LiteLLM format for DeepInfra.

        Args:
            model_id: The model identifier.

        Returns:
            The LiteLLM-formatted model name.
        """
        # DeepInfra uses the format: deepinfra/model_id
        return f"deepinfra/{model_id}"

    def setup_litellm_config(self) -> dict[str, Any]:
        """Setup LiteLLM configuration for DeepInfra.

        Returns:
            Dictionary containing LiteLLM configuration parameters.
        """
        return {
            "api_base": self.config.base_url,
            "api_key": self.api_key,
            "headers": {
                **self.config.headers,
                "Authorization": f"Bearer {self.api_key}",
            },
            "timeout": self.config.timeout,
            "max_retries": self.config.retry_config.get("max_retries", 3),
        }


class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation."""

    def supports_model_type(self, model_type: str) -> bool:
        """Check if OpenAI supports the specified model type.

        Args:
            model_type: The type of model to check support for.

        Returns:
            True if the model type is supported, False otherwise.
        """
        return model_type in self.config.supported_model_types

    def get_litellm_model_name(self, model_id: str) -> str:
        """Convert model ID to LiteLLM format for OpenAI.

        Args:
            model_id: The model identifier.

        Returns:
            The LiteLLM-formatted model name.
        """
        # OpenAI models can be used directly
        return model_id

    def setup_litellm_config(self) -> dict[str, Any]:
        """Setup LiteLLM configuration for OpenAI.

        Returns:
            Dictionary containing LiteLLM configuration parameters.
        """
        return {
            "api_base": self.config.base_url,
            "api_key": self.api_key,
            "timeout": self.config.timeout,
            "max_retries": self.config.retry_config.get("max_retries", 3),
        }


class AnthropicProvider(BaseProvider):
    """Anthropic provider implementation."""

    def supports_model_type(self, model_type: str) -> bool:
        """Check if Anthropic supports the specified model type.

        Args:
            model_type: The type of model to check support for.

        Returns:
            True if the model type is supported, False otherwise.
        """
        return model_type in self.config.supported_model_types

    def get_litellm_model_name(self, model_id: str) -> str:
        """Convert model ID to LiteLLM format for Anthropic.

        Args:
            model_id: The model identifier.

        Returns:
            The LiteLLM-formatted model name.
        """
        # Anthropic uses the format: claude-3-sonnet-20240229
        return model_id

    def setup_litellm_config(self) -> dict[str, Any]:
        """Setup LiteLLM configuration for Anthropic.

        Returns:
            Dictionary containing LiteLLM configuration parameters.
        """
        return {
            "api_base": self.config.base_url,
            "api_key": self.api_key,
            "headers": {
                **self.config.headers,
                "x-api-key": self.api_key,
            },
            "timeout": self.config.timeout,
            "max_retries": self.config.retry_config.get("max_retries", 3),
        }


class HuggingFaceProvider(BaseProvider):
    """Hugging Face provider implementation."""

    def supports_model_type(self, model_type: str) -> bool:
        """Check if Hugging Face supports the specified model type.

        Args:
            model_type: The type of model to check support for.

        Returns:
            True if the model type is supported, False otherwise.
        """
        return model_type in self.config.supported_model_types

    def get_litellm_model_name(self, model_id: str) -> str:
        """Convert model ID to LiteLLM format for Hugging Face.

        Args:
            model_id: The model identifier.

        Returns:
            The LiteLLM-formatted model name.
        """
        # HuggingFace uses the format: huggingface/model_id
        return f"huggingface/{model_id}"

    def setup_litellm_config(self) -> dict[str, Any]:
        """Setup LiteLLM configuration for Hugging Face.

        Returns:
            Dictionary containing LiteLLM configuration parameters.
        """
        return {
            "api_base": self.config.base_url,
            "api_key": self.api_key,
            "headers": {
                **self.config.headers,
                "Authorization": f"Bearer {self.api_key}",
            },
            "timeout": self.config.timeout,
            "max_retries": self.config.retry_config.get("max_retries", 3),
        }


class ProviderManager:
    """Manager for all providers."""

    def __init__(self) -> None:
        """Initialize the provider manager and load all configured providers."""
        self.providers: dict[str, BaseProvider] = {}
        self.config_loader = get_config_loader()
        self._provider_classes = {
            "openrouter": OpenRouterProvider,
            "deepinfra": DeepInfraProvider,
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "huggingface": HuggingFaceProvider,
        }
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize all configured providers."""
        for provider_name in self.config_loader.list_providers():
            try:
                provider_config = self.config_loader.get_provider_config(provider_name)
                provider_class = self._provider_classes.get(provider_name)

                if provider_class:
                    provider = provider_class(provider_config)
                    self.providers[provider_name] = provider
                    logger.info("Initialized provider: %s", provider_name)
                else:
                    logger.warning("No provider class found for: %s", provider_name)

            except Exception as e:
                logger.exception("Failed to initialize provider %s: %s", provider_name, e)

    def get_provider(self, provider_name: str) -> BaseProvider | None:
        """Get a provider by name."""
        return self.providers.get(provider_name)

    def get_active_providers(self) -> dict[str, BaseProvider]:
        """Get all active providers."""
        return {
            name: provider for name, provider in self.providers.items()
            if provider.status == ProviderStatus.ACTIVE
        }

    def get_provider_for_model_type(self, model_type: str) -> list[str]:
        """Get providers that support a specific model type."""
        return [
            name for name, provider in self.providers.items()
            if provider.supports_model_type(model_type) and provider.status == ProviderStatus.ACTIVE
        ]

    def get_provider_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all providers."""
        status = {}
        for name, provider in self.providers.items():
            status[name] = {
                "status": provider.status.value,
                "metrics": {
                    "total_requests": provider.metrics.total_requests,
                    "successful_requests": provider.metrics.successful_requests,
                    "failed_requests": provider.metrics.failed_requests,
                    "average_response_time": provider.metrics.average_response_time,
                    "error_count": provider.metrics.error_count,
                    "last_error": provider.metrics.last_error,
                },
                "config": {
                    "name": provider.config.name,
                    "description": provider.config.description,
                    "supported_model_types": provider.config.supported_model_types,
                },
            }
        return status

    def setup_litellm_for_provider(self, provider_name: str) -> dict[str, Any]:
        """Setup LiteLLM configuration for a specific provider."""
        provider = self.get_provider(provider_name)
        if not provider:
            msg = f"Provider '{provider_name}' not found"
            raise ValueError(msg)

        if provider.status != ProviderStatus.ACTIVE:
            msg = f"Provider '{provider_name}' is not active"
            raise ValueError(msg)

        return provider.setup_litellm_config()

    def get_litellm_model_name(self, provider_name: str, model_id: str) -> str:
        """Get LiteLLM formatted model name."""
        provider = self.get_provider(provider_name)
        if not provider:
            msg = f"Provider '{provider_name}' not found"
            raise ValueError(msg)

        return provider.get_litellm_model_name(model_id)

    def validate_provider_setup(self, provider_name: str) -> tuple[bool, str | None]:
        """Validate that a provider is properly set up."""
        try:
            provider = self.get_provider(provider_name)
            if not provider:
                return False, f"Provider '{provider_name}' not found"

            if provider.status != ProviderStatus.ACTIVE:
                return False, f"Provider '{provider_name}' is not active: {provider.metrics.last_error}"

            if not provider.api_key:
                return False, f"API key not configured for provider '{provider_name}'"

            return True, None

        except Exception as e:
            return False, str(e)

    def refresh_provider(self, provider_name: str) -> None:
        """Refresh a specific provider."""
        if provider_name in self.providers:
            try:
                provider_config = self.config_loader.get_provider_config(provider_name)
                provider_class = self._provider_classes.get(provider_name)

                if provider_class:
                    self.providers[provider_name] = provider_class(provider_config)
                    logger.info("Refreshed provider: %s", provider_name)

            except Exception as e:
                logger.exception("Failed to refresh provider %s: %s", provider_name, e)

    def refresh_all_providers(self) -> None:
        """Refresh all providers."""
        logger.info("Refreshing all providers...")
        self.providers.clear()
        self._initialize_providers()


# Global provider manager instance
_provider_manager = None


def get_provider_manager() -> ProviderManager:
    """Get the global provider manager instance."""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = ProviderManager()
    return _provider_manager


def setup_litellm_logging() -> None:
    """Setup LiteLLM logging configuration."""
    # Configure LiteLLM logging
    litellm.set_verbose = False  # Set to True for debugging

    # Set custom logging level for LiteLLM
    logging.getLogger("litellm").setLevel(logging.WARNING)

    # Setup custom success and failure callbacks if needed
    def log_success(kwargs) -> None:
        logger.debug("LiteLLM request successful: %s", kwargs.get("model", "unknown"))

    def log_failure(kwargs) -> None:
        logger.warning("LiteLLM request failed: %s - %s", kwargs.get("model", "unknown"), kwargs.get("exception", "unknown error"))

    # Set callbacks
    litellm.success_callback = [log_success]
    litellm.failure_callback = [log_failure]


# Initialize LiteLLM logging on module import
setup_litellm_logging()


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Initialize provider manager
        pm = get_provider_manager()

        # Test provider status
        status = pm.get_provider_status()
        for _name, _data in status.items():
            pass

        # Test provider validation
        for provider_name in pm.providers:
            is_valid, error = pm.validate_provider_setup(provider_name)

        # Test model type support
        chat_providers = pm.get_provider_for_model_type("chat")
        embedding_providers = pm.get_provider_for_model_type("embedding")


    except Exception:
        raise
