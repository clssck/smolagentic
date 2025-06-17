"""
Provider Management for Dynamic Model Loading

This module provides provider-specific configurations and abstractions
for different AI model providers, integrating with LiteLLM for unified access.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from abc import ABC, abstractmethod

# Third-party imports
import litellm
from litellm import completion, acompletion, embedding, aembedding
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Local imports
from utils.config_loader import ProviderConfig, get_config_loader, get_api_key

# Set up logging
logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Provider status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


@dataclass
class ProviderMetrics:
    """Metrics for provider performance tracking"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[float] = None
    rate_limit_hits: int = 0
    error_count: int = 0
    last_error: Optional[str] = None


class BaseProvider(ABC):
    """Abstract base class for all providers"""
    
    def __init__(self, provider_config: ProviderConfig):
        self.config = provider_config
        self.status = ProviderStatus.INACTIVE
        self.metrics = ProviderMetrics()
        self.api_key = None
        self._session = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the provider"""
        try:
            self.api_key = get_api_key(self.config.provider_id)
            self._setup_session()
            self.status = ProviderStatus.ACTIVE
            logger.info(f"Initialized provider: {self.config.name}")
        except Exception as e:
            self.status = ProviderStatus.ERROR
            self.metrics.last_error = str(e)
            logger.error(f"Failed to initialize provider {self.config.name}: {e}")
    
    def _setup_session(self):
        """Set up HTTP session with retry configuration"""
        self._session = requests.Session()
        
        retry_strategy = Retry(
            total=self.config.retry_config.get('max_retries', 3),
            backoff_factor=self.config.retry_config.get('backoff_factor', 2.0),
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        
        # Set default headers
        self._session.headers.update(self.config.headers)
        
        # Set authentication
        auth_config = self.config.authentication
        if auth_config['type'] == 'bearer_token':
            self._session.headers[auth_config['header']] = f"{auth_config['prefix']}{self.api_key}"
        elif auth_config['type'] == 'api_key':
            self._session.headers[auth_config['header']] = f"{auth_config['prefix']}{self.api_key}"
        elif auth_config['type'] == 'custom_header':
            self._session.headers[auth_config['header']] = f"{auth_config['prefix']}{self.api_key}"
    
    def _update_metrics(self, success: bool, response_time: float, error: Optional[str] = None):
        """Update provider metrics"""
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
        """Check if rate limits allow the request"""
        # Simple rate limiting check - can be enhanced
        if not self.config.rate_limits:
            return True
        
        requests_per_minute = self.config.rate_limits.get('requests_per_minute', float('inf'))
        current_time = time.time()
        
        # For simplicity, we'll just log rate limit hits
        # In production, implement proper rate limiting logic
        return True
    
    @abstractmethod
    def supports_model_type(self, model_type: str) -> bool:
        """Check if provider supports a model type"""
        pass
    
    @abstractmethod
    def get_litellm_model_name(self, model_id: str) -> str:
        """Convert model ID to LiteLLM format"""
        pass


class OpenRouterProvider(BaseProvider):
    """OpenRouter provider implementation"""
    
    def supports_model_type(self, model_type: str) -> bool:
        return model_type in self.config.supported_model_types
    
    def get_litellm_model_name(self, model_id: str) -> str:
        """Convert model ID to LiteLLM format for OpenRouter"""
        # OpenRouter uses the format: openrouter/model_id
        return f"openrouter/{model_id}"
    
    def setup_litellm_config(self) -> Dict[str, Any]:
        """Setup LiteLLM configuration for OpenRouter"""
        return {
            "api_base": self.config.base_url,
            "api_key": self.api_key,
            "headers": {
                **self.config.headers,
                "Authorization": f"Bearer {self.api_key}"
            },
            "timeout": self.config.timeout,
            "max_retries": self.config.retry_config.get('max_retries', 3),
        }


class DeepInfraProvider(BaseProvider):
    """DeepInfra provider implementation"""
    
    def supports_model_type(self, model_type: str) -> bool:
        return model_type in self.config.supported_model_types
    
    def get_litellm_model_name(self, model_id: str) -> str:
        """Convert model ID to LiteLLM format for DeepInfra"""
        # DeepInfra uses the format: deepinfra/model_id
        return f"deepinfra/{model_id}"
    
    def setup_litellm_config(self) -> Dict[str, Any]:
        """Setup LiteLLM configuration for DeepInfra"""
        return {
            "api_base": self.config.base_url,
            "api_key": self.api_key,
            "headers": {
                **self.config.headers,
                "Authorization": f"Bearer {self.api_key}"
            },
            "timeout": self.config.timeout,
            "max_retries": self.config.retry_config.get('max_retries', 3),
        }


class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation"""
    
    def supports_model_type(self, model_type: str) -> bool:
        return model_type in self.config.supported_model_types
    
    def get_litellm_model_name(self, model_id: str) -> str:
        """Convert model ID to LiteLLM format for OpenAI"""
        # OpenAI models can be used directly
        return model_id
    
    def setup_litellm_config(self) -> Dict[str, Any]:
        """Setup LiteLLM configuration for OpenAI"""
        return {
            "api_base": self.config.base_url,
            "api_key": self.api_key,
            "timeout": self.config.timeout,
            "max_retries": self.config.retry_config.get('max_retries', 3),
        }


class AnthropicProvider(BaseProvider):
    """Anthropic provider implementation"""
    
    def supports_model_type(self, model_type: str) -> bool:
        return model_type in self.config.supported_model_types
    
    def get_litellm_model_name(self, model_id: str) -> str:
        """Convert model ID to LiteLLM format for Anthropic"""
        # Anthropic uses the format: claude-3-sonnet-20240229
        return model_id
    
    def setup_litellm_config(self) -> Dict[str, Any]:
        """Setup LiteLLM configuration for Anthropic"""
        return {
            "api_base": self.config.base_url,
            "api_key": self.api_key,
            "headers": {
                **self.config.headers,
                "x-api-key": self.api_key
            },
            "timeout": self.config.timeout,
            "max_retries": self.config.retry_config.get('max_retries', 3),
        }


class HuggingFaceProvider(BaseProvider):
    """Hugging Face provider implementation"""
    
    def supports_model_type(self, model_type: str) -> bool:
        return model_type in self.config.supported_model_types
    
    def get_litellm_model_name(self, model_id: str) -> str:
        """Convert model ID to LiteLLM format for Hugging Face"""
        # HuggingFace uses the format: huggingface/model_id
        return f"huggingface/{model_id}"
    
    def setup_litellm_config(self) -> Dict[str, Any]:
        """Setup LiteLLM configuration for Hugging Face"""
        return {
            "api_base": self.config.base_url,
            "api_key": self.api_key,
            "headers": {
                **self.config.headers,
                "Authorization": f"Bearer {self.api_key}"
            },
            "timeout": self.config.timeout,
            "max_retries": self.config.retry_config.get('max_retries', 3),
        }


class ProviderManager:
    """Manager for all providers"""
    
    def __init__(self):
        self.providers: Dict[str, BaseProvider] = {}
        self.config_loader = get_config_loader()
        self._provider_classes = {
            'openrouter': OpenRouterProvider,
            'deepinfra': DeepInfraProvider,
            'openai': OpenAIProvider,
            'anthropic': AnthropicProvider,
            'huggingface': HuggingFaceProvider,
        }
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured providers"""
        for provider_name in self.config_loader.list_providers():
            try:
                provider_config = self.config_loader.get_provider_config(provider_name)
                provider_class = self._provider_classes.get(provider_name)
                
                if provider_class:
                    provider = provider_class(provider_config)
                    self.providers[provider_name] = provider
                    logger.info(f"Initialized provider: {provider_name}")
                else:
                    logger.warning(f"No provider class found for: {provider_name}")
                    
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_name}: {e}")
    
    def get_provider(self, provider_name: str) -> Optional[BaseProvider]:
        """Get a provider by name"""
        return self.providers.get(provider_name)
    
    def get_active_providers(self) -> Dict[str, BaseProvider]:
        """Get all active providers"""
        return {
            name: provider for name, provider in self.providers.items()
            if provider.status == ProviderStatus.ACTIVE
        }
    
    def get_provider_for_model_type(self, model_type: str) -> List[str]:
        """Get providers that support a specific model type"""
        return [
            name for name, provider in self.providers.items()
            if provider.supports_model_type(model_type) and provider.status == ProviderStatus.ACTIVE
        ]
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers"""
        status = {}
        for name, provider in self.providers.items():
            status[name] = {
                'status': provider.status.value,
                'metrics': {
                    'total_requests': provider.metrics.total_requests,
                    'successful_requests': provider.metrics.successful_requests,
                    'failed_requests': provider.metrics.failed_requests,
                    'average_response_time': provider.metrics.average_response_time,
                    'error_count': provider.metrics.error_count,
                    'last_error': provider.metrics.last_error,
                },
                'config': {
                    'name': provider.config.name,
                    'description': provider.config.description,
                    'supported_model_types': provider.config.supported_model_types,
                }
            }
        return status
    
    def setup_litellm_for_provider(self, provider_name: str) -> Dict[str, Any]:
        """Setup LiteLLM configuration for a specific provider"""
        provider = self.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not found")
        
        if provider.status != ProviderStatus.ACTIVE:
            raise ValueError(f"Provider '{provider_name}' is not active")
        
        return provider.setup_litellm_config()
    
    def get_litellm_model_name(self, provider_name: str, model_id: str) -> str:
        """Get LiteLLM formatted model name"""
        provider = self.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not found")
        
        return provider.get_litellm_model_name(model_id)
    
    def validate_provider_setup(self, provider_name: str) -> Tuple[bool, Optional[str]]:
        """Validate that a provider is properly set up"""
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
    
    def refresh_provider(self, provider_name: str):
        """Refresh a specific provider"""
        if provider_name in self.providers:
            try:
                provider_config = self.config_loader.get_provider_config(provider_name)
                provider_class = self._provider_classes.get(provider_name)
                
                if provider_class:
                    self.providers[provider_name] = provider_class(provider_config)
                    logger.info(f"Refreshed provider: {provider_name}")
                
            except Exception as e:
                logger.error(f"Failed to refresh provider {provider_name}: {e}")
    
    def refresh_all_providers(self):
        """Refresh all providers"""
        logger.info("Refreshing all providers...")
        self.providers.clear()
        self._initialize_providers()


# Global provider manager instance
_provider_manager = None


def get_provider_manager() -> ProviderManager:
    """Get the global provider manager instance"""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = ProviderManager()
    return _provider_manager


def setup_litellm_logging():
    """Setup LiteLLM logging configuration"""
    # Configure LiteLLM logging
    litellm.set_verbose = False  # Set to True for debugging
    
    # Set custom logging level for LiteLLM
    logging.getLogger("litellm").setLevel(logging.WARNING)
    
    # Setup custom success and failure callbacks if needed
    def log_success(kwargs):
        logger.debug(f"LiteLLM request successful: {kwargs.get('model', 'unknown')}")
    
    def log_failure(kwargs):
        logger.warning(f"LiteLLM request failed: {kwargs.get('model', 'unknown')} - {kwargs.get('exception', 'unknown error')}")
    
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
        print("Provider Status:")
        for name, data in status.items():
            print(f"  {name}: {data['status']} - {data['config']['description']}")
        
        # Test provider validation
        for provider_name in pm.providers.keys():
            is_valid, error = pm.validate_provider_setup(provider_name)
            print(f"Provider {provider_name} validation: {'✓' if is_valid else '✗'} {error or ''}")
        
        # Test model type support
        chat_providers = pm.get_provider_for_model_type('chat')
        embedding_providers = pm.get_provider_for_model_type('embedding')
        print(f"Chat providers: {chat_providers}")
        print(f"Embedding providers: {embedding_providers}")
        
        print("Provider manager test completed successfully!")
        
    except Exception as e:
        print(f"Provider manager test failed: {e}")
        raise