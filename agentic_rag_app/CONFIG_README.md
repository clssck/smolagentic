# Configuration System Documentation

This document explains how to use the flexible configuration system for managing models and providers in the RAG application.

## Overview

The configuration system consists of three main components:

1. **`config/models.yaml`** - Defines available models with their parameters and capabilities
2. **`config/providers.yaml`** - Defines provider configurations, API endpoints, and authentication
3. **`utils/config_loader.py`** - Python utility for loading and managing configurations

## Quick Start

### 1. Set up Environment Variables

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required environment variables:
- `OPENROUTER_API_KEY` - For Qwen3 and other models via OpenRouter
- `DEEPINFRA_API_KEY` - For Qwen3 embedding models

### 2. Basic Usage

```python
from utils.config_loader import get_config_loader, ModelType

# Initialize configuration
config = get_config_loader()

# Load a chat model
chat_model = config.get_model_config("qwen3-14b-instruct", ModelType.CHAT)
print(f"Using {chat_model.display_name} with {chat_model.context_window} context window")

# Load an embedding model
embed_model = config.get_model_config("qwen3-embed", ModelType.EMBEDDING)
print(f"Using {embed_model.display_name} for embeddings")

# Get provider configuration
provider = config.get_provider_config("openrouter")
api_key = config.get_api_key("openrouter")
```

### 3. Run the Demo

```bash
python example_config_usage.py
```

## Configuration Files

### Models Configuration (`config/models.yaml`)

Defines models with their providers, parameters, and capabilities:

```yaml
chat_models:
  qwen3-14b-instruct:
    provider: openrouter
    model_id: "qwen/qwen-2.5-14b-instruct"
    display_name: "Qwen 3 14B Instruct"
    description: "Balanced chat model with good performance"
    parameters:
      temperature: 0.7
      max_tokens: 4000
    capabilities:
      - chat
      - reasoning
    context_window: 32768
```

### Providers Configuration (`config/providers.yaml`)

Defines provider settings, authentication, and rate limits:

```yaml
providers:
  openrouter:
    name: "OpenRouter"
    base_url: "https://openrouter.ai/api/v1"
    api_key_env: "OPENROUTER_API_KEY"
    authentication:
      type: "bearer_token"
    rate_limits:
      requests_per_minute: 60
```

## Available Models

### Chat Models
- **Qwen3 Models** (via OpenRouter): 7B, 14B, 72B variants
- **GPT Models** (via OpenRouter): GPT-4o, GPT-3.5 Turbo
- **Claude Models** (via OpenRouter): Claude 3 Sonnet

### Embedding Models
- **Qwen3 Embed** (via DeepInfra): High-quality embedding model
- **BGE Large EN** (via DeepInfra): Alternative embedding model
- **OpenAI Ada 002** (via OpenAI): OpenAI's embedding model

### Model Categories
- `fast_chat`: Quick response models (Qwen3-7B, GPT-3.5)
- `balanced_chat`: Good performance/cost balance (Qwen3-14B, Claude-3)
- `powerful_chat`: Maximum capability models (Qwen3-72B, GPT-4o)
- `embeddings`: All embedding models

## Common Use Cases

### 1. Switch Between Models

```python
# Get models by category
fast_models = config.get_models_by_category("fast_chat")
powerful_models = config.get_models_by_category("powerful_chat")

# Switch based on use case
model_name = "qwen3-7b-instruct" if need_speed else "qwen3-72b-instruct"
model = config.get_model_config(model_name, ModelType.CHAT)
```

### 2. List Available Models

```python
# List all chat models
chat_models = config.list_models_by_type(ModelType.CHAT)

# List models by provider
openrouter_models = config.list_models_by_provider("openrouter")
```

### 3. Validate Environment

```python
# Check which API keys are available
env_status = config.validate_environment()
for provider, available in env_status.items():
    if not available:
        print(f"Missing API key for {provider}")
```

## Adding New Models

### 1. Add to `config/models.yaml`

```yaml
chat_models:
  your-new-model:
    provider: your_provider
    model_id: "provider/model-name"
    display_name: "Your New Model"
    description: "Description of the model"
    parameters:
      temperature: 0.7
      max_tokens: 4000
    capabilities:
      - chat
    context_window: 8192
```

### 2. Add Provider if Needed

If using a new provider, add to `config/providers.yaml`:

```yaml
providers:
  your_provider:
    name: "Your Provider"
    base_url: "https://api.yourprovider.com/v1"
    api_key_env: "YOUR_PROVIDER_API_KEY"
    authentication:
      type: "bearer_token"
    supported_model_types:
      - chat
```

### 3. Update Environment Variables

Add the new API key to your `.env` file:

```bash
YOUR_PROVIDER_API_KEY=your_api_key_here
```

## Configuration API Reference

### ConfigLoader Class

Main class for managing configurations:

- `get_model_config(model_name, model_type)` - Load model configuration
- `get_provider_config(provider_name)` - Load provider configuration
- `get_api_key(provider_name)` - Get API key from environment
- `list_models_by_type(model_type)` - List models by type
- `list_models_by_provider(provider_name)` - List models by provider
- `get_models_by_category(category)` - Get models by category
- `validate_environment()` - Check API key availability

### Data Classes

- `ModelConfig` - Model configuration with parameters and capabilities
- `ProviderConfig` - Provider configuration with authentication and limits

### Enums

- `ModelType.CHAT` - Chat/completion models
- `ModelType.EMBEDDING` - Embedding models

## Error Handling

The configuration system includes comprehensive error handling:

- **Missing Files**: Clear error messages for missing config files
- **Invalid Configuration**: Validation of required fields and structure
- **Missing API Keys**: Environment variable validation
- **Unknown Models/Providers**: Helpful error messages for typos

## Best Practices

1. **Use Default Models**: Start with the configured defaults for most use cases
2. **Category Selection**: Use model categories to switch between performance tiers
3. **Environment Validation**: Always validate environment before using models
4. **Cost Awareness**: Check model costs before switching to expensive models
5. **Provider Limits**: Respect rate limits defined in provider configurations

## Troubleshooting

### Common Issues

1. **"Model not found"**: Check spelling and ensure model exists in `models.yaml`
2. **"Provider not found"**: Verify provider name and configuration
3. **"API key not found"**: Check environment variable name and value
4. **"Missing required field"**: Validate YAML syntax and required fields

### Debug Mode

Enable debug logging to troubleshoot configuration issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show detailed information about configuration loading and validation.