#!/usr/bin/env python3
"""
Example usage of the configuration system

This script demonstrates how to use the configuration loader to:
1. Load model and provider configurations
2. Switch between different models
3. Access API keys and settings
4. Validate environment setup
"""

import os
import sys
from pathlib import Path

from utils.config_loader import (
    get_config_loader, 
    ModelType, 
    load_model_config, 
    load_provider_config,
    get_api_key
)


def main():
    """Demonstrate configuration system usage"""
    
    print("=== RAG Application Configuration System Demo ===\n")
    
    try:
        # Initialize the configuration loader
        config = get_config_loader()
        
        # 1. Show available models
        print("1. Available Models:")
        print("-" * 40)
        
        chat_models = config.list_models_by_type(ModelType.CHAT)
        embedding_models = config.list_models_by_type(ModelType.EMBEDDING)
        
        print(f"Chat Models ({len(chat_models)}):")
        for model in chat_models:
            model_config = config.get_model_config(model, ModelType.CHAT)
            print(f"  • {model_config.display_name} ({model}) - {model_config.provider}")
        
        print(f"\nEmbedding Models ({len(embedding_models)}):")
        for model in embedding_models:
            model_config = config.get_model_config(model, ModelType.EMBEDDING)
            print(f"  • {model_config.display_name} ({model}) - {model_config.provider}")
        
        # 2. Show default model configuration
        print("\n2. Default Model Configuration:")
        print("-" * 40)
        
        defaults = config.get_default_models()
        print(f"Default Chat Model: {defaults['chat_model']}")
        print(f"Default Embedding Model: {defaults['embedding_model']}")
        
        # Load default chat model details
        default_chat = config.get_model_config(defaults['chat_model'], ModelType.CHAT)
        print(f"\nDefault Chat Model Details:")
        print(f"  Display Name: {default_chat.display_name}")
        print(f"  Provider: {default_chat.provider}")
        print(f"  Model ID: {default_chat.model_id}")
        print(f"  Context Window: {default_chat.context_window:,} tokens")
        print(f"  Temperature: {default_chat.parameters.get('temperature')}")
        print(f"  Max Tokens: {default_chat.parameters.get('max_tokens')}")
        
        # 3. Show provider configurations
        print("\n3. Provider Configurations:")
        print("-" * 40)
        
        providers = config.list_providers()
        for provider_name in providers:
            provider = config.get_provider_config(provider_name)
            print(f"{provider.name} ({provider_name}):")
            print(f"  Base URL: {provider.base_url}")
            print(f"  API Key Env: {provider.api_key_env}")
            print(f"  Supported Types: {', '.join(provider.supported_model_types)}")
            print(f"  Rate Limit: {provider.rate_limits.get('requests_per_minute', 'N/A')} req/min")
            print()
        
        # 4. Show models by category
        print("4. Models by Category:")
        print("-" * 40)
        
        categories = config.models_config.get('categories', {})
        for category, models in categories.items():
            print(f"{category.replace('_', ' ').title()}: {', '.join(models)}")
        
        # 5. Environment validation
        print("\n5. Environment Validation:")
        print("-" * 40)
        
        env_status = config.validate_environment()
        for provider, has_key in env_status.items():
            status = "✓ Available" if has_key else "✗ Missing"
            provider_config = config.get_provider_config(provider)
            print(f"{provider_config.name}: {status} ({provider_config.api_key_env})")
        
        # 6. Demonstrate model switching
        print("\n6. Model Switching Example:")
        print("-" * 40)
        
        # Switch between different chat models
        test_models = ["qwen3-7b-instruct", "qwen3-14b-instruct", "qwen3-72b-instruct"]
        
        for model_name in test_models:
            try:
                model = config.get_model_config(model_name, ModelType.CHAT)
                cost = model.cost_per_1k_tokens
                cost_str = f"${cost['input']:.4f}/${cost['output']:.4f} per 1K tokens" if cost else "N/A"
                
                print(f"Model: {model.display_name}")
                print(f"  Provider: {model.provider}")
                print(f"  Context: {model.context_window:,} tokens")
                print(f"  Cost: {cost_str}")
                print(f"  Capabilities: {', '.join(model.capabilities)}")
                print()
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
        
        # 7. Show how to get API keys (without revealing actual keys)
        print("7. API Key Access Example:")
        print("-" * 40)
        
        for provider_name in ["openrouter", "deepinfra"]:
            try:
                provider_config = config.get_provider_config(provider_name)
                env_var = provider_config.api_key_env
                has_key = bool(os.getenv(env_var))
                
                print(f"{provider_config.name}:")
                print(f"  Environment Variable: {env_var}")
                print(f"  Key Available: {'Yes' if has_key else 'No'}")
                
                if has_key:
                    # Don't actually print the key, just show it's accessible
                    key = get_api_key(provider_name)
                    print(f"  Key Length: {len(key)} characters")
                    print(f"  Key Preview: {key[:8]}...")
                else:
                    print(f"  Status: Set {env_var} environment variable")
                print()
                
            except Exception as e:
                print(f"Error accessing {provider_name} API key: {e}")
        
        print("=== Configuration Demo Completed Successfully! ===")
        
    except Exception as e:
        print(f"Configuration demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()