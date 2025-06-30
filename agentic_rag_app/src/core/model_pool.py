"""
Shared Model Pool for Agent System

Manages model instances to avoid re-initialization and enable sharing across tools and agents.
"""

import time
from typing import Any, Dict, Optional
from smolagents import LiteLLMModel


class ModelPool:
    """
    Centralized model pool that manages model instances for sharing across agents and tools
    
    Features:
    - Model instance caching and reuse
    - Configuration-based model creation
    - Model usage statistics
    - Memory-efficient model sharing
    """
    
    def __init__(self):
        self._models: Dict[str, LiteLLMModel] = {}
        self._model_configs: Dict[str, Dict[str, Any]] = {}
        self._usage_stats: Dict[str, Dict[str, Any]] = {}
        self._created_at = time.time()
    
    def get_model(self, model_id: str, **config) -> LiteLLMModel:
        """
        Get or create a model instance
        
        Args:
            model_id: Unique identifier for the model
            **config: Model configuration (temperature, max_tokens, etc.)
            
        Returns:
            LiteLLMModel instance
        """
        # Create cache key from model_id and config
        cache_key = self._create_cache_key(model_id, config)
        
        # Return existing model if available
        if cache_key in self._models:
            self._update_usage_stats(cache_key, "reused")
            return self._models[cache_key]
        
        # Create new model
        try:
            model = LiteLLMModel(
                model_id=model_id,
                **config
            )
            
            # Cache the model
            self._models[cache_key] = model
            self._model_configs[cache_key] = {
                "model_id": model_id,
                **config
            }
            
            # Initialize usage stats
            self._usage_stats[cache_key] = {
                "created_at": time.time(),
                "reuse_count": 0,
                "last_used": time.time(),
                "model_id": model_id,
                "config": config
            }
            
            self._update_usage_stats(cache_key, "created")
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to create model {model_id}: {e}")
    
    def get_model_by_name(self, name: str, model_config: Dict[str, Any]) -> LiteLLMModel:
        """
        Get model by configuration name (helper method)
        
        Args:
            name: Human-readable name for the model
            model_config: Full model configuration dictionary
            
        Returns:
            LiteLLMModel instance
        """
        model_id = model_config.get("name") or model_config.get("model_id")
        if not model_id:
            raise ValueError(f"No model_id found in config for {name}")
        
        # Extract model parameters
        config = {k: v for k, v in model_config.items() 
                 if k not in ["name", "model_id", "reasoning", "description"]}
        
        return self.get_model(model_id, **config)
    
    def preload_models(self, model_configs: Dict[str, Dict[str, Any]]):
        """
        Preload models from configuration
        
        Args:
            model_configs: Dictionary of model configurations
        """
        for name, config in model_configs.items():
            try:
                self.get_model_by_name(name, config)
                print(f"✅ Preloaded model: {name}")
            except Exception as e:
                print(f"❌ Failed to preload model {name}: {e}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get model pool statistics"""
        return {
            "total_models": len(self._models),
            "pool_age_seconds": time.time() - self._created_at,
            "models": {
                cache_key: {
                    "model_id": stats["model_id"],
                    "reuse_count": stats["reuse_count"],
                    "age_seconds": time.time() - stats["created_at"],
                    "last_used_seconds_ago": time.time() - stats["last_used"]
                }
                for cache_key, stats in self._usage_stats.items()
            }
        }
    
    def clear_unused_models(self, max_age_seconds: int = 3600):
        """
        Clear models that haven't been used recently
        
        Args:
            max_age_seconds: Maximum age in seconds before model is considered unused
        """
        current_time = time.time()
        unused_keys = []
        
        for cache_key, stats in self._usage_stats.items():
            if current_time - stats["last_used"] > max_age_seconds:
                unused_keys.append(cache_key)
        
        for key in unused_keys:
            del self._models[key]
            del self._model_configs[key]
            del self._usage_stats[key]
            print(f"🧹 Cleared unused model: {key}")
        
        return len(unused_keys)
    
    def _create_cache_key(self, model_id: str, config: Dict[str, Any]) -> str:
        """Create cache key from model_id and config"""
        # Sort config items for consistent key generation
        config_str = "_".join(f"{k}={v}" for k, v in sorted(config.items()))
        return f"{model_id}_{config_str}" if config_str else model_id
    
    def _update_usage_stats(self, cache_key: str, action: str):
        """Update usage statistics for a model"""
        if cache_key in self._usage_stats:
            stats = self._usage_stats[cache_key]
            stats["last_used"] = time.time()
            
            if action == "reused":
                stats["reuse_count"] += 1


class SharedModelManager:
    """
    Manager class that provides easy access to shared models across the system
    """
    
    _instance: Optional['SharedModelManager'] = None
    _pool: Optional[ModelPool] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._pool = ModelPool()
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'SharedModelManager':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_model(self, model_id: str, **config) -> LiteLLMModel:
        """Get model from shared pool"""
        return self._pool.get_model(model_id, **config)
    
    def get_model_by_name(self, name: str, model_config: Dict[str, Any]) -> LiteLLMModel:
        """Get model by configuration name"""
        return self._pool.get_model_by_name(name, model_config)
    
    def preload_models(self, model_configs: Dict[str, Dict[str, Any]]):
        """Preload models from configuration"""
        self._pool.preload_models(model_configs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return self._pool.get_pool_stats()
    
    def cleanup(self, max_age_seconds: int = 3600) -> int:
        """Clean up unused models"""
        return self._pool.clear_unused_models(max_age_seconds)


# Global instance for easy access
shared_models = SharedModelManager.get_instance()


def get_shared_model(model_id: str, **config) -> LiteLLMModel:
    """
    Convenience function to get a shared model
    
    Args:
        model_id: Model identifier
        **config: Model configuration
        
    Returns:
        Shared LiteLLMModel instance
    """
    return shared_models.get_model(model_id, **config)


def get_model_by_config(name: str, model_config: Dict[str, Any]) -> LiteLLMModel:
    """
    Convenience function to get model by configuration
    
    Args:
        name: Configuration name
        model_config: Model configuration dictionary
        
    Returns:
        Shared LiteLLMModel instance
    """
    return shared_models.get_model_by_name(name, model_config)