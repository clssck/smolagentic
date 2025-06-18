"""Example usage of the Model Factory system

This script demonstrates how to use the dynamic model loading and management
system with various providers and models.
"""

import asyncio
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import our model factory system
from models import (
    ModelType,
    chat_completion,
    get_embedding,
    get_model_factory,
    switch_chat_model,
)


def setup_environment() -> bool:
    """Setup environment variables for testing (you'll need to set these properly)"""
    # Note: In production, set these in your .env file or environment
    # os.environ['OPENROUTER_API_KEY'] = 'your_openrouter_key_here'
    # os.environ['DEEPINFRA_API_KEY'] = 'your_deepinfra_key_here'

    # Check if required environment variables are set
    required_vars = ["OPENROUTER_API_KEY", "DEEPINFRA_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.warning("Some examples may fail without proper API keys")

    return len(missing_vars) == 0


def example_basic_usage() -> None:
    """Example 1: Basic usage of the model factory"""
    print("\n" + "="*50)
    print("Example 1: Basic Model Factory Usage")
    print("="*50)

    try:
        # Get the model factory instance
        factory = get_model_factory()

        # Check available models
        available_models = factory.get_available_models()
        print("Available models:")
        for model_type, models in available_models.items():
            print(f"  {model_type.title()}: {len(models)} models")
            for model in models[:3]:  # Show first 3
                print(f"    - {model}")
            if len(models) > 3:
                print(f"    ... and {len(models) - 3} more")

        # Check current models
        current_chat = factory.current_chat_model
        current_embedding = factory.current_embedding_model
        print("\nCurrent models:")
        print(f"  Chat: {current_chat}")
        print(f"  Embedding: {current_embedding}")

        # Get model status
        status = factory.get_model_status()
        print(f"\nLoaded models: {len(status)}")
        for key, data in status.items():
            print(f"  {key}: {data['status']} - {data['display_name']}")

    except Exception as e:
        logger.error(f"Basic usage example failed: {e}")


def example_chat_completion() -> None:
    """Example 2: Chat completion with different models"""
    print("\n" + "="*50)
    print("Example 2: Chat Completion")
    print("="*50)

    try:
        factory = get_model_factory()

        # Test with default model
        if factory.current_chat_model:
            print(f"Testing chat with default model: {factory.current_chat_model}")

            messages = [
                {"role": "system", "content": "You are a helpful assistant that gives concise answers."},
                {"role": "user", "content": "What is the capital of France? Answer in one sentence."},
            ]

            response = factory.chat_completion(messages, max_tokens=100)
            print(f"Response: {response.choices[0].message.content}")

            # Show updated metrics
            status = factory.get_model_status()
            model_key = f"{factory.current_chat_model}_chat"
            if model_key in status:
                metrics = status[model_key]["metrics"]
                print(f"Model metrics: {metrics['total_requests']} requests, "
                      f"{metrics['total_tokens_used']} tokens, "
                      f"${metrics['total_cost']:.4f} cost")

        # Test switching models
        available_chat = factory.get_available_models()["chat"]
        if len(available_chat) > 1:
            # Switch to a different model
            new_model = None
            for model in available_chat:
                if model != factory.current_chat_model:
                    new_model = model
                    break

            if new_model:
                print(f"\nSwitching to model: {new_model}")
                factory.switch_model(new_model, ModelType.CHAT)

                # Test with new model
                response = factory.chat_completion(messages, max_tokens=100)
                print(f"Response from {new_model}: {response.choices[0].message.content}")

    except Exception as e:
        logger.error(f"Chat completion example failed: {e}")


def example_embeddings() -> None:
    """Example 3: Embedding generation"""
    print("\n" + "="*50)
    print("Example 3: Embedding Generation")
    print("="*50)

    try:
        factory = get_model_factory()

        if factory.current_embedding_model:
            print(f"Testing embeddings with: {factory.current_embedding_model}")

            # Single text embedding
            text = "This is a sample text for embedding generation."
            response = factory.get_embedding(text)

            embedding_vector = response.data[0].embedding
            print(f"Generated embedding with {len(embedding_vector)} dimensions")
            print(f"First 5 values: {embedding_vector[:5]}")

            # Batch embedding
            texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is transforming various industries.",
                "Natural language processing enables computers to understand human language.",
            ]

            batch_response = factory.get_embedding(texts)
            print(f"\nBatch embedding: {len(batch_response.data)} embeddings generated")

            # Show similarity between first two embeddings (simple dot product)
            emb1 = batch_response.data[0].embedding
            emb2 = batch_response.data[1].embedding

            # Simple dot product similarity
            similarity = sum(a * b for a, b in zip(emb1, emb2, strict=False))
            print(f"Similarity between first two texts: {similarity:.4f}")

            # Show metrics
            status = factory.get_model_status()
            model_key = f"{factory.current_embedding_model}_embedding"
            if model_key in status:
                metrics = status[model_key]["metrics"]
                print(f"Embedding model metrics: {metrics['total_requests']} requests, "
                      f"${metrics['total_cost']:.4f} cost")

    except Exception as e:
        logger.error(f"Embeddings example failed: {e}")


async def example_async_operations() -> None:
    """Example 4: Async operations"""
    print("\n" + "="*50)
    print("Example 4: Async Operations")
    print("="*50)

    try:
        factory = get_model_factory()

        # Async chat completion
        if factory.current_chat_model:
            print(f"Testing async chat with: {factory.current_chat_model}")

            messages = [
                {"role": "user", "content": "Count from 1 to 5, each number on a new line."},
            ]

            response = await factory.achat_completion(messages, max_tokens=50)
            print(f"Async chat response: {response.choices[0].message.content}")

        # Async embedding
        if factory.current_embedding_model:
            print(f"\nTesting async embedding with: {factory.current_embedding_model}")

            text = "Async embedding generation test."
            response = await factory.aget_embedding(text)

            embedding_vector = response.data[0].embedding
            print(f"Async embedding generated: {len(embedding_vector)} dimensions")

        # Concurrent operations
        print("\nTesting concurrent operations...")

        async def chat_task():
            messages = [{"role": "user", "content": "What is 10 + 15?"}]
            return await factory.achat_completion(messages, max_tokens=30)

        async def embedding_task():
            return await factory.aget_embedding("Concurrent embedding test")

        # Run both operations concurrently
        chat_result, embedding_result = await asyncio.gather(
            chat_task(),
            embedding_task(),
            return_exceptions=True,
        )

        if not isinstance(chat_result, Exception):
            print(f"Concurrent chat result: {chat_result.choices[0].message.content}")

        if not isinstance(embedding_result, Exception):
            print(f"Concurrent embedding result: {len(embedding_result.data[0].embedding)} dimensions")

    except Exception as e:
        logger.error(f"Async operations example failed: {e}")


def example_model_management() -> None:
    """Example 5: Model management and monitoring"""
    print("\n" + "="*50)
    print("Example 5: Model Management and Monitoring")
    print("="*50)

    try:
        factory = get_model_factory()

        # Show detailed model status
        status = factory.get_model_status()
        print("Detailed model status:")
        for key, data in status.items():
            print(f"\n{key}:")
            print(f"  Status: {data['status']}")
            print(f"  Provider: {data['provider']}")
            print(f"  Display Name: {data['display_name']}")
            print(f"  Is Current: {data['is_current']}")
            print(f"  Last Used: {data['last_used']}")

            metrics = data["metrics"]
            print("  Metrics:")
            print(f"    Total Requests: {metrics['total_requests']}")
            print(f"    Success Rate: {metrics['successful_requests']}/{metrics['total_requests']}")
            print(f"    Tokens Used: {metrics['total_tokens_used']}")
            print(f"    Total Cost: ${metrics['total_cost']:.4f}")
            print(f"    Avg Response Time: {metrics['average_response_time']:.3f}s")
            if metrics["last_error"]:
                print(f"    Last Error: {metrics['last_error']}")

        # Show provider status
        from models import get_provider_manager
        provider_manager = get_provider_manager()
        provider_status = provider_manager.get_provider_status()

        print("\nProvider status:")
        for name, data in provider_status.items():
            print(f"\n{name}:")
            print(f"  Status: {data['status']}")
            print(f"  Description: {data['config']['description']}")
            print(f"  Supported Types: {data['config']['supported_model_types']}")

            metrics = data["metrics"]
            if metrics["total_requests"] > 0:
                print(f"  Requests: {metrics['total_requests']} total, "
                      f"{metrics['successful_requests']} successful")
                print(f"  Avg Response Time: {metrics['average_response_time']:.3f}s")

        # Test model switching
        available_chat = factory.get_available_models()["chat"]
        if len(available_chat) > 1:
            original_model = factory.current_chat_model

            # Switch to different models and test each
            print("\nTesting model switching...")
            for model_name in available_chat[:2]:  # Test first 2 models
                try:
                    print(f"Switching to: {model_name}")
                    factory.switch_model(model_name, ModelType.CHAT)

                    # Quick test
                    response = factory.chat_completion(
                        [{"role": "user", "content": "Say 'Hello from " + model_name + "'"}],
                        max_tokens=20,
                    )
                    print(f"  Response: {response.choices[0].message.content}")

                except Exception as e:
                    print(f"  Failed to test {model_name}: {e}")

            # Switch back to original
            if original_model:
                factory.switch_model(original_model, ModelType.CHAT)
                print(f"Switched back to: {original_model}")

    except Exception as e:
        logger.error(f"Model management example failed: {e}")


def example_error_handling() -> None:
    """Example 6: Error handling and recovery"""
    print("\n" + "="*50)
    print("Example 6: Error Handling")
    print("="*50)

    try:
        factory = get_model_factory()

        # Test with invalid model
        print("Testing with invalid model name...")
        try:
            factory.load_model("nonexistent-model", ModelType.CHAT)
        except Exception as e:
            print(f"Expected error caught: {e}")

        # Test with invalid parameters
        print("\nTesting with invalid parameters...")
        if factory.current_chat_model:
            try:
                # This might cause an error depending on the model
                response = factory.chat_completion(
                    [{"role": "user", "content": "Test"}],
                    max_tokens=-1,  # Invalid parameter
                )
            except Exception as e:
                print(f"Parameter error caught: {e}")

        # Test rate limiting simulation
        print("\nTesting rapid requests...")
        if factory.current_chat_model:
            successful = 0
            failed = 0

            for i in range(3):  # Small number for testing
                try:
                    response = factory.chat_completion(
                        [{"role": "user", "content": f"Quick test {i+1}"}],
                        max_tokens=10,
                    )
                    successful += 1
                    print(f"  Request {i+1}: Success")
                except Exception as e:
                    failed += 1
                    print(f"  Request {i+1}: Failed - {e}")

            print(f"Results: {successful} successful, {failed} failed")

        # Show any error states in models
        status = factory.get_model_status()
        error_models = [key for key, data in status.items() if data["status"] == "error"]
        if error_models:
            print(f"\nModels in error state: {error_models}")
            for model_key in error_models:
                error = status[model_key]["metrics"]["last_error"]
                print(f"  {model_key}: {error}")

    except Exception as e:
        logger.error(f"Error handling example failed: {e}")


def example_convenience_functions() -> None:
    """Example 7: Using convenience functions"""
    print("\n" + "="*50)
    print("Example 7: Convenience Functions")
    print("="*50)

    try:
        # Using module-level convenience functions
        print("Using convenience functions...")

        # Chat with default model
        response = chat_completion([
            {"role": "user", "content": "What is 2 + 2? Answer briefly."},
        ], max_tokens=20)
        print(f"Chat response: {response.choices[0].message.content}")

        # Embedding with default model
        embedding_response = get_embedding("Test embedding with convenience function")
        print(f"Embedding dimensions: {len(embedding_response.data[0].embedding)}")

        # Model switching convenience functions
        factory = get_model_factory()
        available_chat = factory.get_available_models()["chat"]

        if len(available_chat) > 1:
            # Find a different model to switch to
            current = factory.current_chat_model
            new_model = None
            for model in available_chat:
                if model != current:
                    new_model = model
                    break

            if new_model:
                print(f"\nSwitching from {current} to {new_model}")
                switch_chat_model(new_model)

                # Test new model
                response = chat_completion([
                    {"role": "user", "content": "Hello from the new model!"},
                ], max_tokens=20)
                print(f"New model response: {response.choices[0].message.content}")

                # Switch back
                switch_chat_model(current)
                print(f"Switched back to {current}")

    except Exception as e:
        logger.error(f"Convenience functions example failed: {e}")


async def main() -> None:
    """Main function to run all examples"""
    print("Model Factory System Examples")
    print("=" * 60)

    # Setup environment
    env_ready = setup_environment()
    if not env_ready:
        print("\nWARNING: Some API keys are missing. Examples may fail.")
        print("Please set OPENROUTER_API_KEY and DEEPINFRA_API_KEY environment variables.")

    # Run examples
    try:
        example_basic_usage()
        example_chat_completion()
        example_embeddings()
        await example_async_operations()
        example_model_management()
        example_error_handling()
        example_convenience_functions()

        print("\n" + "="*60)
        print("All examples completed!")
        print("="*60)

    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        logger.error(f"Examples failed: {e}")
        raise


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
