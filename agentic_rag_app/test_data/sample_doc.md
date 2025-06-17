# Sample Document for RAG Testing

## Introduction
This is a sample document to test the agentic RAG application. It contains various types of information that can be used to demonstrate the system's capabilities.

## Technical Specifications
- **Framework**: LlamaIndex with custom agents
- **Vector Store**: Qdrant with hybrid chunking
- **Models**: Qwen3 family models via OpenRouter and DeepInfra
- **Interface**: Gradio web application
- **Chunking Strategy**: Semantic + Hierarchical + Sentence-based

## Features
1. **Dynamic Model Switching**: Users can switch between different chat and embedding models on the fly
2. **Hybrid Chunking**: Combines semantic understanding with hierarchical document structure
3. **Agentic Workflow**: Uses ReAct agents for intelligent query planning and tool use
4. **Multi-Provider Support**: Supports OpenRouter, DeepInfra, OpenAI, and Anthropic

## Configuration
The application uses YAML configuration files to define:
- Available models and their parameters
- Provider settings and API configurations
- Default system behaviors

## Usage Instructions
1. Set up your API keys in the `.env` file
2. Start your Qdrant server
3. Run `python main.py` to start the application
4. Upload documents to the `test_data/` folder
5. Use the "Ingest Documents" button to process them
6. Start chatting with your documents!

## Advanced Features
- Real-time model switching
- Conversation memory management
- Advanced retrieval with metadata extraction
- System status monitoring
- Knowledge base search functionality