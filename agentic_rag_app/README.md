# Agentic RAG System

A modular RAG (Retrieval-Augmented Generation) system with intelligent agent routing and swappable components.

## 🚀 Quick Start

### Web UI (Recommended)
```bash
# Basic launch
python main.py

# Custom port
python main.py --port 8080

# Enable sharing
python main.py --share
```

### CLI Mode
```bash
python main.py --cli
```

### Direct UI Launch
```bash
# Launch web UI directly
python launch_ui.py

# With custom settings
python launch_ui.py --port 8080 --share --upload-folder ./uploads
```

## ✨ Features

- **🤖 Intelligent Agent Routing** - Automatically selects the best agent for each query
- **🔧 Modular Architecture** - Swappable models, embedders, and retrievers
- **📁 File Upload Support** - Upload and process documents directly in the UI
- **💬 Conversation Memory** - Maintains context across interactions
- **🔍 Knowledge Base Search** - RAG capabilities with vector search
- **🧠 Research Capabilities** - Advanced research and analysis tools

## 🏗️ Architecture

- **Models**: OpenRouter, DeepInfra, OpenAI support
- **Embedders**: Multiple embedding model options
- **Retrievers**: Qdrant vector database integration
- **Agents**: RAG, Research, Simple QA agents
- **UI**: Native smolagents interface with Gradio

## 🔧 Configuration

Set up your environment variables in `.env`:
```bash
OPENROUTER_API_KEY=your_key_here
DEEPINFRA_API_KEY=your_key_here
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
```

## 📊 Components

The system uses a modular configuration in `config.json` with swappable:
- **Models** (OpenRouter, DeepInfra, OpenAI)
- **Embedders** (Various embedding models)
- **Retrievers** (Qdrant with enhanced processing)
- **Agents** (RAG, Research, Simple QA)

## 🎯 Usage Examples

- **Knowledge Questions**: "What are the key features of our RAG system?"
- **Research Tasks**: "Analyze the latest trends in AI"
- **Document Processing**: Upload files and ask questions about them
- **General QA**: "What is the capital of France?"

## 🧪 Testing

```bash
# Run system tests
python tests/test_working_system.py

# Full end-to-end tests
python tests/test_full_e2e_agentic_rag.py
```