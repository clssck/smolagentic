# 🤖 Agentic RAG Chat Application

A sophisticated RAG (Retrieval-Augmented Generation) application with agentic capabilities, built using LlamaIndex, Qdrant, LiteLLM, and Gradio.

## ✨ Features

- **🔀 Dynamic Model Switching**: Switch between different chat and embedding models on-the-fly
- **🧠 Agentic Workflow**: ReAct agents for intelligent query planning and tool usage
- **📚 Hybrid Chunking**: Advanced document processing with semantic, hierarchical, and sentence-based chunking
- **🔌 Multi-Provider Support**: OpenRouter, DeepInfra, OpenAI, Anthropic, and more
- **💬 Interactive Chat Interface**: Modern Gradio web interface with real-time conversation
- **📊 System Monitoring**: Real-time status monitoring and configuration management

## 🚀 Quick Start

1. **Clone and Setup**
   ```bash
   cd agentic_rag_app
   pip install -r requirements.txt
   ```

2. **Configure API Keys**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start Qdrant** (if using local instance)
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

4. **Add Documents**
   ```bash
   # Place your documents in test_data/ folder
   # Supports: .txt, .md, .pdf, .docx, .json
   ```

5. **Launch Application**
   ```bash
   python main.py
   ```

6. **Access Interface**
   - Open http://localhost:7860 in your browser
   - Use the "Ingest Documents" button to process your documents
   - Start chatting!

## 🛠️ Configuration

### Environment Variables (.env)
```env
OPENROUTER_API_KEY=your_key_here
DEEPINFRA_API_KEY=your_key_here
QDRANT_URL=http://database-qdrant.itsfl7.easypanel.host
QDRANT_API_KEY=your_qdrant_api_key_here
DEFAULT_CHAT_MODEL=qwen3-14b-instruct
DEFAULT_EMBEDDING_MODEL=qwen3-embed
```

### Model Configuration (config/models.yaml)
- Pre-configured with Qwen3, GPT, and Claude models
- Easy to add new models and providers
- Configurable parameters and cost tracking

## 📋 Usage

### Chat Interface
- Natural conversation with your documents
- Real-time model switching
- Conversation history management
- System status monitoring

### Document Management
- Automatic ingestion from `test_data/` folder
- Advanced chunking with metadata extraction
- Knowledge base search functionality
- Support for multiple file formats

### Model Switching
- Chat models: Switch between different LLMs
- Embedding models: Change embedding providers
- Real-time configuration updates

## 🏗️ Architecture

```
agentic_rag_app/
├── config/                 # Configuration files
│   ├── models.yaml        # Model definitions
│   └── providers.yaml     # Provider settings
├── models/                # Model factory and management
├── vector_store/          # Qdrant integration
├── agents/                # RAG agent implementation
├── ui/                    # Gradio interface
├── utils/                 # Configuration utilities
├── test_data/             # Document storage
└── main.py               # Application entry point
```

## 🔧 Advanced Usage

### Command Line Options
```bash
python main.py --help
python main.py --qdrant-url http://your-qdrant:6333
python main.py --host 0.0.0.0 --port 8080
python main.py --share  # Create public link
python main.py --debug  # Enable debug logging
```

### Adding New Models
1. Edit `config/models.yaml` to add model definitions
2. Update `config/providers.yaml` for new providers
3. Restart the application

### Custom Document Processing
- Supports hierarchical, semantic, and sentence-based chunking
- Metadata extractors: titles, summaries, keywords, Q&A
- Configurable chunk sizes and overlap

## 🧪 Example Usage

1. **Ingest Documents**: Place PDFs, text files, or markdown in `test_data/`
2. **Process**: Click "Ingest Documents" in the web interface
3. **Chat**: Ask questions like:
   - "What are the main topics in the documents?"
   - "Summarize the key findings"
   - "How does the system handle model switching?"

## 🤝 Contributing

The application is designed to be modular and extensible:
- Add new chunking strategies in `vector_store/`
- Create custom agents in `agents/`
- Extend the UI in `ui/gradio_app.py`
- Add new model providers in `models/`

## 📝 License

This project is open source and available under the MIT License.