# Chat Models Guide: Mistral & Qwen via OpenRouter

This guide helps you choose and configure the best Mistral and Qwen models for chat applications through OpenRouter.

## 🚀 Quick Start

1. **Set your OpenRouter API key:**
   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   ```

2. **Test the models:**
   ```bash
   python test_chat_models.py
   ```

3. **Launch with your preferred model:**
   ```bash
   python main.py
   ```

## 🤖 Available Models

### Mistral Models

| Model | Size | Context | Best For | Speed | Quality |
|-------|------|---------|----------|-------|---------|
| `mistral_large` | Large | 128K | Complex reasoning, analysis | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| `mistral_small` | Small | 128K | General chat, fast responses | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| `mistral_nemo` | Medium | 128K | Balanced performance | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### Qwen Models

| Model | Size | Context | Best For | Speed | Quality |
|-------|------|---------|----------|-------|---------|
| `qwen25_72b` | 72B | 131K | Top performance, complex tasks | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| `qwen25_32b` | 32B | 131K | Balanced performance/speed | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| `qwen25_14b` | 14B | 131K | Fast general chat | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| `qwen25_7b` | 7B | 131K | Lightweight, very fast | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| `qwen25_coder_32b` | 32B | 131K | Coding, technical discussions | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🎯 Model Selection Guide

### For Different Use Cases:

**💬 General Chat & Conversation:**
- **Recommended:** `qwen25_32b` or `mistral_small`
- **Why:** Great balance of speed and quality for everyday chat

**🧠 Complex Reasoning & Analysis:**
- **Recommended:** `qwen25_72b` or `mistral_large`
- **Why:** Best performance for complex questions and detailed analysis

**⚡ Fast Responses (Speed Priority):**
- **Recommended:** `qwen25_7b` or `qwen25_14b`
- **Why:** Fastest response times while maintaining good quality

**💻 Coding & Technical Chat:**
- **Recommended:** `qwen25_coder_32b`
- **Why:** Specialized for programming and technical discussions

**🔄 High-Volume Applications:**
- **Recommended:** `qwen25_14b` or `mistral_small`
- **Why:** Good balance of cost, speed, and quality

## ⚙️ Configuration

### 1. Update Default Model

Edit `config.json`:
```json
{
  "defaults": {
    "model": "qwen25_32b"
  }
}
```

### 2. Web UI Model

Edit `src/ui/web_ui.py` to change the web interface model:
```python
model = OpenRouterModel(
    model_id="qwen/qwen-2.5-32b-instruct",  # Change this
    api_key=api_key,
    max_tokens=8000,
    temperature=0.7,
)
```

### 3. Direct Model Usage

```python
from src.core.models import OpenRouterModel
from src.utils.config import Config

# Create model
model = OpenRouterModel(
    model_id="qwen/qwen-2.5-32b-instruct",
    api_key=Config.OPENROUTER_API_KEY,
    max_tokens=8000,
    temperature=0.7,
)

# Use with smolagents
from src.agents.rag_agent import RAGAgent
agent = RAGAgent(model=model)
```

## 🔧 Advanced Configuration

### Temperature Settings:
- **0.1-0.3:** More focused, deterministic responses
- **0.7:** Balanced creativity and consistency (recommended)
- **0.9-1.0:** More creative and varied responses

### Max Tokens:
- **2000-4000:** Short to medium responses
- **8000:** Long detailed responses (recommended)
- **16000+:** Very long responses (use carefully)

### Context Window Usage:
- All models support large contexts (128K-131K tokens)
- Perfect for long conversations and document analysis
- Qwen models have slightly larger context windows

## 📊 Performance Tips

1. **Start with `qwen25_32b`** - best overall balance
2. **Use `qwen25_7b`** for high-volume or cost-sensitive applications
3. **Switch to `qwen25_72b`** for complex reasoning tasks
4. **Try `qwen25_coder_32b`** for technical/coding conversations
5. **Test multiple models** with your specific use case

## 🚨 Troubleshooting

### Common Issues:

1. **"API key not found"**
   - Set `OPENROUTER_API_KEY` environment variable
   - Check `.env` file in project root

2. **"Model not available"**
   - Check OpenRouter model availability
   - Verify model ID spelling

3. **Slow responses**
   - Try smaller models (7B, 14B)
   - Reduce max_tokens
   - Check your internet connection

4. **Poor quality responses**
   - Try larger models (32B, 72B)
   - Adjust temperature (0.7 is usually good)
   - Check your prompt quality

## 💡 Best Practices

1. **Model Selection:** Start with `qwen25_32b`, adjust based on needs
2. **Temperature:** Use 0.7 for most chat applications
3. **Context:** Leverage the large context windows for better conversations
4. **Testing:** Use `test_chat_models.py` to compare performance
5. **Monitoring:** Track response times and quality for your use case

## 📈 Cost Optimization

- **Qwen models** are generally more cost-effective than Mistral
- **Smaller models** (7B, 14B) are much cheaper for simple tasks
- **Batch requests** when possible to reduce overhead
- **Monitor usage** through OpenRouter dashboard
