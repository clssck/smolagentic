# Comprehensive E2E Testing for Agentic RAG System

This document describes the comprehensive end-to-end testing suite for the Agentic RAG system with live APIs.

## 🎯 Overview

The testing suite validates the complete agentic RAG system including:
- Environment setup and API key validation
- Individual component testing (models, embedders, retrievers, routers)
- Agent functionality testing (RAG, Simple QA, Research, Code agents)
- Intelligent routing system
- Full workflow integration
- Web UI functionality
- Performance metrics and reporting

## 📁 Test Files Created

### 1. `test_full_e2e_agentic_rag.py`
**Main comprehensive test suite** - Tests the entire system end-to-end with live APIs.

**Features:**
- ✅ Environment validation
- ✅ Component testing (models, embedders, retrievers, routers)
- ✅ Agent functionality testing
- ✅ Intelligent routing validation
- ✅ Full workflow integration tests
- ✅ Web UI testing
- ✅ Performance metrics
- ✅ Detailed JSON reporting

**Usage:**
```bash
# Run full test suite
python test_full_e2e_agentic_rag.py

# Run quick tests only
python test_full_e2e_agentic_rag.py --quick

# Test specific agent
python test_full_e2e_agentic_rag.py --agent rag_agent

# Show help
python test_full_e2e_agentic_rag.py --help
```

### 2. `test_environment_only.py`
**Simple environment checker** - Quick validation of API keys and basic setup.

**Features:**
- ✅ API key validation
- ✅ Basic import testing
- ✅ Config validation
- ✅ Setup guidance

**Usage:**
```bash
python test_environment_only.py
```

### 3. `.env.example`
**Environment template** - Example configuration file for API keys.

**Setup:**
```bash
cp .env.example .env
# Edit .env with your actual API keys
```

## 🔧 Required Environment Variables

### Required API Keys
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
DEEPINFRA_API_KEY=your_deepinfra_api_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
```

### Optional API Keys
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### UI Configuration (Optional)
```bash
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
```

## 🧪 Test Categories

### 1. Environment Setup & API Key Validation
- Validates all required environment variables
- Checks API key availability
- Provides setup guidance for missing keys

### 2. Individual Component Testing
- **Models**: Tests OpenRouter and DeepInfra models
- **Embedders**: Tests DeepInfra and local embeddings
- **Retrievers**: Tests Qdrant vector database
- **Routers**: Tests intelligent routing system

### 3. Agent Functionality Testing
- **RAG Agent**: Knowledge base search and retrieval
- **Simple QA Agent**: Basic question answering
- **Research Agent**: Comprehensive analysis
- **Code Agent**: Code generation and explanation

### 4. Intelligent Routing System Testing
- Tests automatic agent selection
- Validates routing decisions
- Measures routing performance

### 5. Full Workflow Integration Testing
- End-to-end workflow validation
- Multi-component integration
- Real-world scenario testing

### 6. Web UI Functionality Testing
- Gradio interface testing
- Message processing validation
- UI component verification

### 7. Performance Metrics Calculation
- Response time analysis
- Success rate calculation
- Component performance comparison

### 8. Test Report Generation
- Detailed JSON reports
- Performance metrics
- Error analysis
- Test duration tracking

## 📊 Test Output

### Console Output
The test suite provides real-time console output with:
- ✅ Success indicators
- ❌ Error messages
- ⏱️ Performance metrics
- 📝 Response previews

### JSON Reports
Detailed test reports are saved as:
```
test_report_YYYYMMDD_HHMMSS.json
```

**Report Structure:**
```json
{
  "test_start_time": "2024-01-01T12:00:00",
  "test_end_time": "2024-01-01T12:05:00",
  "environment": {
    "available_keys": ["OPENROUTER_API_KEY", "DEEPINFRA_API_KEY"],
    "missing_keys": [],
    "config_valid": true
  },
  "components": {
    "models": {...},
    "embedders": {...},
    "retrievers": {...},
    "routers": {...}
  },
  "agents": {
    "functionality": {...}
  },
  "routing": {
    "intelligent_routing": {...}
  },
  "workflows": {
    "integration": {...}
  },
  "web_ui": {
    "functionality": {...}
  },
  "performance": {
    "total_tests": 25,
    "successful_tests": 23,
    "failed_tests": 2,
    "overall_success_rate": 0.92,
    "average_response_time": 2.5
  },
  "summary": {
    "test_duration": "0:05:23",
    "total_components_tested": 6,
    "total_errors": 2
  }
}
```

## 🚀 Quick Start

1. **Setup Environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Run Environment Check:**
   ```bash
   python test_environment_only.py
   ```

3. **Run Quick Tests:**
   ```bash
   python test_full_e2e_agentic_rag.py --quick
   ```

4. **Run Full Test Suite:**
   ```bash
   python test_full_e2e_agentic_rag.py
   ```

## 🎯 Test Scenarios

### Quick Mode (`--quick`)
- Environment validation
- Basic component testing
- Single query per agent
- Skip comprehensive workflows
- Skip UI testing

### Full Mode (default)
- Complete environment validation
- Comprehensive component testing
- Multiple queries per agent
- Full workflow integration
- Web UI testing
- Detailed performance analysis

### Agent-Specific Testing (`--agent`)
- Focus on single agent type
- Comprehensive agent testing
- Skip routing tests
- Detailed agent analysis

## 📈 Performance Benchmarks

The test suite measures:
- **Response Times**: Individual component and agent response times
- **Success Rates**: Component and agent success percentages
- **Throughput**: Requests per second capabilities
- **Error Rates**: Failure analysis and categorization

## 🔍 Troubleshooting

### Common Issues

1. **Missing API Keys**
   - Check .env file exists
   - Verify API key format
   - Test API key validity

2. **Config Loading Errors**
   - Verify config.json syntax
   - Check file permissions
   - Validate JSON structure

3. **Import Errors**
   - Install required dependencies
   - Check Python path
   - Verify package structure

4. **Network Issues**
   - Check internet connectivity
   - Verify API endpoints
   - Test firewall settings

### Debug Mode
For detailed debugging, check the JSON report for:
- Error messages
- Stack traces
- Component details
- Performance metrics

## 🎉 Success Criteria

A successful test run should show:
- ✅ All required API keys present
- ✅ All components initialize successfully
- ✅ All agents respond to queries
- ✅ Routing system works correctly
- ✅ Workflows complete end-to-end
- ✅ Performance within acceptable ranges

## 📞 Support

If you encounter issues:
1. Check the JSON test report for detailed error information
2. Verify all environment variables are set correctly
3. Ensure all dependencies are installed
4. Check network connectivity to API endpoints
