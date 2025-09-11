# OpenRouter Integration Guide for AI-Scientist-v2

## 🚀 Overview

This guide covers the complete OpenRouter integration for AI-Scientist-v2, providing unified access to 200+ AI models with advanced features including cost optimization, RAG document integration, and per-stage model configuration.

## ✨ Key Features

### 🌐 Unified LLM Access
- **200+ Models**: Access models from OpenAI, Anthropic, Google, Meta, DeepSeek, and more
- **Single API**: One interface for all providers
- **Automatic Routing**: Intelligent model selection and fallback
- **Cost Optimization**: Automatic provider selection for best pricing

### 🎯 Advanced Capabilities
- **Prompt Caching**: 25-90% cost reduction through intelligent caching
- **Reasoning Tokens**: Enhanced outputs for complex reasoning tasks
- **Tool Calling**: Universal function calling across all models
- **Streaming**: Real-time response generation
- **RAG Integration**: Document ingestion and context-aware responses

### ⚙️ Per-Stage Configuration
- **Independent Model Selection**: Different models for each pipeline stage
- **Custom Parameters**: Temperature, max tokens, caching strategy per stage
- **Tool Configuration**: Stage-specific tool access
- **Fallback Models**: Automatic failover for reliability

## 🛠️ Installation

### 1. Install Dependencies

```bash
# Install OpenRouter integration dependencies
pip install -r requirements_openrouter.txt

# Or install core dependencies only
pip install openai anthropic rich pydantic PyYAML chromadb
```

### 2. Get OpenRouter API Key

1. Visit [OpenRouter](https://openrouter.ai)
2. Create an account or log in
3. Navigate to Settings → API Keys
4. Generate a new API key
5. Set environment variable:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## 🚀 Quick Start

### Option 1: Interactive Setup

```bash
# Run configuration wizard
python launch_with_openrouter.py --configure
```

This launches an interactive wizard that guides you through:
- OpenRouter API configuration
- Pipeline selection (original vs enhanced)
- Per-stage model configuration
- RAG system setup
- Testing and validation

### Option 2: Environment Variables

```bash
# Set required environment variables
export OPENROUTER_API_KEY="your-api-key-here"
export USE_OPENROUTER="true"

# Launch with defaults
python launch_with_openrouter.py
```

### Option 3: Configuration File

Create a configuration file and launch:

```bash
# Generate template
python -c "from ai_scientist.openrouter import ConfigManager; ConfigManager().export_config_template('config.yaml')"

# Edit config.yaml with your settings
# Then launch
python launch_with_openrouter.py --config-file config.yaml
```

## 📋 Configuration

### System Configuration Structure

```yaml
openrouter:
  api_key: "your-api-key"
  site_name: "AI-Scientist-v2"
  default_model: "anthropic/claude-3.5-sonnet"
  enable_fallbacks: true
  max_retries: 3
  timeout: 120

pipeline:
  use_enhanced_pipeline: true
  enable_theory_evolution: true
  enable_rag_integration: true
  enable_multi_agent: true
  enable_parallel_processing: false

rag:
  vector_store: "chroma"
  embedding_model: "text-embedding-3-large"
  chunk_size: 1000
  chunk_overlap: 200
  max_retrieval_docs: 10
  similarity_threshold: 0.7
  enable_reranking: true

stage_configs:
  ideation:
    model: "anthropic/claude-3.5-sonnet"
    temperature: 0.7
    max_tokens: 4096
    tools: ["semantic_scholar", "arxiv_search"]
    caching: "ephemeral"
    reasoning:
      effort: "high"
    fallback_models: ["openai/gpt-4o", "google/gemini-2.0-flash"]
  
  experiments:
    model: "openai/gpt-4o"
    temperature: 0.3
    max_tokens: 8192
    tools: ["code_execution", "data_analysis"]
    caching: "ephemeral"
    reasoning:
      max_tokens: 2000
    fallback_models: ["anthropic/claude-3.5-sonnet"]
  
  # ... additional stages
```

### Per-Stage Configuration Options

Each pipeline stage can be configured independently:

| Setting | Description | Options |
|---------|-------------|---------|
| `model` | Primary model for the stage | Any OpenRouter model |
| `temperature` | Sampling temperature | 0.0 - 2.0 |
| `max_tokens` | Maximum response length | 1 - context_length |
| `tools` | Available tools | List of tool names |
| `caching` | Caching strategy | `auto`, `ephemeral`, `disabled` |
| `reasoning` | Reasoning configuration | `effort`, `max_tokens` |
| `streaming` | Enable streaming | `true`, `false` |
| `fallback_models` | Backup models | List of model names |

## 📚 RAG Document Integration

### Supported Document Types

- **PDF**: Research papers, books, documentation
- **Text**: Plain text files, Markdown
- **DOCX**: Microsoft Word documents
- **URLs**: Web pages, online papers

### Document Ingestion

#### Manual Ingestion

```python
from ai_scientist.openrouter import RAGSystem, RAGConfig

# Initialize RAG system
config = RAGConfig()
rag = RAGSystem(config)

# Ingest documents
doc_id = rag.ingest_file(Path("paper.pdf"))
doc_id = rag.ingest_url("https://arxiv.org/abs/2301.00001")
```

#### Automatic Ingestion

Place documents in the `./documents` directory:

```
documents/
├── papers/
│   ├── attention_is_all_you_need.pdf
│   ├── gpt_paper.pdf
│   └── bert_paper.pdf
├── books/
│   └── deep_learning_book.pdf
└── notes/
    ├── research_notes.md
    └── methodology.txt
```

The system will automatically discover and offer to ingest these documents.

### RAG Usage in Pipeline

Once documents are ingested, they're automatically available as context:

```python
# RAG-enhanced query
query = "What are the key innovations in transformer architecture?"
context = rag.get_context_for_query(query)

# Context is automatically injected into prompts
response = llm_client.get_response(
    messages=[
        {"role": "system", "content": f"{context}\n\nYou are a research assistant."},
        {"role": "user", "content": query}
    ]
)
```

## 🔧 Advanced Features

### Prompt Caching

Reduce costs by 25-90% through intelligent prompt caching:

```python
# Automatic caching (OpenAI, DeepSeek, Grok)
response = client.get_response(
    messages=messages,
    model="openai/gpt-4o",
    caching=CacheStrategy.AUTO
)

# Explicit caching (Anthropic, Gemini)
messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful assistant."},
            {
                "type": "text",
                "text": "LARGE_CONTEXT_DOCUMENT_HERE",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    }
]
```

### Reasoning Tokens

Enable enhanced reasoning for complex tasks:

```python
# Configure reasoning tokens
reasoning_config = {
    "effort": "high",      # low, medium, high
    "max_tokens": 4000     # Maximum reasoning tokens
}

response = client.get_response(
    messages=messages,
    model="openai/o1",
    reasoning_config=reasoning_config
)
```

### Tool Calling

Universal tool calling across all supported models:

```python
# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search for academic papers",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    }
]

# Use with any model
response = client.get_response(
    messages=messages,
    model="google/gemini-2.0-flash",
    tools=tools
)

# Handle tool calls
if response.tool_calls:
    results = client.handle_tool_calls(response.tool_calls, tool_registry)
```

### Streaming Responses

Real-time response generation:

```python
# Enable streaming
for chunk in client.get_response(
    messages=messages,
    model="anthropic/claude-3.5-sonnet",
    streaming=True
):
    print(chunk, end="", flush=True)
```

## 🎛️ Model Recommendations

### By Pipeline Stage

| Stage | Recommended Models | Rationale |
|-------|-------------------|-----------|
| **Ideation** | Claude 3.5 Sonnet, GPT-4o | Creative reasoning, research synthesis |
| **Experiments** | GPT-4o, DeepSeek Coder | Code generation, data analysis |
| **Plotting** | GPT-4o, Claude 3.5 Sonnet | Data visualization, chart interpretation |
| **Writeup** | Claude 3.5 Sonnet, Gemini Pro | Long-form writing, academic style |
| **Review** | O1-Mini, Claude 3 Opus | Critical analysis, reasoning |
| **Theory Evolution** | O1, Claude 3 Opus | Deep reasoning, hypothesis generation |

### Cost Optimization Tips

1. **Use smaller models for simple tasks**: GPT-4o-mini for preprocessing
2. **Enable caching**: Reduce costs by 50-90% for repeated content
3. **Set appropriate max_tokens**: Avoid unnecessary token generation
4. **Use fallback models**: Cheaper alternatives when primary model fails
5. **Monitor usage**: Check OpenRouter dashboard for cost tracking

## 🔄 Migration from Legacy System

### Automatic Migration

The system maintains full backward compatibility:

```python
# Legacy code continues to work
client, model = create_client("claude-3-5-sonnet-20240620")
response, history = get_response_from_llm(
    prompt="Hello",
    client=client,
    model=model,
    system_message="You are helpful"
)
```

### OpenRouter Enhancement

Enable OpenRouter for immediate benefits:

```bash
export USE_OPENROUTER="true"
export OPENROUTER_API_KEY="your-key"
```

Legacy model names are automatically mapped:
- `claude-3-5-sonnet-20240620` → `anthropic/claude-3.5-sonnet`
- `gpt-4o` → `openai/gpt-4o`
- `o1-mini` → `openai/o1-mini`

## 🛡️ Error Handling and Fallbacks

### Automatic Fallbacks

The system provides multiple levels of fallback:

1. **Primary Model**: Your configured choice
2. **Fallback Models**: Stage-specific alternatives
3. **Provider Fallback**: OpenRouter automatic routing
4. **Legacy Client**: Direct API fallback

### Error Recovery

```python
# Comprehensive error handling
try:
    response = client.get_response(messages, model="primary/model")
except Exception as primary_error:
    # Automatic fallback to secondary model
    try:
        response = client.get_response(messages, model="fallback/model")
    except Exception as fallback_error:
        # Legacy client fallback
        legacy_client, legacy_model = create_legacy_client(model)
        response = legacy_get_response(messages, legacy_client, legacy_model)
```

### Rate Limit Handling

Built-in exponential backoff and retry logic:

```python
@backoff.on_exception(
    backoff.expo,
    (RateLimitError, APITimeoutError, InternalServerError),
    max_tries=3
)
def get_response(...):
    # Automatic retry with exponential backoff
```

## 📊 Monitoring and Analytics

### Usage Tracking

Monitor API usage and costs:

```python
# Get usage statistics
stats = client.get_usage_stats()
print(f"Tokens used: {stats['total_tokens']}")
print(f"Estimated cost: ${stats['estimated_cost']}")

# Per-model breakdown
for model, usage in stats['by_model'].items():
    print(f"{model}: {usage['tokens']} tokens, ${usage['cost']}")
```

### Performance Monitoring

Track response times and success rates:

```python
# Enable performance tracking
import time
start_time = time.time()

response = client.get_response(messages)

end_time = time.time()
response_time = end_time - start_time

logger.info(f"Response time: {response_time:.2f}s")
```

## 🧪 Testing

### Configuration Testing

Test your configuration before running experiments:

```bash
# Test OpenRouter connection
python -c "
from ai_scientist.openrouter import get_global_client
client = get_global_client()
response, _ = client.get_response(
    messages=[{'role': 'user', 'content': 'Hello'}],
    max_tokens=10
)
print('✅ OpenRouter connection successful')
"
```

### Model Testing

Verify specific models work correctly:

```python
# Test model availability
from ai_scientist.openrouter import ModelCapabilities

model = "anthropic/claude-3.5-sonnet"
print(f"Supports tools: {ModelCapabilities.supports_tools(model)}")
print(f"Supports caching: {ModelCapabilities.supports_caching(model)}")
print(f"Supports streaming: {ModelCapabilities.supports_streaming(model)}")
```

### RAG Testing

Test document ingestion and retrieval:

```python
# Test RAG system
rag = RAGSystem(config.rag)

# Test document ingestion
doc_id = rag.ingest_file(Path("test_document.pdf"))
print(f"Ingested document: {doc_id}")

# Test retrieval
results = rag.search("test query")
print(f"Found {len(results)} relevant chunks")
```

## 🚨 Troubleshooting

### Common Issues

#### 1. OpenRouter API Key Issues

```bash
# Verify API key is set
echo $OPENROUTER_API_KEY

# Test API key validity
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/models | head
```

#### 2. Model Not Available

```python
# Check available models
from ai_scientist.openrouter import get_available_models
models = get_available_models()
print(models)

# Check model capabilities
from ai_scientist.openrouter import ModelCapabilities
print(ModelCapabilities.supports_tools("your-model"))
```

#### 3. RAG System Issues

```python
# Check RAG statistics
rag = RAGSystem(config.rag)
stats = rag.get_statistics()
print(f"Documents: {stats['total_documents']}")
print(f"Vector store: {stats['vector_store']}")

# Test vector store connection
try:
    results = rag.search("test")
    print("✅ Vector store working")
except Exception as e:
    print(f"❌ Vector store error: {e}")
```

#### 4. Configuration Issues

```python
# Validate configuration
from ai_scientist.openrouter import ConfigManager
manager = ConfigManager()
issues = manager.validate_config(config)
if issues:
    print("Configuration issues:")
    for issue in issues:
        print(f"  • {issue}")
```

### Performance Optimization

#### 1. Optimize Chunk Size

```python
# Test different chunk sizes for RAG
chunk_sizes = [500, 1000, 1500, 2000]
for size in chunk_sizes:
    config.rag.chunk_size = size
    # Test retrieval quality
```

#### 2. Model Selection

```python
# Benchmark different models
models = ["openai/gpt-4o-mini", "anthropic/claude-3-haiku", "google/gemini-1.5-flash"]
for model in models:
    start_time = time.time()
    response = client.get_response(messages, model=model)
    end_time = time.time()
    print(f"{model}: {end_time - start_time:.2f}s")
```

#### 3. Caching Optimization

```python
# Monitor cache hit rates
cache_stats = client.get_cache_statistics()
print(f"Cache hit rate: {cache_stats['hit_rate']:.1%}")
print(f"Cost savings: ${cache_stats['savings']:.2f}")
```

## 🤝 Contributing

### Adding New Models

1. Update `AVAILABLE_LLMS` in `llm.py`
2. Add model mapping in `LEGACY_MODEL_MAPPING`
3. Update model capabilities in `ModelCapabilities`
4. Test with various pipeline stages

### Adding New Tools

1. Implement tool function
2. Add to tool registry
3. Update stage configurations
4. Document tool usage

### Improving RAG

1. Add new document processors
2. Implement new vector stores
3. Enhance embedding strategies
4. Optimize retrieval algorithms

## 📞 Support

### Community Resources

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides and examples
- **Discord**: Community discussion and support

### Professional Support

- **Priority Support**: Enhanced SLA and dedicated assistance
- **Custom Integration**: Tailored deployment and configuration
- **Training**: Team training and best practices workshops

## 📜 License

This OpenRouter integration is provided under the same license as AI-Scientist-v2. See LICENSE file for details.

## 🙏 Acknowledgments

- **OpenRouter Team**: For providing unified LLM access
- **AI-Scientist Original Authors**: For the foundational research automation platform
- **Open Source Community**: For continuous improvements and contributions

---

**Ready to supercharge your AI research? Start with the configuration wizard:**

```bash
python launch_with_openrouter.py --configure
```