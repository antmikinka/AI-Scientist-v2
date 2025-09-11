# Complete OpenRouter Integration Guide for AI-Scientist-v2

## 🚀 Overview

This guide provides comprehensive documentation for the OpenRouter integration in AI-Scientist-v2, enabling unified access to 200+ AI models through a single API interface with advanced features like prompt caching, RAG document ingestion, and per-stage pipeline configuration.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [API Reference](#api-reference)
6. [Advanced Features](#advanced-features)
7. [RAG System](#rag-system)
8. [Pipeline Configuration](#pipeline-configuration)
9. [Cost Optimization](#cost-optimization)
10. [Troubleshooting](#troubleshooting)
11. [Migration Guide](#migration-guide)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install OpenRouter integration dependencies
pip install -r requirements_openrouter.txt

# Or install core dependencies only
pip install aiohttp chromadb rich pyyaml pydantic sentence-transformers
```

### 2. Get OpenRouter API Key
1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up for an account
3. Generate an API key
4. Set environment variable: `export OPENROUTER_API_KEY="your-api-key-here"`

### 3. Run Enhanced Launcher
```bash
python launch_enhanced_scientist.py
```

The interactive setup wizard will guide you through the configuration process.

## 📦 Installation

### Prerequisites
- Python 3.8+
- Valid OpenRouter API key
- At least 4GB RAM (for local embeddings)
- 2GB free disk space (for vector storage)

### Core Dependencies
```bash
# Required dependencies
pip install openai anthropic aiohttp rich pydantic pyyaml

# RAG system dependencies
pip install chromadb sentence-transformers scikit-learn

# Document processing dependencies
pip install PyPDF2 python-docx beautifulsoup4 requests

# Optional: Enhanced embedding models
pip install torch transformers  # For better embeddings
```

### Verification
Run the integration test to verify installation:
```bash
python test_openrouter_integration.py
```

## ⚙️ Configuration

### Automatic Configuration (Recommended)
Use the interactive setup wizard:
```bash
python launch_enhanced_scientist.py
```

### Manual Configuration
Create `openrouter_config.yaml`:
```yaml
api_key: "your-openrouter-api-key"
app_name: "AI-Scientist-v2"
base_url: "https://openrouter.ai/api/v1"

# Model selection for each pipeline stage
stage_configs:
  ideation:
    model: "anthropic/claude-3.5-sonnet"
    temperature: 0.9
    max_tokens: 4096
    fallback_models: ["openai/gpt-4o", "google/gemini-2.0-flash"]
    caching: "auto"
    
  experiment_design:
    model: "openai/o1"
    temperature: 1.0
    max_tokens: 8192
    fallback_models: ["openai/gpt-4o"]
    caching: "ephemeral"
    tools: ["python", "analysis"]
    
  code_generation:
    model: "anthropic/claude-3.5-sonnet"
    temperature: 0.3
    max_tokens: 8192
    fallback_models: ["openai/gpt-4o"]
    caching: "auto"
    
  writeup:
    model: "anthropic/claude-3.5-sonnet"
    temperature: 0.7
    max_tokens: 8192
    fallback_models: ["openai/gpt-4o"]
    caching: "ephemeral"

# RAG configuration
rag_config:
  enabled: true
  chunk_size: 1000
  chunk_overlap: 200
  embedding_model: "text-embedding-3-large"
  vector_store: "chroma"
  collection_name: "ai_scientist_docs"
  similarity_threshold: 0.7
  max_results: 10

# Advanced settings
retry_config:
  max_retries: 3
  backoff_factor: 2
  timeout: 60

logging:
  level: "INFO"
  file: "openrouter.log"
```

## 💡 Usage Examples

### Basic Usage

#### 1. Single Model Query
```python
from ai_scientist.openrouter import get_global_client, initialize_openrouter
from ai_scientist.openrouter.config import load_config

# Load configuration
config = load_config("openrouter_config.yaml")
await initialize_openrouter(config)
client = get_global_client()

# Simple query
messages = [{"role": "user", "content": "Explain quantum computing"}]
response, history = await client.get_response(
    messages=messages,
    model="anthropic/claude-3.5-sonnet",
    temperature=0.7
)
print(response)
```

#### 2. Batch Processing
```python
# Multiple queries with different models
queries = [
    ("Explain machine learning", "anthropic/claude-3.5-sonnet"),
    ("Write Python code for sorting", "openai/gpt-4o"),
    ("Analyze this data", "google/gemini-2.0-flash")
]

results = []
for query, model in queries:
    messages = [{"role": "user", "content": query}]
    response, _ = await client.get_response(messages, model)
    results.append(response)
```

#### 3. Streaming Response
```python
async for chunk in client.stream_response(
    messages=[{"role": "user", "content": "Write a long essay about AI"}],
    model="anthropic/claude-3.5-sonnet"
):
    print(chunk, end='', flush=True)
```

### Advanced Usage

#### 1. Prompt Caching
```python
# Automatic caching
response, _ = await client.get_response(
    messages=messages,
    model="anthropic/claude-3.5-sonnet",
    cache_strategy="auto"  # Automatically optimized
)

# Explicit caching with breakpoints
messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a research assistant."},
            {
                "type": "text", 
                "text": "HUGE_DOCUMENT_CONTENT_HERE",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    },
    {"role": "user", "content": "Summarize this document"}
]

response, _ = await client.get_response(messages, "anthropic/claude-3.5-sonnet")
```

#### 2. Tool Calling
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                }
            }
        }
    }
]

response, history = await client.get_response(
    messages=[{"role": "user", "content": "Calculate 15 * 23 + 7"}],
    model="openai/gpt-4o",
    tools=tools
)
```

#### 3. RAG-Enhanced Queries
```python
from ai_scientist.openrouter.rag_system import RAGSystem

# Initialize RAG system
rag_config = {
    'enabled': True,
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'embedding_model': 'text-embedding-3-large',
    'vector_store': 'chroma'
}
rag = RAGSystem(rag_config)

# Ingest documents
await rag.ingest_file(Path("research_paper.pdf"))
await rag.ingest_file(Path("methodology.txt"))

# RAG-enhanced query
query = "What are the key findings about neural network optimization?"
context = rag.get_context_for_query(query)

messages = [
    {"role": "system", "content": f"Context:\n{context}"},
    {"role": "user", "content": query}
]

response, _ = await client.get_response(messages, "anthropic/claude-3.5-sonnet")
```

## 📚 API Reference

### Core Classes

#### OpenRouterClient
Main client for interacting with OpenRouter API.

```python
class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1")
    
    async def get_response(
        self, 
        messages: List[Dict[str, Any]], 
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        cache_strategy: CacheStrategy = CacheStrategy.AUTO,
        **kwargs
    ) -> Tuple[str, List[Dict[str, Any]]]
    
    async def get_batch_responses(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        n_responses: int = 2,
        **kwargs
    ) -> Tuple[List[str], List[List[Dict[str, Any]]]]
    
    async def stream_response(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs
    ) -> AsyncIterator[str]
```

#### RAGSystem
Document ingestion and retrieval system.

```python
class RAGSystem:
    def __init__(self, config: RAGConfig, storage_dir: str = "./rag_storage")
    
    async def ingest_file(self, file_path: Path) -> Optional[str]
    def ingest_url(self, url: str) -> Optional[str]
    def search(self, query: str, max_results: int = 10) -> List[Tuple[str, float, Dict]]
    def get_context_for_query(self, query: str, max_context_length: int = 4000) -> str
    def delete_document(self, doc_id: str) -> bool
    def list_documents(self) -> List[Dict[str, Any]]
```

### Configuration Classes

#### OpenRouterConfig
Main configuration class for the integration.

```python
@dataclass
class OpenRouterConfig:
    api_key: str
    app_name: str = "AI-Scientist-v2"
    base_url: str = "https://openrouter.ai/api/v1"
    stage_configs: Dict[str, StageConfig] = field(default_factory=dict)
    rag_config: RAGConfig = field(default_factory=RAGConfig)
    retry_config: RetryConfig = field(default_factory=RetryConfig)
```

#### StageConfig
Per-stage configuration for pipeline stages.

```python
@dataclass
class StageConfig:
    model: str = "openai/gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    fallback_models: List[str] = field(default_factory=list)
    caching: str = "auto"  # auto, ephemeral, none
    tools: List[str] = field(default_factory=list)
    custom_prompts: Dict[str, str] = field(default_factory=dict)
```

## 🔧 Advanced Features

### 1. Intelligent Model Selection
The system automatically selects optimal models based on task requirements:

```python
# Reasoning tasks -> O1/O3 models
# Creative writing -> Claude models
# Code generation -> GPT-4 or Claude
# Analysis -> Gemini or Claude

config.enable_intelligent_routing = True
```

### 2. Cost Optimization
Automatic prompt caching and model selection for cost efficiency:

```python
# Monitor costs
cost_tracker = client.get_cost_tracker()
print(f"Total cost: ${cost_tracker.total_cost:.4f}")
print(f"Cache savings: ${cost_tracker.cache_savings:.4f}")

# Set budget limits
client.set_budget_limit(50.0)  # $50 limit
```

### 3. Multi-Provider Fallback
Automatic failover between providers:

```python
config.fallback_strategy = "intelligent"  # intelligent, sequential, parallel
config.fallback_providers = ["openai", "anthropic", "google"]
```

### 4. Custom Prompt Templates
Define reusable prompt templates:

```python
templates = {
    "research_analysis": """
    Analyze the following research paper and provide:
    1. Key contributions
    2. Methodology strengths/weaknesses  
    3. Future research directions
    
    Paper: {document}
    """,
    
    "code_review": """
    Review this code for:
    - Correctness
    - Performance
    - Security
    - Best practices
    
    Code: {code}
    """
}

config.prompt_templates = templates
```

## 📊 Pipeline Configuration

### Stage-Specific Optimization

#### Ideation Stage
```yaml
ideation:
  model: "anthropic/claude-3.5-sonnet"
  temperature: 0.9  # High creativity
  max_tokens: 4096
  caching: "none"  # Ideas should be fresh
  custom_prompts:
    generate_ideas: "Generate novel research ideas in {domain}..."
```

#### Experiment Design Stage
```yaml
experiment_design:
  model: "openai/o1"  # Best reasoning model
  temperature: 1.0
  max_tokens: 8192
  caching: "ephemeral"
  tools: ["python", "statistics"]
  custom_prompts:
    design_experiment: "Design a controlled experiment to test..."
```

#### Code Generation Stage  
```yaml
code_generation:
  model: "anthropic/claude-3.5-sonnet"
  temperature: 0.3  # Low for consistency
  max_tokens: 8192
  caching: "auto"
  tools: ["python", "jupyter"]
  custom_prompts:
    implement_code: "Implement the following algorithm..."
```

#### Writing Stage
```yaml
writeup:
  model: "anthropic/claude-3.5-sonnet"
  temperature: 0.7
  max_tokens: 8192
  caching: "ephemeral"  # Reuse research context
  custom_prompts:
    write_paper: "Write a research paper with the following structure..."
```

## 💰 Cost Optimization

### 1. Prompt Caching Strategies

#### Automatic Caching
```python
# System automatically identifies cacheable content
cache_strategy = "auto"  # Recommended for most use cases
```

#### Explicit Caching
```python
# Mark specific content for caching
messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a research assistant."},
            {
                "type": "text", 
                "text": large_document,  # This will be cached
                "cache_control": {"type": "ephemeral"}
            }
        ]
    }
]
```

### 2. Model Selection Guidelines

| Task Type | Recommended Model | Cost | Performance |
|-----------|------------------|------|-------------|
| Creative Writing | Claude 3.5 Sonnet | $$$ | High |
| Code Generation | Claude 3.5 Sonnet | $$$ | High |
| Complex Reasoning | OpenAI O1 | $$$$ | Highest |
| Simple Tasks | GPT-4O Mini | $ | Medium |
| Long Context | Gemini 1.5 Pro | $$ | High |
| Analysis | Claude 3 Haiku | $ | Medium |

### 3. Budget Management
```python
# Set spending limits
client.set_daily_budget(10.0)
client.set_monthly_budget(100.0)

# Monitor usage
usage = client.get_usage_stats()
print(f"Today: ${usage.daily_cost:.2f}")
print(f"Month: ${usage.monthly_cost:.2f}")
```

## 🗂️ RAG System

### Supported File Formats
- **Documents**: PDF, DOCX, TXT, MD
- **Code**: PY, JS, TS, JSON, CSV
- **Web**: HTML, XML, URLs
- **Notebooks**: IPYNB

### Document Ingestion

#### From Files
```python
# Single file
doc_id = await rag.ingest_file(Path("paper.pdf"))

# Directory (recursive)
for pdf in Path("papers/").glob("**/*.pdf"):
    await rag.ingest_file(pdf)

# With metadata
metadata = {"category": "ml", "year": 2024}
doc_id = await rag.ingest_file(Path("paper.pdf"), metadata)
```

#### From URLs
```python
# Web pages
doc_id = rag.ingest_url("https://arxiv.org/abs/2408.06292")

# PDF URLs
doc_id = rag.ingest_url("https://example.com/paper.pdf")
```

### Document Search and Retrieval

#### Basic Search
```python
results = rag.search("neural network optimization", max_results=5)
for content, score, metadata in results:
    print(f"Score: {score:.3f}")
    print(f"Content: {content[:100]}...")
    print(f"Source: {metadata['title']}")
```

#### Context Generation
```python
query = "What are the latest advances in transformer architecture?"
context = rag.get_context_for_query(query, max_context_length=2000)

# Use context in LLM query
messages = [
    {"role": "system", "content": f"Context:\n{context}"},
    {"role": "user", "content": query}
]
```

### RAG-Enhanced Pipeline
Enable RAG for automatic context injection:

```yaml
rag_config:
  enabled: true
  auto_context_injection: true
  context_stages: ["ideation", "writeup", "review"]
  max_context_per_stage: 4000
```

## 🐛 Troubleshooting

### Common Issues

#### 1. API Key Issues
```bash
# Error: Invalid API key
export OPENROUTER_API_KEY="your-key-here"

# Verify key is set
echo $OPENROUTER_API_KEY

# Test connection
python -c "
from ai_scientist.openrouter import get_global_client
client = get_global_client()
print('Connection OK')
"
```

#### 2. Model Not Found
```python
# Check available models
from ai_scientist.openrouter.config import get_available_models
models = await get_available_models()
print([m for m in models if 'claude' in m])
```

#### 3. Memory Issues with RAG
```yaml
# Reduce chunk size and batch size
rag_config:
  chunk_size: 500    # Reduce from 1000
  batch_size: 10     # Process fewer documents at once
  embedding_model: "all-MiniLM-L6-v2"  # Smaller model
```

#### 4. Rate Limiting
```python
# Increase retry delays
retry_config:
  max_retries: 5
  backoff_factor: 3
  timeout: 120
```

### Performance Optimization

#### 1. Async Batch Processing
```python
import asyncio

async def process_batch(queries):
    tasks = []
    for query in queries:
        task = client.get_response(
            [{"role": "user", "content": query}],
            "openai/gpt-4o-mini"
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

#### 2. Connection Pooling
```python
# Increase connection limits
client = OpenRouterClient(
    api_key="your-key",
    max_connections=20,
    max_keepalive_connections=5
)
```

#### 3. Caching Configuration
```python
# Enable local response caching
config.enable_local_cache = True
config.cache_ttl = 3600  # 1 hour
config.cache_size = 1000  # Max cached responses
```

### Debugging

#### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in config
logging:
  level: "DEBUG"
  file: "debug.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

#### Test Individual Components
```python
# Test OpenRouter connection
python test_openrouter_integration.py

# Test RAG system only
python -c "
from ai_scientist.openrouter.rag_system import RAGSystem
rag = RAGSystem({'enabled': True})
print('RAG system OK')
"

# Test specific model
python -c "
import asyncio
from ai_scientist.openrouter import get_global_client, initialize_openrouter
from ai_scientist.openrouter.config import load_config

async def test():
    config = load_config('openrouter_config.yaml')
    await initialize_openrouter(config)
    client = get_global_client()
    response, _ = await client.get_response(
        [{'role': 'user', 'content': 'Hello'}],
        'openai/gpt-4o-mini'
    )
    print(response)

asyncio.run(test())
"
```

## 🔄 Migration Guide

### From Original AI-Scientist

#### 1. Minimal Migration (Legacy Compatible)
```python
# No code changes needed
# OpenRouter integration works as drop-in replacement

# Just set environment variable
export USE_OPENROUTER=true
export OPENROUTER_API_KEY="your-key"

# Run original launcher
python launch_scientist_bfts.py
```

#### 2. Enhanced Migration
```bash
# 1. Install new dependencies
pip install -r requirements_openrouter.txt

# 2. Run setup wizard
python launch_enhanced_scientist.py

# 3. Configure per-stage models
# Follow interactive prompts

# 4. Optional: Ingest existing papers
# Use RAG management menu
```

#### 3. Configuration Mapping
```yaml
# Old config style
model: "claude-3-5-sonnet-20240620"

# New config style
stage_configs:
  ideation:
    model: "anthropic/claude-3.5-sonnet"
    temperature: 0.9
    fallback_models: ["openai/gpt-4o"]
```

### From Other LLM Integrations

#### LangChain Migration
```python
# Old LangChain code
from langchain.llms import OpenAI
llm = OpenAI(model="gpt-4")

# New OpenRouter code
from ai_scientist.openrouter import get_global_client
client = get_global_client()
```

#### Direct API Migration
```python
# Old direct OpenAI API
import openai
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages
)

# New OpenRouter API
response, history = await client.get_response(
    messages=messages,
    model="openai/gpt-4"
)
```

## 📖 Additional Resources

### Documentation
- [OpenRouter API Documentation](https://openrouter.ai/docs)
- [Original AI-Scientist Paper](https://arxiv.org/abs/2408.06292)
- [Prompt Caching Best Practices](./openrouter-docs/openrouter-prompt-caching.md)

### Examples
- [Configuration Examples](./config/examples/)
- [Pipeline Templates](./templates/)
- [RAG Workflows](./examples/rag_workflows/)

### Community
- [GitHub Issues](https://github.com/SakanaAI/AI-Scientist/issues)
- [Discord Community](https://discord.gg/sakana-ai)
- [OpenRouter Discord](https://discord.gg/openrouter)

---

## 🎉 Conclusion

The OpenRouter integration transforms AI-Scientist-v2 into a powerful, unified research platform with access to the latest AI models, intelligent caching, and RAG-enhanced capabilities. Whether you're conducting cutting-edge research or building production workflows, this integration provides the tools and flexibility you need.

For support, please check the troubleshooting section above or reach out through the community channels.

**Happy researching! 🚀🧪**