# OpenRouter Integration for AI-Scientist-v2

This document provides comprehensive documentation for the OpenRouter integration in AI-Scientist-v2, which enables access to 200+ AI models from multiple providers through a unified API.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Advanced Features](#advanced-features)
7. [Cost Management](#cost-management)
8. [RAG System](#rag-system)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

## Overview

The OpenRouter integration provides a unified interface to access models from:
- OpenAI (GPT-4o, O1, O3, etc.)
- Anthropic (Claude 3.5 Sonnet, Opus, Haiku)
- Google (Gemini 2.0 Flash, 1.5 Pro)
- DeepSeek (DeepSeek-V3)
- X.AI (Grok-2)
- Meta (Llama 3.1 405B)
- And many more!

### Key Benefits

- **Unified API**: Single interface for 200+ models
- **Cost Optimization**: Prompt caching and intelligent model selection
- **Fallback Support**: Automatic failover between models
- **Advanced Features**: Tool calling, reasoning tokens, streaming
- **Budget Tracking**: Comprehensive cost monitoring and alerts

## Features

### 🚀 Core Features
- ✅ 200+ models from multiple providers
- ✅ Prompt caching (25-90% cost savings)
- ✅ Universal tool calling support
- ✅ Reasoning tokens for O1/O3 models
- ✅ Streaming responses
- ✅ Smart fallbacks and error handling

### 💰 Cost Management
- ✅ Real-time cost tracking
- ✅ Budget alerts and limits
- ✅ Cost optimization suggestions
- ✅ Usage analytics and reporting

### 📚 Enhanced Capabilities
- ✅ RAG document ingestion system
- ✅ Interactive CLI configuration
- ✅ Per-stage model selection
- ✅ Enhanced pipeline integration

## Quick Start

### 1. Install Dependencies

```bash
pip install aiohttp backoff PyYAML rich chromadb sentence-transformers
```

### 2. Set API Key

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### 3. Run Configuration Wizard

```bash
python launch_with_openrouter.py --configure
```

This will guide you through:
- API key setup and testing
- Pipeline mode selection (Enhanced vs Original)
- Model configuration for each stage
- Advanced feature setup
- RAG system configuration

### 4. Launch AI-Scientist

```bash
python launch_with_openrouter.py
```

## Configuration

### Configuration Structure

The system uses a YAML configuration file stored at `~/.ai_scientist/openrouter_config.yaml`:

```yaml
api_key: "sk-or-v1-..."
use_enhanced_pipeline: true
use_original_pipeline: false

stage_configs:
  ideation:
    model: "anthropic/claude-3.5-sonnet"
    temperature: 0.7
    max_tokens: 4096
    caching: "ephemeral"
    tools: ["semantic_scholar", "arxiv_search"]
    fallback_models: ["openai/gpt-4o"]
  
  experiment:
    model: "openai/gpt-4o"
    temperature: 0.3
    max_tokens: 4096
    caching: "auto"
    tools: ["code_execution", "data_analysis"]
  
  analysis:
    model: "openai/o1-preview"
    temperature: 0.2
    max_tokens: 4096
    caching: "auto"
    reasoning_config:
      effort: "high"
  
  writeup:
    model: "anthropic/claude-3.5-sonnet"
    temperature: 0.6
    max_tokens: 8192
    caching: "ephemeral"
    tools: ["latex_formatting", "citation_search"]
  
  review:
    model: "openai/gpt-4o"
    temperature: 0.4
    max_tokens: 4096
    caching: "auto"
    tools: ["quality_assessment", "reproducibility_check"]

rag_config:
  enabled: true
  chunk_size: 1000
  chunk_overlap: 200
  embedding_model: "text-embedding-ada-002"
  similarity_threshold: 0.7
  max_results: 10

enable_streaming: false
enable_parallel_processing: true
max_concurrent_requests: 5
enable_cost_optimization: true
budget_alerts: true
```

### Configuration Templates

Pre-built templates are available:

- `research`: Balanced configuration for research work
- `cost_optimized`: Cost-optimized models with caching enabled
- `high_quality`: Premium models for best results
- `experimental`: Cutting-edge models and features

Create from template:
```bash
python -m ai_scientist.openrouter.cli --template research
```

## Usage Examples

### Basic Usage

```python
from ai_scientist.openrouter import initialize_openrouter, get_global_client

# Initialize
client = initialize_openrouter("sk-or-v1-...")

# Simple completion
response, history = await client.get_response(
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    model="anthropic/claude-3.5-sonnet",
    temperature=0.7
)

print(response)
```

### Batch Responses for Ensembling

```python
# Get multiple responses for ensembling
responses, histories = await client.get_batch_responses(
    messages=[{"role": "user", "content": "Design an experiment"}],
    model="openai/gpt-4o",
    n_responses=3,
    temperature=0.8
)

for i, response in enumerate(responses):
    print(f"Response {i+1}: {response}")
```

### Tool Calling

```python
from ai_scientist.openrouter import ToolDefinition

# Define tools
tools = [
    ToolDefinition(
        name="web_search",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    )
]

# Use tools
final_response, history = await client.call_function(
    messages=[{"role": "user", "content": "Research recent AI developments"}],
    model="openai/gpt-4o",
    tools=tools,
    tool_choice="auto"
)
```

### Fallback Models

```python
# Automatic fallback on failure
response, history = await client.get_response_with_fallback(
    messages=[{"role": "user", "content": "Complex analysis task"}],
    primary_model="openai/o1-preview",
    fallback_models=["openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
    temperature=0.3
)
```

### Enhanced Pipeline

```python
from ai_scientist.integration.simple_enhanced_launcher import SimpleEnhancedLauncher

# Initialize enhanced launcher
launcher = SimpleEnhancedLauncher(config, rag_system)

# Run complete pipeline
results = await launcher.run_complete_pipeline("machine_learning_optimization")

print(f"Success: {results['success']}")
for stage, result in results['stages'].items():
    print(f"{stage}: {result['status']}")
```

## Advanced Features

### Prompt Caching

Reduce costs by 25-90% with intelligent caching:

```python
from ai_scientist.openrouter import CacheStrategy

# Automatic caching (recommended)
response, _ = await client.get_response(
    messages=messages,
    model="openai/gpt-4o",
    cache_strategy=CacheStrategy.AUTO
)

# Ephemeral caching for Anthropic/Gemini
response, _ = await client.get_response(
    messages=messages,
    model="anthropic/claude-3.5-sonnet",
    cache_strategy=CacheStrategy.EPHEMERAL
)
```

### Reasoning Models

Leverage O1/O3 reasoning capabilities:

```python
# O1/O3 models with reasoning configuration
response, _ = await client.get_response(
    messages=[{"role": "user", "content": "Solve this complex problem step by step"}],
    model="openai/o1-preview",
    reasoning_config={"effort": "high"}
)
```

### Streaming Responses

Get real-time response generation:

```python
async for chunk in client.stream_response(
    messages=[{"role": "user", "content": "Write a long essay"}],
    model="anthropic/claude-3.5-sonnet"
):
    if chunk.get("choices"):
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            print(delta["content"], end="", flush=True)
```

## Cost Management

### Setting Budgets

```python
from ai_scientist.openrouter.cost_tracker import get_global_cost_tracker

tracker = get_global_cost_tracker()

# Set monthly budget
tracker.set_budget("monthly", 100.0, "monthly")  # $100/month

# Set per-request limit
tracker.set_budget("per_request", 1.0, "request")  # $1/request
```

### Usage Analytics

```python
# Get current usage
usage = tracker.get_current_usage("monthly")
print(f"Total cost: ${usage['total_cost']:.4f}")
print(f"Total tokens: {usage['total_tokens']:,}")

# Generate cost report
report = tracker.generate_cost_report("monthly")
print(json.dumps(report, indent=2))
```

### Cost Optimization

```python
# Get optimization suggestions
suggestions = tracker.get_optimization_suggestions()

for suggestion in suggestions:
    print(f"💡 {suggestion.description}")
    print(f"   Potential savings: ${suggestion.potential_savings:.4f}")
    if suggestion.suggested_model:
        print(f"   Suggested model: {suggestion.suggested_model}")
```

### Export Usage Data

```python
# Export to JSON
json_file = tracker.export_usage_data("json")
print(f"Data exported to: {json_file}")

# Export to CSV
csv_file = tracker.export_usage_data("csv")
print(f"Data exported to: {csv_file}")
```

## RAG System

### Document Ingestion

```python
from ai_scientist.openrouter import RAGSystem
from ai_scientist.openrouter.config import RAGConfig

# Initialize RAG system
rag_config = RAGConfig(enabled=True, chunk_size=1000)
rag_system = RAGSystem(rag_config)

# Ingest documents
doc_id = await rag_system.ingest_file(Path("research_paper.pdf"))
url_id = rag_system.ingest_url("https://arxiv.org/abs/2301.00001")
```

### Document Search

```python
# Search for relevant documents
results = rag_system.search("machine learning algorithms", max_results=5)

for content, score, metadata in results:
    print(f"Score: {score:.3f}")
    print(f"Source: {metadata['title']}")
    print(f"Content: {content[:200]}...")
    print()
```

### Context Integration

```python
# Get context for queries
context = rag_system.get_context_for_query("neural networks", max_context_length=4000)

# Use in prompts
enhanced_prompt = f"{context}\n\nQuestion: How do neural networks work?"
response, _ = await client.get_response(
    messages=[{"role": "user", "content": enhanced_prompt}],
    model="anthropic/claude-3.5-sonnet"
)
```

## API Reference

### OpenRouterClient

```python
class OpenRouterClient:
    def __init__(self, api_key: str = None, enable_cost_tracking: bool = True)
    
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
        n_responses: int = 1,
        **kwargs
    ) -> Tuple[List[str], List[List[Dict[str, Any]]]]
    
    async def call_function(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: List[ToolDefinition],
        tool_choice: str = "auto",
        **kwargs
    ) -> Tuple[str, List[Dict[str, Any]]]
    
    async def stream_response(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]
```

### Configuration Management

```python
def create_default_config() -> OpenRouterConfig
def load_config(config_path: str = None) -> OpenRouterConfig
def save_config(config: OpenRouterConfig, config_path: str = None) -> None
def validate_config(config: OpenRouterConfig) -> List[str]
def create_config_from_template(template_name: str) -> OpenRouterConfig
```

### Cost Tracking

```python
class CostTracker:
    def set_budget(self, budget_type: str, amount: float, period: str = "monthly")
    def record_usage(self, model: str, prompt_tokens: int, completion_tokens: int, ...)
    def get_current_usage(self, period: str = "monthly") -> Dict[str, Any]
    def get_optimization_suggestions(self) -> List[CostOptimizationSuggestion]
    def generate_cost_report(self, period: str = "monthly") -> Dict[str, Any]
    def export_usage_data(self, format: str = "json") -> str
```

## Troubleshooting

### Common Issues

**1. API Key Not Found**
```
ValueError: OpenRouter API key required
```
**Solution:** Set the `OPENROUTER_API_KEY` environment variable or pass it to `initialize_openrouter()`.

**2. Model Not Available**
```
HTTP 404: Model not found
```
**Solution:** Check available models with `client.get_available_models()` or use the model list command:
```bash
python -m ai_scientist.openrouter.cli --models
```

**3. Rate Limiting**
```
HTTP 429: Rate limited
```
**Solution:** The client automatically retries with exponential backoff. Consider:
- Using cheaper models for development
- Implementing request batching
- Setting up fallback models

**4. Configuration Issues**
```
Configuration validation errors
```
**Solution:** Run the configuration validator:
```bash
python -m ai_scientist.openrouter.cli --validate ~/.ai_scientist/openrouter_config.yaml
```

**5. Cost Tracking Not Working**
```
Cost tracking not available
```
**Solution:** Ensure all dependencies are installed:
```bash
pip install pydantic dataclasses-json
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use the test suite:

```bash
python test_openrouter_comprehensive.py
```

### Performance Optimization

**For faster responses:**
- Use `gpt-4o-mini` or `claude-3-haiku` for simple tasks
- Enable parallel processing for batch operations
- Implement prompt caching for repeated queries

**For cost optimization:**
- Use the cost optimization suggestions
- Set budget alerts
- Monitor usage with the cost tracker
- Use templates designed for cost efficiency

### Getting Help

1. **Check the logs**: Enable debug logging for detailed error information
2. **Run the test suite**: `python test_openrouter_comprehensive.py`
3. **Validate configuration**: Use the CLI validation tools
4. **Review usage**: Check cost tracking reports for insights

## Changelog

### Version 2.0.0
- ✅ Complete OpenRouter integration
- ✅ Interactive CLI configuration system
- ✅ Comprehensive cost tracking and budgets
- ✅ RAG document ingestion system
- ✅ Tool calling with universal support
- ✅ Enhanced pipeline with per-stage configuration
- ✅ Prompt caching implementation
- ✅ Fallback model support
- ✅ Streaming response capabilities
- ✅ Production-ready error handling
- ✅ Extensive testing suite
- ✅ Complete documentation

---

For more examples and advanced usage, see the `examples/` directory and test files.