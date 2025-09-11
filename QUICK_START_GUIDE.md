# 🚀 Quick Start Guide - AI-Scientist-v2 with OpenRouter

Get up and running with the enhanced AI-Scientist-v2 in under 5 minutes!

## ⚡ 1-Minute Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements_openrouter.txt
```

### Step 2: Get OpenRouter API Key
1. Visit https://openrouter.ai/
2. Sign up and get your API key
3. Set environment variable:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

### Step 3: Run Enhanced Launcher
```bash
python launch_enhanced_scientist.py
```

Follow the interactive setup wizard - it will guide you through everything!

## 🎯 What You Get

- **200+ AI Models**: Access to OpenAI, Anthropic, Google, Meta, and more
- **Smart Caching**: Automatic cost optimization with prompt caching
- **RAG System**: Ingest papers and documents for enhanced research
- **Per-Stage Config**: Different models for different pipeline stages
- **Interactive CLI**: Beautiful, user-friendly interface

## 📚 Common First Steps

### 1. Test the Connection
```bash
python test_openrouter_integration.py
```

### 2. Ingest Research Papers
```python
# In the launcher, choose option 2 (Manage RAG Documents)
# Then option 1 (Ingest Documents from Files)
# Point to your PDF directory
```

### 3. Run Your First Enhanced Pipeline
```bash
# In the launcher, choose option 1 (Run Full AI-Scientist Pipeline)
# Select "enhanced" pipeline
# Choose models for each stage
```

## 🔧 Minimal Configuration

If you prefer manual setup, create `openrouter_config.yaml`:

```yaml
api_key: "your-openrouter-api-key"
app_name: "AI-Scientist-v2"

stage_configs:
  ideation:
    model: "anthropic/claude-3.5-sonnet"
    temperature: 0.9
  experiment_design:
    model: "openai/o1"
    temperature: 1.0
  code_generation:
    model: "anthropic/claude-3.5-sonnet"
    temperature: 0.3
  writeup:
    model: "anthropic/claude-3.5-sonnet"
    temperature: 0.7

rag_config:
  enabled: true
  chunk_size: 1000
  embedding_model: "text-embedding-3-large"
```

## 🆘 Need Help?

- **Full Documentation**: See `OPENROUTER_COMPLETE_GUIDE.md`
- **Issues**: Check troubleshooting section in the complete guide
- **Support**: Use the help menu in the launcher (option 7)

## 🎉 You're Ready!

The enhanced launcher provides everything you need through its interactive menus. Explore the options and start your AI research journey!

**Happy researching! 🧪✨**