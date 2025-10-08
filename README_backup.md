<div align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist_v2/blob/main/docs/logo_v1.jpg">
    <img src="docs/logo_v1.png" width="215" alt="AI Scientist v2 Logo" />
  </a>
  <h1>
    <b>The AI Scientist-v2: Autonomous Scientific Discovery Platform</b><br>
    <b>Workshop-Level Research via Agentic Tree Search + Comprehensive AI Integration</b>
  </h1>
</div>

<p align="center">
  📚 <a href="https://pub.sakana.ai/ai-scientist-v2/paper">[Paper]</a> |
  📝 <a href="https://sakana.ai/ai-scientist-first-publication/"> [Blog Post]</a> |
  📂 <a href="https://github.com/SakanaAI/AI-Scientist-ICLR2025-Workshop-Experiment"> [ICLR2025 Workshop Experiment]</a>
</p>

**🚀 Enhanced Version Features:**
- 🤖 **200+ AI Models** via OpenRouter integration (OpenAI, Anthropic, Google, Meta, and more)
- 📚 **RAG Knowledge System** with document ingestion (10+ file formats) 
- ⚙️ **Interactive Configuration** with per-stage model selection
- 💰 **Cost Tracking & Optimization** across all providers
- 🔄 **Enhanced Launchers** for flexible workflows
- 📊 **Advanced Analytics** and experiment management

Fully autonomous scientific research systems are becoming increasingly capable, with AI playing a pivotal role in transforming how scientific discoveries are made.
We are excited to introduce The AI Scientist-v2, a generalized end-to-end agentic system that has generated the first workshop paper written entirely by AI and accepted through peer review.

This system autonomously generates hypotheses, runs experiments, analyzes data, and writes scientific manuscripts. Unlike [its predecessor (AI Scientist-v1)](https://github.com/SakanaAI/AI-Scientist), the AI Scientist-v2 removes reliance on human-authored templates, generalizes across Machine Learning (ML) domains, and employs a progressive agentic tree search, guided by an experiment manager agent.

**What's New**: The enhanced version adds comprehensive OpenRouter integration, providing access to 200+ state-of-the-art models, RAG-powered knowledge enhancement, intelligent cost optimization, and flexible configuration options while maintaining full backward compatibility with the original workflow.

> **Note:**
> The AI Scientist-v2 doesn’t necessarily produce better papers than v1, especially when a strong starting template is available. v1 follows well-defined templates, leading to high success rates, while v2 takes a broader, more exploratory approach with lower success rates. v1 works best for tasks with clear objectives and a solid foundation, whereas v2 is designed for open-ended scientific exploration.

> **Caution!**
> This codebase will execute Large Language Model (LLM)-written code. There are various risks and challenges associated with this autonomy, including the potential use of dangerous packages, uncontrolled web access, and the possibility of spawning unintended processes. Ensure that you run this within a controlled sandbox environment (e.g., a Docker container). Use at your own discretion.

## 🚀 Quick Start

### Choose Your Workflow

| Workflow | Best For | Key Features |
|----------|----------|--------------|
| **🤖 Enhanced (Recommended)** | New users, exploration, cost optimization | 200+ models, RAG, cost tracking, interactive setup |
| **⚡ OpenRouter Direct** | Model flexibility, streamlined workflow | Direct OpenRouter access, simplified configuration |
| **🔬 Original** | Proven templates, clear objectives | Template-based, high success rates, established workflow |

```bash
# 🤖 Enhanced Interactive Launcher (Recommended)
python scripts/launch_enhanced_scientist.py

# ⚡ OpenRouter Direct Access  
python scripts/launch_with_openrouter.py --ideas ideas.json

# 🔬 Original Proven Workflow (Legacy)
python archive/original_launchers/launch_scientist_bfts.py --load_ideas ideas.json
```

## Table of Contents

**Getting Started**
1. [Quick Start](#quick-start)
2. [Installation & Setup](#installation--setup)
3. [Configuration Options](#configuration-options)

**Core Workflows**  
4. [Generate Research Ideas](#generate-research-ideas)
5. [Run Scientific Discovery Pipeline](#run-scientific-discovery-pipeline)
6. [Enhanced Features](#enhanced-features)

**Advanced Usage**
7. [OpenRouter Integration](#openrouter-integration)  
8. [RAG Knowledge System](#rag-knowledge-system)
9. [Cost Management](#cost-management)

**Tools & Utilities**
10. [Journal Article Tools](#-journal-article-tools)

**Reference**
10. [Journal Article Tools](#-journal-article-tools)
11. [Migration Guide](#migration-guide)
12. [Citing The AI Scientist-v2](#citing-the-ai-scientist-v2)
13. [Frequently Asked Questions](#frequently-asked-questions)
14. [Acknowledgement](#acknowledgement)

## Installation & Setup

This code is designed to run on Linux with NVIDIA GPUs using CUDA and PyTorch.

### Basic Installation

```bash
# Create a new conda environment
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# Install PyTorch with CUDA support (adjust pytorch-cuda version for your setup)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install PDF and LaTeX tools
conda install anaconda::poppler
conda install conda-forge::chktex

# Install Python package requirements
pip install -r requirements.txt
```

### Enhanced Setup with OpenRouter Integration

```bash
# Install OpenRouter enhanced requirements
pip install -r requirements_openrouter.txt

# Interactive setup (recommended for new users)
python scripts/launch_enhanced_scientist.py --setup
```

Installation usually takes no more than one hour.

## Configuration Options

### 🎛️ API Keys & Model Access

The enhanced version supports multiple configuration methods:

#### Multiple Provider Support
- **OpenRouter**: Single API key for 200+ models (Recommended)
- **Direct Providers**: OpenAI, Anthropic, Google, AWS Bedrock
- **Hybrid Setup**: Combine multiple providers for optimal cost/performance

#### Configuration Methods

**1. Interactive Setup (Recommended)**
```bash
python launch_enhanced_scientist.py --setup
```

**2. Environment Variables**
```bash
# OpenRouter (provides access to 200+ models)
export OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"

# Direct provider access
export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_KEY"
export GEMINI_API_KEY="YOUR_GEMINI_KEY"

# AWS Bedrock for Claude models
export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_KEY"
export AWS_REGION_NAME="your-aws-region"

# Literature search (optional)
export S2_API_KEY="YOUR_S2_KEY_HERE"
```

**3. YAML Configuration Files**
Edit configuration files in `ai_scientist/config/` for advanced customization.

### 🎯 Per-Stage Model Configuration

Configure different models for each pipeline stage:
- **Ideation**: Creative models for hypothesis generation
- **Experimentation**: Code-capable models for implementation  
- **Analysis**: Analytical models for data interpretation
- **Writing**: Writing-optimized models for paper generation

### 📋 Available Models

#### Via OpenRouter (200+ models)
- **OpenAI**: GPT-4o, GPT-4 Turbo, o1-preview, o1-mini
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus/Haiku
- **Google**: Gemini Pro/Ultra, PaLM
- **Meta**: Llama 3.1/3.2 (various sizes)
- **Mistral**: 7B, 8x7B, 8x22B models
- **And many more...**

#### Direct Provider Access
- **OpenAI Models**: Uses `OPENAI_API_KEY`
- **Gemini Models**: Uses `GEMINI_API_KEY` 
- **Claude via AWS Bedrock**: Uses AWS credentials

#### Literature Search
Semantic Scholar API (`S2_API_KEY`) is optional but recommended for enhanced novelty checking and citation management.

## Generate Research Ideas

Before running the full AI Scientist-v2 experiment pipeline, you first use the `ai_scientist/perform_ideation_temp_free.py` script to generate potential research ideas. This script uses an LLM to brainstorm and refine ideas based on a high-level topic description you provide, interacting with tools like Semantic Scholar to check for novelty.

1.  **Prepare a Topic Description:** Create a Markdown file (e.g., `my_research_topic.md`) describing the research area or theme you want the AI to explore. This file should contain sections like `Title`, `Keywords`, `TL;DR`, and `Abstract` to define the scope of the research. Refer to the example file `ai_scientist/ideas/i_cant_believe_its_not_better.md` for the expected structure and content format. Place your file in a location accessible by the script (e.g., the `ai_scientist/ideas/` directory).

2.  **Run the Ideation Script:** Execute the script from the main project directory, pointing it to your topic description file and specifying the desired LLM.

    ```bash
    python ai_scientist/perform_ideation_temp_free.py \
     --workshop-file "ai_scientist/ideas/my_research_topic.md" \
     --model gpt-4o-2024-05-13 \
     --max-num-generations 20 \
     --num-reflections 5
    ```
    *   `--workshop-file`: Path to your topic description Markdown file.
    *   `--model`: The LLM to use for generating ideas (ensure you have the corresponding API key set).
    *   `--max-num-generations`: How many distinct research ideas to attempt generating.
    *   `--num-reflections`: How many refinement steps the LLM should perform for each idea.

3.  **Output:** The script will generate a JSON file named after your input Markdown file (e.g., `ai_scientist/ideas/my_research_topic.json`). This file will contain a list of structured research ideas, including hypotheses, proposed experiments, and related work analysis.

4.  **Proceed to Experiments:** Once you have the generated JSON file containing research ideas, you can proceed to the next section to run the experiments.

This ideation step guides the AI Scientist towards specific areas of interest and produces concrete research directions to be tested in the main experimental pipeline.

## Run Scientific Discovery Pipeline

Using the JSON file generated in the previous ideation step, you can now launch the main AI Scientist-v2 pipeline. Choose your preferred workflow:

### 🤖 Enhanced Interactive Launcher (Recommended)

```bash
# Interactive guided setup
python launch_enhanced_scientist.py

# Or with pre-configured ideas
python launch_enhanced_scientist.py --ideas "ai_scientist/ideas/my_research_topic.json"
```

**Benefits:**
- Interactive model selection per stage
- Real-time cost tracking and optimization
- RAG knowledge integration
- Progress monitoring with rich UI
- Automatic fallback handling

### ⚡ OpenRouter Direct Workflow

```bash
python scripts/launch_with_openrouter.py \
  --ideas "ai_scientist/ideas/my_research_topic.json" \
  --model_ideation gpt-4o \
  --model_experiment claude-3-5-sonnet \
  --model_writeup o1-preview
```

**Benefits:**
- Direct access to 200+ models
- Streamlined configuration
- Cost-optimized model routing
- Advanced prompt caching

### 🔬 Original Proven Workflow (Legacy)

```bash
python archive/original_launchers/launch_scientist_bfts.py \
 --load_ideas "ai_scientist/ideas/my_research_topic.json" \
 --load_code \
 --add_dataset_ref \
 --model_writeup o1-preview-2024-09-12 \
 --model_citation gpt-4o-2024-11-20 \
 --model_review gpt-4o-2024-11-20 \
 --model_agg_plots o3-mini-2025-01-31 \
 --num_cite_rounds 20
```

**Benefits:**
- Template-based approach with high success rates
- Proven workflow for established research patterns
- Detailed tree search configuration via `bfts_config.yaml`

### 🔧 Tree Search Configuration

Key tree search parameters in `archive/legacy_scripts/bfts_config.yaml` (original workflow) or `ai_scientist/config/enhanced_config.yaml` (enhanced workflow):

**Agent Config:**
- `num_workers`: Number of parallel exploration paths
- `steps`: Maximum nodes to explore  
- `num_seeds`: Number of independent starting points

**Search Config:**
- `max_debug_depth`: Maximum debugging attempts per node
- `debug_prob`: Probability of debugging failed nodes
- `num_drafts`: Number of initial root nodes

### 📊 Pipeline Outputs

**During Execution:**
- Real-time progress in terminal (enhanced launchers)
- Tree visualization: `experiments/timestamp_ideaname/logs/0-run/unified_tree_viz.html`
- Cost tracking dashboard (enhanced version)

**Final Outputs:**
- Research paper: `timestamp_ideaname.pdf`
- Experiment logs and data
- Model performance analytics
- Cost breakdown reports

Typical execution time: 2-6 hours depending on complexity and models used.

## Enhanced Features

### 🤖 OpenRouter Integration

**Access 200+ State-of-the-Art Models**
- **OpenAI**: GPT-4o, o1-preview, GPT-4 Turbo series
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus/Haiku  
- **Google**: Gemini Pro/Ultra, PaLM 2
- **Meta**: Llama 3.1/3.2 (8B, 70B, 405B)
- **Mistral**: 7B, 8x7B, 8x22B, Large 2
- **Many more providers and models**

**Intelligent Features:**
- **Model Routing**: Automatic selection based on task requirements
- **Prompt Caching**: Reduce costs up to 50% with intelligent caching
- **Tool Calling**: Enhanced function calling across all models  
- **Cost Optimization**: Real-time cost tracking and budget controls

### 📚 RAG Knowledge System

**Document Processing Support:**
- **Academic Papers**: PDF extraction with citation parsing
- **Code Documentation**: Python, JavaScript, Jupyter notebooks
- **Research Data**: CSV, JSON, text files
- **Books & Reports**: DOCX, markdown, plain text
- **Web Content**: HTML processing and cleaning

**Advanced Capabilities:**
- **Semantic Search**: ChromaDB vector storage with similarity search
- **Context Enhancement**: Automatic relevant context injection
- **Citation Management**: Automatic reference tracking and formatting
- **Knowledge Graphs**: Relationship mapping between concepts

### 💰 Cost Management & Analytics

**Real-Time Tracking:**
- Live cost monitoring across all providers
- Per-stage cost breakdown (ideation, experiment, writing)
- Model-specific usage analytics
- Budget alerts and controls

**Optimization Features:**
- Intelligent model selection for cost/performance balance
- Prompt caching recommendations  
- Alternative model suggestions
- Usage pattern analysis

**Reporting:**
- Detailed cost reports per experiment
- Provider comparison analytics
- Historical usage trends
- ROI analysis for different approaches

### 🔄 Enhanced Configuration & Management

**Multiple Configuration Methods:**
- **Interactive CLI**: Guided setup with real-time validation
- **YAML Configs**: Advanced settings in `ai_scientist/config/`
- **Environment Variables**: Traditional approach for CI/CD
- **Hybrid Setup**: Mix and match approaches as needed

**Advanced Features:**
- **Per-Stage Configuration**: Different models for each pipeline stage
- **Fallback Chains**: Automatic failover to alternative models
- **A/B Testing**: Compare different model configurations
- **Profile Management**: Save and switch between configuration profiles

## OpenRouter Integration

### Getting Started with OpenRouter

**1. Get API Access**
```bash
# Sign up at openrouter.ai and get your API key
export OPENROUTER_API_KEY="your_openrouter_key_here"
```

**2. Model Selection**
```bash
# List available models
python -c "from ai_scientist.openrouter.client import OpenRouterClient; client = OpenRouterClient(); print(client.list_models())"

# Interactive model selection
python launch_enhanced_scientist.py --setup
```

**3. Usage Examples**
```bash
# Use specific models for different stages
python scripts/launch_with_openrouter.py \
  --ideas ideas.json \
  --model_experiment anthropic/claude-3.5-sonnet \
  --model_writeup openai/gpt-4o \
  --enable_caching
```

### Advanced OpenRouter Features

**Model Families:**
- **Research & Analysis**: Claude 3.5 Sonnet, GPT-4o, Gemini Pro
- **Code Generation**: Claude 3.5 Sonnet, GPT-4o, Codestral  
- **Writing & Synthesis**: o1-preview, Claude 3 Opus, GPT-4 Turbo
- **Cost-Effective**: Llama 3.1 70B, Mistral 8x22B, GPT-4o mini

**Cost Optimization:**
- **Prompt Caching**: Automatic caching for repeated contexts
- **Model Routing**: Route to optimal model based on task complexity
- **Batch Processing**: Group similar requests for efficiency
- **Usage Analytics**: Track and optimize spending patterns

## RAG Knowledge System

### Document Ingestion

**Supported Formats:**
```bash
# Add documents to knowledge base
python -c "
from ai_scientist.rag.reasoning_rag_engine import ReasoningRAGEngine
rag = ReasoningRAGEngine()
rag.add_documents(['paper.pdf', 'code.py', 'data.csv'])
"
```

**Processing Pipeline:**
1. **Content Extraction**: Text, code, and metadata extraction
2. **Chunking Strategy**: Intelligent content segmentation  
3. **Vector Encoding**: Semantic embeddings generation
4. **Storage**: ChromaDB vector database storage
5. **Indexing**: Optimized retrieval indexing

### Knowledge Enhancement

**Research Context:**
- **Literature Integration**: Automatic relevant paper discovery
- **Code Context**: Related implementation examples
- **Data Context**: Relevant datasets and benchmarks
- **Historical Context**: Previous experiment results

**Query Enhancement:**
```python
# Example: Enhanced research query
from ai_scientist.rag.reasoning_rag_engine import ReasoningRAGEngine

rag = ReasoningRAGEngine()
enhanced_context = rag.enhance_query(
    "neural network optimization", 
    include_papers=True,
    include_code=True,
    max_context=5000
)
```

## Cost Management

### Real-Time Tracking

**Dashboard Features:**
- Live cost monitoring during experiments
- Model-by-model usage breakdown
- Provider comparison analytics
- Budget threshold alerts

**Cost Controls:**
```bash
# Set budget limits
python launch_enhanced_scientist.py \
  --budget_limit 50.00 \
  --cost_alert_threshold 0.8 \
  --enable_cost_optimization
```

### Optimization Strategies

**Automatic Optimizations:**
- **Model Selection**: Route to cost-effective models when appropriate
- **Prompt Caching**: Cache repeated contexts across requests
- **Batch Processing**: Group similar requests for efficiency
- **Early Stopping**: Stop expensive operations when results are sufficient

**Manual Controls:**
- **Budget Limits**: Hard stops when budget exceeded
- **Model Restrictions**: Limit to specific cost tiers
- **Usage Quotas**: Daily/weekly spending limits
- **Approval Workflows**: Require confirmation for expensive operations

## 📄 Journal Article Tools

The AI-Scientist-v2 includes specialized tools for working with academic papers in LaTeX format. These tools facilitate the extraction and processing of research content for AI analysis and automation.

### Enhanced LaTeX to JSON Converter

Convert LaTeX academic papers to comprehensive structured JSON format for advanced AI processing and analysis. The enhanced version provides granular element extraction for deep AI-driven analysis and manipulation of scientific articles.

#### 🚀 Advanced Features

- **Comprehensive Element Extraction**: Parses every single element from LaTeX documents
- **Hierarchical Document Structure**: Preserves document hierarchy and relationships
- **Pre/Post-Processing Hooks**: Extensible workflow with custom processing scripts
- **Robust Error Handling**: Graceful handling of malformed LaTeX with detailed error reporting
- **Metadata Enrichment**: Rich metadata extraction including line numbers, word counts, and element statistics
- **Multiple Output Formats**: Structured JSON conforming to comprehensive schema

#### 📋 Supported LaTeX Elements

**Document Structure:**
- **Title & Author**: `\title`, `\author`, `\date`, `\maketitle`
- **Abstract**: Complete `abstract` environment with paragraph extraction
- **Table of Contents**: `\tableofcontents`, `\listoffigures`, `\listoftables`
- **Sections**: All levels (`\section`, `\subsection`, `\subsubsection`, `\paragraph`, `\subparagraph`)

**Content Elements:**
- **Paragraphs**: Each paragraph as separate entry with word count metadata
- **Mathematical Content**: 
  - Inline math (`$...$`) and display math (`$$...$$`)
  - Equation environments (`equation`, `align`, `gather`, `split`, etc.)
- **Lists**: `itemize`, `enumerate`, and `description` with individual `\item` extraction
- **Tables**: `table`, `tabular`, and related environments with cell structure parsing
- **Figures**: `figure` environments including `\includegraphics`, captions, and labels

#### 💻 Enhanced CLI Usage

```bash
# Basic comprehensive conversion
python tools/tex_to_json.py --tex_file paper.tex --json_file output.json

# With verbose output and validation
python tools/tex_to_json.py --tex_file manuscript.tex --json_file data.json --verbose

# Using pre/post-processing hooks
python tools/tex_to_json.py \
  --tex_file paper.tex \
  --json_file output.json \
  --pre-hook scripts/clean_latex.py \
  --post-hook scripts/validate_output.py

# Short form with all options
python tools/tex_to_json.py -t paper.tex -j output.json -v --validate
```

#### 📊 JSON Schema Structure

The enhanced converter produces comprehensive JSON with the following structure:

```json
{
  "document": [
    {
      "type": "title",
      "content": "Your Article Title",
      "attributes": {
        "command": "\\title",
        "line_number": 5
      }
    },
    {
      "type": "author", 
      "content": "Your Name",
      "attributes": {
        "command": "\\author",
        "line_number": 6
      }
    },
    {
      "type": "abstract",
      "content": [
        {
          "type": "paragraph",
          "content": "Your abstract text here."
        }
      ],
      "attributes": {
        "word_count": 25
      }
    },
    {
      "type": "section",
      "content": [
        {
          "type": "paragraph",
          "content": "Section content here."
        },
        {
          "type": "equation",
          "content": "E = mc^2",
          "attributes": {
            "environment": "equation",
            "numbered": true
          }
        }
      ],
      "attributes": {
        "title": "Introduction",
        "level": 1,
        "command": "\\section"
      }
    },
    {
      "type": "itemize_list",
      "content": [
        {
          "type": "list_item",
          "content": "First bullet point",
          "attributes": {
            "marker": "bullet",
            "level": 1
          }
        }
      ],
      "attributes": {
        "list_type": "itemize",
        "item_count": 3
      }
    },
    {
      "type": "table",
      "content": [
        ["Header 1", "Header 2", "Header 3"],
        ["Data A", "Data B", "Data C"]
      ],
      "attributes": {
        "caption": "Sample Data Table",
        "label": "tab:sample",
        "rows": 2,
        "columns": 3
      }
    },
    {
      "type": "figure",
      "content": [
        {
          "file": "sample_figure.png",
          "options": "width=0.5\\textwidth"
        }
      ],
      "attributes": {
        "caption": "Sample figure description",
        "label": "fig:sample",
        "graphics_count": 1
      }
    }
  ],
  "metadata": {
    "parser_version": "2.0.0",
    "document_type": "article",
    "total_elements": 15,
    "processing_time": 0.045,
    "has_errors": false
  }
}
```

#### 🔧 Hook System for Extensibility

**Pre-Processing Hooks** (receives tex file path):
```python
#!/usr/bin/env python3
# clean_latex.py - Example pre-hook
import sys
import re

def clean_latex_file(tex_file_path):
    """Clean LaTeX file before processing."""
    with open(tex_file_path, 'r') as f:
        content = f.read()
    
    # Remove comments
    content = re.sub(r'%.*$', '', content, flags=re.MULTILINE)
    
    # Normalize whitespace
    content = re.sub(r'\s+', ' ', content)
    
    with open(tex_file_path, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    clean_latex_file(sys.argv[1])
```

**Post-Processing Hooks** (receives json file path):
```python
#!/usr/bin/env python3
# validate_output.py - Example post-hook
import sys
import json

def validate_json_output(json_file_path):
    """Validate and enhance JSON output."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Add custom validation
    total_words = sum(
        elem.get('attributes', {}).get('word_count', 0) 
        for elem in data['document']
    )
    data['metadata']['total_words'] = total_words
    
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    validate_json_output(sys.argv[1])
```

#### 🎯 Advanced AI Integration Use Cases

**Granular Content Analysis:**
```python
import json

# Load parsed document
with open('paper.json', 'r') as f:
    doc = json.load(f)

# Extract all equations for analysis
equations = [
    elem for elem in doc['document'] 
    if elem['type'] in ['equation', 'inline_equation']
]

# Get section-wise content for targeted processing
sections = {
    elem['attributes']['title']: elem['content']
    for elem in doc['document']
    if elem['type'] == 'section'
}

# Analyze table structures
tables = [
    elem for elem in doc['document']
    if elem['type'] == 'table'
]
```

**Multi-Model Processing Pipeline:**
- **Structure Analysis**: Feed document hierarchy to organization models
- **Content Summarization**: Process individual sections with summarization models
- **Mathematical Validation**: Extract equations for symbolic reasoning systems
- **Citation Mining**: Parse references and bibliography for knowledge graphs
- **Quality Assessment**: Analyze completeness and structure for peer review

#### 🧪 Comprehensive Testing

Run the full test suite to validate all parsing capabilities:

```bash
# Run comprehensive test suite
python test_tex_to_json.py

# Test specific element types
python tools/tex_to_json.py --tex_file test_sample.tex --json_file output.json --verbose

# Validate with project templates
python tools/tex_to_json.py --tex_file ai_scientist/templates/journal_articles/template.tex --json_file test.json
```

**Test Coverage:**
- ✅ Document structure parsing (title, author, abstract)
- ✅ Section hierarchy extraction (all levels)
- ✅ Mathematical content (inline and display equations)
- ✅ List parsing (itemize, enumerate, description)
- ✅ Table structure extraction with cell parsing
- ✅ Figure parsing with graphics and captions
- ✅ JSON schema compliance validation
- ✅ Error handling and recovery
- ✅ Metadata extraction and enrichment

#### 🚀 Performance & Scalability

- **Fast Processing**: Optimized tokenization and parsing algorithms
- **Memory Efficient**: Streaming-based processing for large documents
- **Error Recovery**: Continues parsing despite malformed LaTeX sections
- **Detailed Logging**: Comprehensive progress tracking and error reporting
- **Schema Validation**: Built-in JSON schema compliance checking

The enhanced LaTeX to JSON converter transforms academic papers into rich, structured data ideal for advanced AI research workflows, enabling unprecedented depth of analysis and automated processing of scientific literature.

## 📄 Journal Article Tools

The AI-Scientist-v2 includes specialized tools for working with academic papers in LaTeX format. These tools facilitate the extraction and processing of research content for AI analysis and automation.

### Enhanced LaTeX to JSON Converter

Convert LaTeX academic papers to comprehensive structured JSON format for advanced AI processing and analysis. The enhanced version provides granular element extraction for deep AI-driven analysis and manipulation of scientific articles.

#### 🚀 Advanced Features

- **Comprehensive Element Extraction**: Parses every single element from LaTeX documents
- **Hierarchical Document Structure**: Preserves document hierarchy and relationships
- **Pre/Post-Processing Hooks**: Extensible workflow with custom processing scripts
- **Robust Error Handling**: Graceful handling of malformed LaTeX with detailed error reporting
- **Metadata Enrichment**: Rich metadata extraction including line numbers, word counts, and element statistics
- **Multiple Output Formats**: Structured JSON conforming to comprehensive schema

#### 📋 Supported LaTeX Elements

**Document Structure:**
- **Title & Author**: `\title`, `\author`, `\date`, `\maketitle`
- **Abstract**: Complete `abstract` environment with paragraph extraction
- **Table of Contents**: `\tableofcontents`, `\listoffigures`, `\listoftables`
- **Sections**: All levels (`\section`, `\subsection`, `\subsubsection`, `\paragraph`, `\subparagraph`)

**Content Elements:**
- **Paragraphs**: Each paragraph as separate entry with word count metadata
- **Mathematical Content**: 
  - Inline math (`$...$`) and display math (`$$...$$`)
  - Equation environments (`equation`, `align`, `gather`, `split`, etc.)
- **Lists**: `itemize`, `enumerate`, and `description` with individual `\item` extraction
- **Tables**: `table`, `tabular`, and related environments with cell structure parsing
- **Figures**: `figure` environments including `\includegraphics`, captions, and labels

#### 💻 Enhanced CLI Usage

```bash
# Basic comprehensive conversion
python tools/tex_to_json.py --tex_file paper.tex --json_file output.json

# With verbose output and validation
python tools/tex_to_json.py --tex_file manuscript.tex --json_file data.json --verbose

# Using pre/post-processing hooks
python tools/tex_to_json.py \
  --tex_file paper.tex \
  --json_file output.json \
  --pre-hook scripts/clean_latex.py \
  --post-hook scripts/validate_output.py

# Short form with all options
python tools/tex_to_json.py -t paper.tex -j output.json -v --validate
```

#### 📊 JSON Schema Structure

The enhanced converter produces comprehensive JSON with the following structure:

```json
{
  "document": [
    {
      "type": "title",
      "content": "Your Article Title",
      "attributes": {
        "command": "\\title",
        "line_number": 5
      }
    },
    {
      "type": "author", 
      "content": "Your Name",
      "attributes": {
        "command": "\\author",
        "line_number": 6
      }
    },
    {
      "type": "abstract",
      "content": [
        {
          "type": "paragraph",
          "content": "Your abstract text here."
        }
      ],
      "attributes": {
        "word_count": 25
      }
    },
    {
      "type": "section",
      "content": [
        {
          "type": "paragraph",
          "content": "Section content here."
        },
        {
          "type": "equation",
          "content": "E = mc^2",
          "attributes": {
            "environment": "equation",
            "numbered": true
          }
        }
      ],
      "attributes": {
        "title": "Introduction",
        "level": 1,
        "command": "\\section"
      }
    },
    {
      "type": "itemize_list",
      "content": [
        {
          "type": "list_item",
          "content": "First bullet point",
          "attributes": {
            "marker": "bullet",
            "level": 1
          }
        }
      ],
      "attributes": {
        "list_type": "itemize",
        "item_count": 3
      }
    },
    {
      "type": "table",
      "content": [
        ["Header 1", "Header 2", "Header 3"],
        ["Data A", "Data B", "Data C"]
      ],
      "attributes": {
        "caption": "Sample Data Table",
        "label": "tab:sample",
        "rows": 2,
        "columns": 3
      }
    },
    {
      "type": "figure",
      "content": [
        {
          "file": "sample_figure.png",
          "options": "width=0.5\\\\textwidth"
        }
      ],
      "attributes": {
        "caption": "Sample figure description",
        "label": "fig:sample",
        "graphics_count": 1
      }
    }
  ],
  "metadata": {
    "parser_version": "2.0.0",
    "document_type": "article",
    "total_elements": 15,
    "processing_time": 0.045,
    "has_errors": false
  }
}
```

#### 🔧 Hook System for Extensibility

**Pre-Processing Hooks** (receives tex file path):
```python
#!/usr/bin/env python3
# clean_latex.py - Example pre-hook
import sys
import re

def clean_latex_file(tex_file_path):
    """Clean LaTeX file before processing."""
    with open(tex_file_path, 'r') as f:
        content = f.read()
    
    # Remove comments
    content = re.sub(r'%.*$', '', content, flags=re.MULTILINE)
    
    # Normalize whitespace
    content = re.sub(r'\s+', ' ', content)
    
    with open(tex_file_path, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    clean_latex_file(sys.argv[1])
```

**Post-Processing Hooks** (receives json file path):
```python
#!/usr/bin/env python3
# validate_output.py - Example post-hook
import sys
import json

def validate_json_output(json_file_path):
    """Validate and enhance JSON output."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Add custom validation
    total_words = sum(
        elem.get('attributes', {}).get('word_count', 0) 
        for elem in data['document']
    )
    data['metadata']['total_words'] = total_words
    
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    validate_json_output(sys.argv[1])
```

#### 🎯 Advanced AI Integration Use Cases

**Granular Content Analysis:**
```python
import json

# Load parsed document
with open('paper.json', 'r') as f:
    doc = json.load(f)

# Extract all equations for analysis
equations = [
    elem for elem in doc['document'] 
    if elem['type'] in ['equation', 'inline_equation']
]

# Get section-wise content for targeted processing
sections = {
    elem['attributes']['title']: elem['content']
    for elem in doc['document']
    if elem['type'] == 'section'
}

# Analyze table structures
tables = [
    elem for elem in doc['document']
    if elem['type'] == 'table'
]
```

**Multi-Model Processing Pipeline:**
- **Structure Analysis**: Feed document hierarchy to organization models
- **Content Summarization**: Process individual sections with summarization models
- **Mathematical Validation**: Extract equations for symbolic reasoning systems
- **Citation Mining**: Parse references and bibliography for knowledge graphs
- **Quality Assessment**: Analyze completeness and structure for peer review

#### 🧪 Comprehensive Testing

Run the full test suite to validate all parsing capabilities:

```bash
# Run comprehensive test suite
python test_tex_to_json.py

# Test specific element types
python tools/tex_to_json.py --tex_file test_sample.tex --json_file output.json --verbose

# Validate with project templates
python tools/tex_to_json.py --tex_file ai_scientist/templates/journal_articles/template.tex --json_file test.json
```

**Test Coverage:**
- ✅ Document structure parsing (title, author, abstract)
- ✅ Section hierarchy extraction (all levels)
- ✅ Mathematical content (inline and display equations)
- ✅ List parsing (itemize, enumerate, description)
- ✅ Table structure extraction with cell parsing
- ✅ Figure parsing with graphics and captions
- ✅ JSON schema compliance validation
- ✅ Error handling and recovery
- ✅ Metadata extraction and enrichment

#### 🚀 Performance & Scalability

- **Fast Processing**: Optimized tokenization and parsing algorithms
- **Memory Efficient**: Streaming-based processing for large documents
- **Error Recovery**: Continues parsing despite malformed LaTeX sections
- **Detailed Logging**: Comprehensive progress tracking and error reporting
- **Schema Validation**: Built-in JSON schema compliance checking

The enhanced LaTeX to JSON converter transforms academic papers into rich, structured data ideal for advanced AI research workflows, enabling unprecedented depth of analysis and automated processing of scientific literature.

## Migration Guide

### From Original AI-Scientist-v2

**Backward Compatibility:**
- ✅ All original commands work unchanged
- ✅ Existing `bfts_config.yaml` files supported
- ✅ Same output formats and directory structure
- ✅ Original API key environment variables respected

**Gradual Migration Path:**

**Step 1: Install Enhanced Dependencies**
```bash
pip install -r requirements_openrouter.txt
```

**Step 2: Try Enhanced Launcher**
```bash
# Your existing workflow
python launch_scientist_bfts.py --load_ideas ideas.json

# Enhanced equivalent
python launch_enhanced_scientist.py --ideas ideas.json
```

**Step 3: Add OpenRouter (Optional)**
```bash
export OPENROUTER_API_KEY="your_key"
python scripts/launch_with_openrouter.py --ideas ideas.json
```

**Step 4: Enable Advanced Features (Optional)**
```bash
# Add RAG knowledge enhancement
python launch_enhanced_scientist.py --ideas ideas.json --enable_rag

# Add cost tracking
python launch_enhanced_scientist.py --ideas ideas.json --track_costs
```

### Configuration Migration

**YAML Config Updates:**
```yaml
# Add to existing bfts_config.yaml
openrouter:
  enabled: true
  models:
    experiment: "anthropic/claude-3.5-sonnet" 
    writeup: "openai/gpt-4o"
  caching: true

rag:
  enabled: false  # Start with disabled, enable when ready
  knowledge_base: "research_papers/"
```

**Environment Variables:**
```bash
# Keep existing keys
export OPENAI_API_KEY="existing_openai_key"

# Add new capabilities  
export OPENROUTER_API_KEY="new_openrouter_key"
```

### When to Use Each Workflow

| Use Case | Recommended Workflow | Reason |
|----------|---------------------|---------|
| **First Time User** | Enhanced Interactive | Guided setup, best practices |
| **Proven Research Template** | Original | High success rate, established patterns |
| **Cost Optimization** | OpenRouter Direct | Access to cost-effective models |
| **Exploration & Discovery** | Enhanced with RAG | Advanced context and knowledge |
| **Production Research** | Hybrid Approach | Best of all worlds |

## Citing The AI Scientist-v2

If you use **The AI Scientist-v2** in your research, please cite our work as follows:

```bibtex
@article{aiscientist_v2,
  title={The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search},
  author={Yamada, Yutaro and Lange, Robert Tjarko and Lu, Cong and Hu, Shengran and Lu, Chris and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2504.08066},
  year={2025}
}
```

## Frequently Asked Questions

**Why wasn't a PDF or a review generated for my experiment?**

The AI Scientist-v2 completes experiments with a success rate that depends on the chosen foundation model, and the complexity of the idea. Higher success rates are generally observed when using powerful models like Claude 3.5 Sonnet for the experimentation phase.

**What is the estimated cost per experiment?**

The ideation step cost depends on the LLM used and the number of generations/reflections, but is generally low (a few dollars). For the main experiment pipeline, using Claude 3.5 Sonnet for the experimentation phase typically costs around $15–$20 per run. The subsequent writing phase adds approximately $5 when using the default models specified in the example command. Using GPT-4o for `model_citation` is recommended as it can help reduce writing costs.

**How do I run The AI Scientist-v2 for different subject fields?**

First, perform the [Generate Research Ideas](#generate-research-ideas) step. Create a new Markdown file describing your desired subject field or topic, following the structure of the example `ai_scientist/ideas/i_cant_believe_its_not_better.md`. Run the `perform_ideation_temp_free.py` script with this file to generate a corresponding JSON idea file. Then, proceed to the [Run Scientific Discovery Pipeline](#run-scientific-discovery-pipeline) step, using this JSON file with one of the launcher scripts.

**What should I do if I have problems accessing the Semantic Scholar API?**

The Semantic Scholar API is used to assess the novelty of generated ideas and to gather citations during the paper write-up phase. If you don't have an API key, encounter rate limits, you may be able to skip these phases.

**I encountered a "CUDA Out of Memory" error. What can I do?**

This error typically occurs when the AI Scientist-v2 attempts to load or run a model that requires more GPU memory than available on your system. To resolve this, you can try updating your ideation prompt file (`ai_scientist/ideas/my_research_topic.md`) to suggest using smaller models for the experiments.

## Acknowledgement

The tree search component implemented within the `ai_scientist` directory is built on top of the [AIDE](https://github.com/WecoAI/aideml) project. We thank the AIDE developers for their valuable contributions and for making their work publicly available.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SakanaAI/AI-Scientist-v2&type=Date)](https://star-history.com/#SakanaAI/AI-Scientist-v2&Date)

