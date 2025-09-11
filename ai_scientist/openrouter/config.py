"""
OpenRouter Configuration Management

Per-stage configuration system for AI-Scientist-v2 pipeline with OpenRouter integration.
Supports model selection, tool configuration, caching strategies, and reasoning settings.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import yaml

from .client import CacheStrategy

logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """Pipeline stages for configuration"""
    IDEATION = "ideation"
    EXPERIMENT = "experiment" 
    ANALYSIS = "analysis"
    WRITEUP = "writeup"
    REVIEW = "review"
    PLOTTING = "plotting"
    VLM = "vlm"
    THEORY_EVOLUTION = "theory_evolution"

@dataclass
class StageConfig:
    """Configuration for a single pipeline stage"""
    # Model configuration
    model: str = "openai/gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    
    # Fallback models
    fallback_models: List[str] = field(default_factory=lambda: ["anthropic/claude-3.5-sonnet"])
    
    # Caching configuration
    caching: str = "auto"  # auto, none, ephemeral, explicit
    
    # Tool calling configuration
    tools: List[str] = field(default_factory=list)
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    
    # Reasoning configuration (for O1/O3 models)
    reasoning_config: Optional[Dict[str, Any]] = None
    
    # Custom prompts
    use_custom_prompts: bool = False
    custom_system_prompt: Optional[str] = None
    
    # Advanced parameters
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    
    # Provider preferences
    provider_preferences: Optional[Dict[str, Any]] = None

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    enabled: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "text-embedding-ada-002"
    vector_store: str = "chroma"
    collection_name: str = "ai_scientist_docs"
    similarity_threshold: float = 0.7
    max_results: int = 10
    max_retrieval_docs: int = 10
    
    # Document ingestion
    auto_ingest: bool = True
    supported_formats: List[str] = field(default_factory=lambda: ["pdf", "txt", "md", "docx"])
    max_file_size: str = "50MB"

@dataclass
class OpenRouterConfig:
    """Complete OpenRouter configuration for AI-Scientist-v2"""
    
    # API Configuration
    api_key: Optional[str] = None
    site_url: str = "https://ai-scientist-v2.com"
    app_name: str = "AI-Scientist-v2"
    
    # Pipeline selection
    use_enhanced_pipeline: bool = True
    use_original_pipeline: bool = False
    
    # Stage configurations
    stage_configs: Dict[str, StageConfig] = field(default_factory=lambda: {
        "ideation": StageConfig(
            model="anthropic/claude-3.5-sonnet",
            temperature=0.7,
            tools=["semantic_scholar", "arxiv_search"],
            caching="ephemeral"
        ),
        "experiment": StageConfig(
            model="openai/gpt-4o",
            temperature=0.3,
            tools=["code_execution", "data_analysis"],
            caching="auto"
        ),
        "analysis": StageConfig(
            model="openai/o1-preview",
            temperature=0.2,
            reasoning_config={"effort": "high"},
            caching="auto"
        ),
        "writeup": StageConfig(
            model="anthropic/claude-3.5-sonnet",
            temperature=0.6,
            tools=["latex_formatting", "citation_search"],
            caching="ephemeral"
        ),
        "review": StageConfig(
            model="openai/gpt-4o",
            temperature=0.4,
            tools=["plagiarism_check", "quality_assessment"],
            caching="auto"
        )
    })
    
    # RAG configuration
    rag_config: RAGConfig = field(default_factory=RAGConfig)
    
    # Global settings
    enable_streaming: bool = False
    enable_parallel_processing: bool = True
    max_concurrent_requests: int = 5
    retry_attempts: int = 3
    
    # Cost optimization
    enable_cost_optimization: bool = True
    max_cost_per_request: Optional[float] = None
    budget_alerts: bool = True

def create_default_config() -> OpenRouterConfig:
    """Create default OpenRouter configuration"""
    return OpenRouterConfig()

def load_config(config_path: Optional[str] = None) -> OpenRouterConfig:
    """
    Load OpenRouter configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OpenRouter configuration
    """
    if config_path is None:
        config_path = os.path.expanduser("~/.ai_scientist/openrouter_config.yaml")
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.info(f"Config file {config_path} not found, creating default configuration")
        config = create_default_config()
        save_config(config, config_path)
        return config
    
    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Convert to dataclass
        config = _dict_to_config(config_data)
        
        logger.info(f"Loaded OpenRouter configuration from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        logger.info("Using default configuration")
        return create_default_config()

def save_config(config: OpenRouterConfig, config_path: Optional[str] = None) -> None:
    """
    Save OpenRouter configuration to file.
    
    Args:
        config: Configuration to save
        config_path: Path to save configuration
    """
    if config_path is None:
        config_path = os.path.expanduser("~/.ai_scientist/openrouter_config.yaml")
    
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert to dictionary
        config_data = _config_to_dict(config)
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved OpenRouter configuration to {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")
        raise

def _dict_to_config(data: Dict[str, Any]) -> OpenRouterConfig:
    """Convert dictionary to OpenRouterConfig"""
    
    # Handle stage configs
    stage_configs = {}
    for stage_name, stage_data in data.get("stage_configs", {}).items():
        stage_configs[stage_name] = StageConfig(**stage_data)
    
    # Handle RAG config
    rag_data = data.get("rag_config", {})
    rag_config = RAGConfig(**rag_data)
    
    # Create main config
    config_data = {k: v for k, v in data.items() if k not in ["stage_configs", "rag_config"]}
    config_data["stage_configs"] = stage_configs
    config_data["rag_config"] = rag_config
    
    return OpenRouterConfig(**config_data)

def _config_to_dict(config: OpenRouterConfig) -> Dict[str, Any]:
    """Convert OpenRouterConfig to dictionary"""
    
    # Convert stage configs
    stage_configs = {}
    for stage_name, stage_config in config.stage_configs.items():
        stage_configs[stage_name] = asdict(stage_config)
    
    # Convert main config
    config_dict = asdict(config)
    config_dict["stage_configs"] = stage_configs
    
    return config_dict

async def get_available_models(client=None) -> List[Dict[str, Any]]:
    """
    Get available models from OpenRouter API.
    
    Args:
        client: OpenRouter client instance
        
    Returns:
        List of available models
    """
    if client is None:
        from .client import get_global_client
        try:
            client = get_global_client()
        except RuntimeError:
            logger.error("OpenRouter client not initialized")
            return []
    
    try:
        models = await client.get_available_models()
        return models
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        return []

def get_models_by_provider(models: List[Dict[str, Any]], provider: str) -> List[Dict[str, Any]]:
    """
    Filter models by provider.
    
    Args:
        models: List of all models
        provider: Provider name (e.g., "openai", "anthropic")
        
    Returns:
        List of models from specified provider
    """
    return [model for model in models if model.get("id", "").startswith(f"{provider}/")]

def get_models_with_tools(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Get models that support tool calling.
    
    Args:
        models: List of all models
        
    Returns:
        List of models with tool calling support
    """
    return [
        model for model in models 
        if "tools" in model.get("supported_parameters", []) or 
           "function_calling" in model.get("supported_parameters", [])
    ]

def get_models_with_caching(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Get models that support prompt caching.
    
    Args:
        models: List of all models
        
    Returns:
        List of models with caching support
    """
    caching_providers = ["openai", "anthropic", "google", "deepseek", "x-ai"]
    return [
        model for model in models 
        if any(model.get("id", "").startswith(f"{provider}/") for provider in caching_providers)
    ]

def validate_config(config: OpenRouterConfig) -> List[str]:
    """
    Validate OpenRouter configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check API key
    if not config.api_key and not os.getenv("OPENROUTER_API_KEY"):
        errors.append("OpenRouter API key not provided")
    
    # Validate stage configurations
    for stage_name, stage_config in config.stage_configs.items():
        if not stage_config.model:
            errors.append(f"No model specified for stage: {stage_name}")
        
        if not 0.0 <= stage_config.temperature <= 2.0:
            errors.append(f"Invalid temperature for stage {stage_name}: {stage_config.temperature}")
        
        if stage_config.max_tokens <= 0:
            errors.append(f"Invalid max_tokens for stage {stage_name}: {stage_config.max_tokens}")
    
    # Validate RAG configuration
    if config.rag_config.chunk_size <= 0:
        errors.append(f"Invalid RAG chunk_size: {config.rag_config.chunk_size}")
    
    if not 0.0 <= config.rag_config.similarity_threshold <= 1.0:
        errors.append(f"Invalid RAG similarity_threshold: {config.rag_config.similarity_threshold}")
    
    return errors

def optimize_config_for_cost(config: OpenRouterConfig, max_budget: Optional[float] = None) -> OpenRouterConfig:
    """
    Optimize configuration for cost reduction.
    
    Args:
        config: Original configuration
        max_budget: Maximum budget per request
        
    Returns:
        Optimized configuration
    """
    optimized_config = _dict_to_config(_config_to_dict(config))  # Deep copy
    
    # Cost-optimized model mappings
    cost_optimized_models = {
        "openai/gpt-4o": "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet": "anthropic/claude-3-haiku",
        "google/gemini-1.5-pro": "google/gemini-1.5-flash"
    }
    
    for stage_config in optimized_config.stage_configs.values():
        # Use cost-optimized models
        if stage_config.model in cost_optimized_models:
            stage_config.fallback_models.insert(0, stage_config.model)  # Keep original as fallback
            stage_config.model = cost_optimized_models[stage_config.model]
        
        # Enable aggressive caching
        if stage_config.caching == "none":
            stage_config.caching = "auto"
        
        # Reduce max tokens for non-critical stages
        if stage_config.max_tokens > 2048:
            stage_config.max_tokens = min(stage_config.max_tokens, 2048)
    
    # Enable cost optimization features
    optimized_config.enable_cost_optimization = True
    if max_budget:
        optimized_config.max_cost_per_request = max_budget
    
    logger.info("Applied cost optimization to configuration")
    return optimized_config

def get_stage_config(config: OpenRouterConfig, stage: Union[str, PipelineStage]) -> StageConfig:
    """
    Get configuration for a specific pipeline stage.
    
    Args:
        config: OpenRouter configuration
        stage: Pipeline stage
        
    Returns:
        Stage configuration
    """
    if isinstance(stage, PipelineStage):
        stage = stage.value
    
    if stage not in config.stage_configs:
        logger.warning(f"No configuration found for stage: {stage}. Using default.")
        return StageConfig()
    
    return config.stage_configs[stage]

def update_stage_config(config: OpenRouterConfig, stage: Union[str, PipelineStage], 
                       updates: Dict[str, Any]) -> None:
    """
    Update configuration for a specific pipeline stage.
    
    Args:
        config: OpenRouter configuration
        stage: Pipeline stage
        updates: Updates to apply
    """
    if isinstance(stage, PipelineStage):
        stage = stage.value
    
    if stage not in config.stage_configs:
        config.stage_configs[stage] = StageConfig()
    
    stage_config = config.stage_configs[stage]
    
    for key, value in updates.items():
        if hasattr(stage_config, key):
            setattr(stage_config, key, value)
        else:
            logger.warning(f"Unknown stage config attribute: {key}")

def create_config_from_template(template_name: str) -> OpenRouterConfig:
    """
    Create configuration from predefined template.
    
    Args:
        template_name: Template name ("research", "cost_optimized", "high_quality", "experimental")
        
    Returns:
        Configuration based on template
    """
    templates = {
        "research": {
            "ideation": {"model": "anthropic/claude-3.5-sonnet", "temperature": 0.8},
            "experiment": {"model": "openai/gpt-4o", "temperature": 0.3},
            "analysis": {"model": "openai/o1-preview", "temperature": 0.2},
            "writeup": {"model": "anthropic/claude-3.5-sonnet", "temperature": 0.6},
            "review": {"model": "openai/gpt-4o", "temperature": 0.4}
        },
        "cost_optimized": {
            "ideation": {"model": "anthropic/claude-3-haiku", "temperature": 0.7},
            "experiment": {"model": "openai/gpt-4o-mini", "temperature": 0.3},
            "analysis": {"model": "openai/gpt-4o-mini", "temperature": 0.2},
            "writeup": {"model": "anthropic/claude-3-haiku", "temperature": 0.6},
            "review": {"model": "openai/gpt-4o-mini", "temperature": 0.4}
        },
        "high_quality": {
            "ideation": {"model": "anthropic/claude-3-opus", "temperature": 0.8},
            "experiment": {"model": "openai/o1-preview", "temperature": 0.3},
            "analysis": {"model": "openai/o1", "temperature": 0.2},
            "writeup": {"model": "anthropic/claude-3-opus", "temperature": 0.6},
            "review": {"model": "openai/o1-preview", "temperature": 0.4}
        },
        "experimental": {
            "ideation": {"model": "x-ai/grok-2", "temperature": 0.9},
            "experiment": {"model": "deepseek/deepseek-v3", "temperature": 0.4},
            "analysis": {"model": "google/gemini-2.0-flash", "temperature": 0.3},
            "writeup": {"model": "meta-llama/llama-3.1-405b", "temperature": 0.7},
            "review": {"model": "x-ai/grok-2", "temperature": 0.5}
        }
    }
    
    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")
    
    config = create_default_config()
    template = templates[template_name]
    
    for stage_name, stage_updates in template.items():
        update_stage_config(config, stage_name, stage_updates)
    
    logger.info(f"Created configuration from template: {template_name}")
    return config