"""
OpenRouter Integration for AI-Scientist-v2

Comprehensive OpenRouter integration providing unified access to 200+ AI models
with advanced features like prompt caching, tool calling, reasoning tokens, and more.
"""

from .client import (
    OpenRouterClient, 
    get_global_client, 
    initialize_openrouter,
    CacheStrategy,
    ToolCall,
    ToolDefinition
)
from .config import (
    OpenRouterConfig,
    RAGConfig,
    StageConfig,
    PipelineStage,
    load_config,
    save_config,
    get_available_models,
    create_default_config
)
from .rag_system import RAGSystem
from .utils import extract_json_between_markers
from .cli import CLIInterface

__version__ = "1.0.0"

__all__ = [
    "OpenRouterClient",
    "get_global_client",
    "initialize_openrouter",
    "CacheStrategy",
    "ToolCall",
    "ToolDefinition",
    "OpenRouterConfig",
    "RAGConfig",
    "StageConfig",
    "PipelineStage",
    "load_config",
    "save_config",
    "get_available_models",
    "create_default_config",
    "RAGSystem",
    "extract_json_between_markers",
    "CLIInterface"
]