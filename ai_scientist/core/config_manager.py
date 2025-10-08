"""
Unified Configuration Management System for AI-Scientist-v2

Provides centralized configuration loading, validation, and management
across all system components with proper dependency injection.
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Type, TypeVar, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from functools import lru_cache

# Type variable for generic config loading
T = TypeVar('T')

logger = logging.getLogger(__name__)

class ConfigSource(Enum):
    """Configuration source priorities"""
    ENVIRONMENT = 1
    USER_CONFIG = 2
    PROJECT_CONFIG = 3
    DEFAULT_CONFIG = 4

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str = "sqlite:///ai_scientist.db"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    timeout: int = 30

@dataclass
class SecurityConfig:
    """Security and authentication configuration"""
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    encryption_key: Optional[str] = None
    session_timeout: int = 3600
    max_login_attempts: int = 5
    rate_limit_requests: int = 100

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: str = "10MB"
    backup_count: int = 5
    structured_logging: bool = True

@dataclass
class PerformanceConfig:
    """Performance and resource management"""
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    cache_size_mb: int = 512
    timeout_seconds: int = 300
    enable_optimizations: bool = True

@dataclass
class RAGConfig:
    """RAG System Configuration - Unified Definition"""
    enabled: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "text-embedding-ada-002"
    vector_store: str = "chroma"
    collection_name: str = "ai_scientist_docs"
    similarity_threshold: float = 0.7
    max_results: int = 10
    max_retrieval_docs: int = 10

    # Document ingestion settings
    auto_ingest: bool = True
    supported_formats: list = field(default_factory=lambda: ["pdf", "txt", "md", "docx"])
    max_file_size: str = "50MB"

    # Advanced settings
    enable_hybrid_search: bool = False
    enable_semantic_cache: bool = True
    cache_ttl: int = 3600

    # Reasoning integration
    reasoning_depth: int = 3
    enable_page_index: bool = True
    page_index_max_context: int = 8192

@dataclass
class CoreConfig:
    """Root configuration for AI-Scientist-v2"""

    # Core settings
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)

    # Custom configuration sections
    custom_sections: Dict[str, Any] = field(default_factory=dict)

class ConfigManager:
    """
    Centralized configuration management with environment-aware loading
    and validation.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent / "config"
        self._config_cache: Dict[str, Any] = {}
        self._config_sources: Dict[str, ConfigSource] = {}

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self, config_type: Type[T], config_name: str = None) -> T:
        """
        Load configuration with environment-aware fallback.

        Args:
            config_type: The dataclass type to load
            config_name: Optional configuration name (defaults to class name)

        Returns:
            Loaded and validated configuration instance
        """
        if config_name is None:
            config_name = config_type.__name__.lower().replace('config', '')

        # Check cache first
        cache_key = f"{config_name}_{config_type.__name__}"
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]

        # Load configuration with fallback chain
        config_data = self._load_config_with_fallbacks(config_name)

        # Convert to dataclass
        config = self._dict_to_dataclass(config_data, config_type)

        # Cache the result
        self._config_cache[cache_key] = config
        self._config_sources[cache_key] = self._determine_source(config_name)

        logger.info(f"Loaded {config_type.__name__} from {self._config_sources[cache_key].name}")
        return config

    def _load_config_with_fallbacks(self, config_name: str) -> Dict[str, Any]:
        """Load configuration with environment-aware fallback chain."""

        # Priority 1: Environment variables
        env_config = self._load_from_environment(config_name)
        if env_config:
            return env_config

        # Priority 2: User-specific config
        user_config = self._load_from_file(self.config_dir / f"{config_name}_user.yaml")
        if user_config:
            return user_config

        # Priority 3: Project-level config
        project_config = self._load_from_file(self.config_dir / f"{config_name}.yaml")
        if project_config:
            return project_config

        # Priority 4: Default configuration
        return self._get_default_config(config_name)

    def _load_from_environment(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Load configuration from environment variables."""
        prefix = f"AI_SCIENTIST_{config_name.upper()}_"
        env_vars = {k: v for k, v in os.environ.items() if k.startswith(prefix)}

        if not env_vars:
            return None

        config = {}
        for key, value in env_vars.items():
            # Convert AI_SCIENTIST_RAG_CHUNK_SIZE -> chunk_size
            config_key = key[len(prefix):].lower()

            # Convert string values to appropriate types
            try:
                if value.lower() in ('true', 'false'):
                    config[config_key] = value.lower() == 'true'
                elif '.' in value:
                    config[config_key] = float(value)
                else:
                    config[config_key] = int(value)
            except ValueError:
                config[config_key] = value

        return config

    def _load_from_file(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """Load configuration from YAML file."""
        if not config_path.exists():
            return None

        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return None

    def _get_default_config(self, config_name: str) -> Dict[str, Any]:
        """Get default configuration based on config name."""
        if config_name == 'rag':
            return asdict(RAGConfig())
        elif config_name == 'core':
            return asdict(CoreConfig())
        else:
            return {}

    def _dict_to_dataclass(self, data: Dict[str, Any], config_type: Type[T]) -> T:
        """Convert dictionary to dataclass with proper type handling."""
        try:
            return config_type(**data)
        except TypeError as e:
            logger.error(f"Configuration validation failed for {config_type.__name__}: {e}")
            raise

    def _determine_source(self, config_name: str) -> ConfigSource:
        """Determine which source was used for configuration."""
        prefix = f"AI_SCIENTIST_{config_name.upper()}_"
        if any(k.startswith(prefix) for k in os.environ):
            return ConfigSource.ENVIRONMENT

        user_config_path = self.config_dir / f"{config_name}_user.yaml"
        if user_config_path.exists():
            return ConfigSource.USER_CONFIG

        project_config_path = self.config_dir / f"{config_name}.yaml"
        if project_config_path.exists():
            return ConfigSource.PROJECT_CONFIG

        return ConfigSource.DEFAULT_CONFIG

    def save_config(self, config: Any, config_name: str,
                   user_specific: bool = False) -> None:
        """Save configuration to file."""
        config_data = asdict(config)

        filename = f"{config_name}{'_user' if user_specific else ''}.yaml"
        config_path = self.config_dir / filename

        try:
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            raise

    def validate_config(self, config: Any) -> list:
        """Validate configuration and return list of errors."""
        errors = []

        # Type validation is handled by dataclass
        # Add custom validation rules here
        if hasattr(config, 'rag'):
            rag_config = config.rag
            if rag_config.chunk_size <= 0:
                errors.append("RAG chunk_size must be positive")
            if not 0.0 <= rag_config.similarity_threshold <= 1.0:
                errors.append("RAG similarity_threshold must be between 0 and 1")

        if hasattr(config, 'performance'):
            perf_config = config.performance
            if perf_config.max_workers <= 0:
                errors.append("Performance max_workers must be positive")
            if perf_config.memory_limit_gb <= 0:
                errors.append("Performance memory_limit_gb must be positive")

        return errors

    def reload_config(self, config_type: Type[T], config_name: str = None) -> T:
        """Force reload configuration from sources."""
        if config_name is None:
            config_name = config_type.__name__.lower().replace('config', '')

        cache_key = f"{config_name}_{config_type.__name__}"

        # Clear cache
        if cache_key in self._config_cache:
            del self._config_cache[cache_key]

        # Reload
        return self.load_config(config_type, config_name)

    def get_config_source(self, config_name: str, config_type: Type) -> ConfigSource:
        """Get the source of a loaded configuration."""
        cache_key = f"{config_name}_{config_type.__name__}"
        return self._config_sources.get(cache_key, ConfigSource.DEFAULT_CONFIG)

# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def load_config(config_type: Type[T], config_name: str = None) -> T:
    """Convenience function to load configuration."""
    return get_config_manager().load_config(config_type, config_name)

def get_rag_config() -> RAGConfig:
    """Get the RAG configuration with proper caching."""
    return load_config(RAGConfig, "rag")

def get_core_config() -> CoreConfig:
    """Get the core configuration with proper caching."""
    return load_config(CoreConfig, "core")

# Configuration validation utilities
def validate_environment() -> list:
    """Validate that required environment variables are set."""
    errors = []

    # Check for required API keys
    if not os.getenv("OPENROUTER_API_KEY"):
        errors.append("OPENROUTER_API_KEY environment variable is required")

    # Check for database URL if not using default
    db_url = os.getenv("AI_SCIENTIST_DATABASE_URL")
    if db_url and not db_url.startswith(('sqlite://', 'postgresql://', 'mysql://')):
        errors.append("Invalid AI_SCIENTIST_DATABASE_URL format")

    return errors

# Configuration migration utilities
def migrate_legacy_configs():
    """Migrate legacy configuration files to new unified format."""
    legacy_files = [
        "bfts_config.yaml",
        "enhanced_config.yaml",
        "rag_config.yaml"
    ]

    config_manager = get_config_manager()

    for legacy_file in legacy_files:
        legacy_path = Path.cwd() / legacy_file
        if legacy_path.exists():
            try:
                with open(legacy_path, 'r') as f:
                    legacy_data = yaml.safe_load(f)

                # Convert to new format and save
                if 'rag' in legacy_file:
                    new_config = RAGConfig(**legacy_data.get('rag', {}))
                    config_manager.save_config(new_config, 'rag')
                else:
                    new_config = CoreConfig(**legacy_data.get('core', {}))
                    config_manager.save_config(new_config, 'core')

                logger.info(f"Migrated {legacy_file} to new configuration format")

                # Optionally backup and remove legacy file
                backup_path = legacy_path.with_suffix('.yaml.backup')
                legacy_path.rename(backup_path)

            except Exception as e:
                logger.error(f"Failed to migrate {legacy_file}: {e}")

# Initialize configuration system on import
try:
    migrate_legacy_configs()
except Exception as e:
    logger.warning(f"Configuration migration failed: {e}")