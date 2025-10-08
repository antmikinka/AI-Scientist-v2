"""
Core Infrastructure Module

Provides unified configuration, logging, and error handling
for the AI-Scientist-v2 system.
"""

from .config_manager import (
    ConfigManager,
    CoreConfig,
    RAGConfig,
    DatabaseConfig,
    SecurityConfig,
    LoggingConfig,
    PerformanceConfig,
    get_config_manager,
    load_config,
    get_rag_config,
    get_core_config,
    validate_environment,
    migrate_legacy_configs
)

from .error_handler import (
    AIScientistError,
    ConfigurationError,
    APIError,
    SecurityError,
    ValidationError,
    handle_errors,
    retry_with_backoff,
    circuit_breaker
)

from .logging_system import (
    setup_logging,
    get_logger,
    StructuredLogger,
    LogFormatter
)

__all__ = [
    # Configuration
    'ConfigManager',
    'CoreConfig',
    'RAGConfig',
    'DatabaseConfig',
    'SecurityConfig',
    'LoggingConfig',
    'PerformanceConfig',
    'get_config_manager',
    'load_config',
    'get_rag_config',
    'get_core_config',
    'validate_environment',
    'migrate_legacy_configs',

    # Error Handling
    'AIScientistError',
    'ConfigurationError',
    'APIError',
    'SecurityError',
    'ValidationError',
    'handle_errors',
    'retry_with_backoff',
    'circuit_breaker',

    # Logging
    'setup_logging',
    'get_logger',
    'StructuredLogger',
    'LogFormatter'
]