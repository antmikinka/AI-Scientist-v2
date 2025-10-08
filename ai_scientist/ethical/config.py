"""
Ethical Framework Configuration

This module provides comprehensive configuration management for the Ethical Framework Agent,
including default settings, framework-specific configurations, and deployment parameters.
"""

import os
import yaml
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from .ethical_framework_agent import EthicalFrameworkType, DecisionMode, ComplianceStatus
from .integration import IntegrationMode, EthicalIntegrationLevel


class LogLevel(Enum):
    """Logging levels for ethical framework"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LearningConfig:
    """Configuration for adaptive learning"""
    enabled: bool = True
    learning_rate: float = 0.01
    batch_size: int = 32
    max_history_size: int = 10000
    feedback_integration: bool = True
    pattern_update_interval: int = 3600  # seconds
    constraint_adaptation_threshold: float = 0.8


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and compliance"""
    real_time_monitoring: bool = True
    monitoring_interval: int = 60  # seconds
    compliance_check_interval: int = 300  # seconds
    reporting_interval: int = 3600  # seconds
    audit_trail_enabled: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low_compliance_rate": 0.7,
        "high_violation_rate": 0.3,
        "ethical_score_alert": 0.5
    })


@dataclass
class HumanOversightConfig:
    """Configuration for human oversight"""
    enabled: bool = True
    oversight_threshold: float = 0.7
    escalation_threshold: float = 0.4
    response_timeout: int = 86400  # 24 hours
    automatic_escalation: bool = True
    multi_reviewer_required: bool = False
    stakeholder_notification: bool = True


@dataclass
class FrameworkWeightConfig:
    """Configuration for ethical framework weights"""
    utilitarian: float = 0.2
    deontological: float = 0.2
    virtue_ethics: float = 0.15
    care_ethics: float = 0.15
    principle_based: float = 0.2
    precautionary: float = 0.1
    cultural_relativist: float = 0.0  # Disabled by default


@dataclass
class SecurityConfig:
    """Security configuration for ethical framework"""
    encryption_enabled: bool = True
    audit_log_encryption: bool = True
    secure_data_storage: bool = True
    access_control_enabled: bool = True
    data_retention_period: int = 31536000  # 1 year in seconds
    anonymization_enabled: bool = True


@dataclass
class PerformanceConfig:
    """Performance configuration"""
    max_concurrent_assessments: int = 10
    assessment_timeout: int = 300  # seconds
    cache_enabled: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # seconds
    optimization_level: str = "balanced"  # "fast", "balanced", "thorough"


@dataclass
class CulturalConfig:
    """Configuration for cultural ethical considerations"""
    enabled: bool = True
    cultural_frameworks: List[str] = field(default_factory=lambda: [
        "western", "eastern", "african", "indigenous", "islamic", "confucian"
    ])
    cross_cultural_validation: bool = True
    cultural_adaptation_enabled: bool = True
    stakeholder_diversity_weight: float = 0.3


@dataclass
class EthicalFrameworkConfig:
    """Main configuration class for Ethical Framework Agent"""
    # Basic settings
    agent_name: str = "EthicalFrameworkAgent"
    version: str = "2.0.0"
    environment: str = "production"  # "development", "staging", "production"
    log_level: LogLevel = LogLevel.INFO

    # Framework settings
    enabled_frameworks: List[EthicalFrameworkType] = field(default_factory=lambda: [
        EthicalFrameworkType.UTILITARIAN,
        EthicalFrameworkType.DEONTOLOGICAL,
        EthicalFrameworkType.VIRTUE_ETHICS,
        EthicalFrameworkType.CARE_ETHICS,
        EthicalFrameworkType.PRINCIPLE_BASED,
        EthicalFrameworkType.PRECAUTIONARY
    ])
    framework_weights: FrameworkWeightConfig = field(default_factory=FrameworkWeightConfig)
    default_decision_mode: DecisionMode = DecisionMode.AUTONOMOUS

    # Thresholds and limits
    ethical_threshold: float = 0.8
    human_oversight_threshold: float = 0.7
    blocking_threshold: float = 0.4
    max_constraint_violations: int = 3
    confidence_threshold: float = 0.6

    # Sub-configurations
    learning: LearningConfig = field(default_factory=LearningConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    human_oversight: HumanOversightConfig = field(default_factory=HumanOversightConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    cultural: CulturalConfig = field(default_factory=CulturalConfig)

    # Integration settings
    integration_mode: IntegrationMode = IntegrationMode.COMPREHENSIVE
    integration_level: EthicalIntegrationLevel = EthicalIntegrationLevel.STANDARD
    real_time_integration: bool = True

    # Database and storage
    database_url: str = "sqlite:///ethical_framework.db"
    log_storage_path: str = "./logs/ethical_framework"
    audit_log_path: str = "./audit/ethical_framework"
    backup_enabled: bool = True
    backup_interval: int = 86400  # 24 hours

    # API and networking
    api_enabled: bool = True
    api_host: str = "localhost"
    api_port: int = 8081
    api_rate_limit: int = 100  # requests per minute

    # Development and testing
    test_mode: bool = False
    mock_assessments: bool = False
    debug_mode: bool = False

    # Compliance and reporting
    compliance_standards: List[str] = field(default_factory=lambda: [
        "belmont_report", "helsinki_declaration", "cioms_guidelines", "iso_9001"
    ])
    reporting_formats: List[str] = field(default_factory=lambda: [
        "json", "pdf", "html", "csv"
    ])
    automated_reporting: bool = True
    report_distribution: List[str] = field(default_factory=lambda: [
        "ethics_committee", "research_administration", "institutional_review_board"
    ])

    # Custom settings
    custom_constraints: List[Dict[str, Any]] = field(default_factory=list)
    custom_frameworks: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)


class ConfigManager:
    """
    Configuration Manager for Ethical Framework Agent

    Handles loading, validating, and managing configuration for the ethical framework,
    with support for environment-specific settings and runtime updates.
    """

    def __init__(self, config_path: str = None):
        self.config_path = config_path or "ethical_framework_config.yaml"
        self.config: EthicalFrameworkConfig = None
        self._load_config()

    def _load_config(self):
        """Load configuration from file and environment variables"""
        try:
            # Load from file if exists
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
            else:
                file_config = {}

            # Override with environment variables
            env_config = self._load_environment_config()

            # Merge configurations
            merged_config = self._merge_configs(file_config, env_config)

            # Create configuration object
            self.config = self._create_config_from_dict(merged_config)

            # Validate configuration
            self._validate_config()

        except Exception as e:
            raise Exception(f"Failed to load configuration: {e}")

    def _load_environment_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_mapping = {
            "ETHICAL_FRAMEWORK_ENVIRONMENT": "environment",
            "ETHICAL_FRAMEWORK_LOG_LEVEL": "log_level",
            "ETHICAL_FRAMEWORK_ETHICAL_THRESHOLD": "ethical_threshold",
            "ETHICAL_FRAMEWORK_HUMAN_OVERSIGHT_THRESHOLD": "human_oversight_threshold",
            "ETHICAL_FRAMEWORK_DATABASE_URL": "database_url",
            "ETHICAL_FRAMEWORK_API_ENABLED": "api_enabled",
            "ETHICAL_FRAMEWORK_API_HOST": "api_host",
            "ETHICAL_FRAMEWORK_API_PORT": "api_port",
            "ETHICAL_FRAMEWORK_TEST_MODE": "test_mode",
            "ETHICAL_FRAMEWORK_DEBUG_MODE": "debug_mode",
            "ETHICAL_FRAMEWORK_LEARNING_ENABLED": "learning.enabled",
            "ETHICAL_FRAMEWORK_MONITORING_ENABLED": "monitoring.real_time_monitoring",
            "ETHICAL_FRAMEWORK_HUMAN_OVERSIGHT_ENABLED": "human_oversight.enabled",
            "ETHICAL_FRAMEWORK_CULTURAL_ENABLED": "cultural.enabled"
        }

        env_config = {}
        for env_var, config_key in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]

                # Convert string values to appropriate types
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif '.' in value and value.replace('.', '').isdigit():
                    value = float(value)

                # Handle nested keys
                keys = config_key.split('.')
                current = env_config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = value

        return env_config

    def _merge_configs(self, file_config: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge file configuration with environment configuration"""
        def deep_merge(base, override):
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        return deep_merge(file_config, env_config)

    def _create_config_from_dict(self, config_dict: Dict[str, Any]) -> EthicalFrameworkConfig:
        """Create EthicalFrameworkConfig from dictionary"""
        # Convert string enums to enum values
        if 'log_level' in config_dict and isinstance(config_dict['log_level'], str):
            config_dict['log_level'] = LogLevel(config_dict['log_level'])

        if 'default_decision_mode' in config_dict and isinstance(config_dict['default_decision_mode'], str):
            config_dict['default_decision_mode'] = DecisionMode(config_dict['default_decision_mode'])

        if 'integration_mode' in config_dict and isinstance(config_dict['integration_mode'], str):
            config_dict['integration_mode'] = IntegrationMode(config_dict['integration_mode'])

        if 'integration_level' in config_dict and isinstance(config_dict['integration_level'], str):
            config_dict['integration_level'] = EthicalIntegrationLevel(config_dict['integration_level'])

        # Convert framework types
        if 'enabled_frameworks' in config_dict:
            enabled_frameworks = []
            for framework in config_dict['enabled_frameworks']:
                if isinstance(framework, str):
                    enabled_frameworks.append(EthicalFrameworkType(framework))
                else:
                    enabled_frameworks.append(framework)
            config_dict['enabled_frameworks'] = enabled_frameworks

        # Create nested config objects
        if 'framework_weights' not in config_dict:
            config_dict['framework_weights'] = {}

        config_dict['framework_weights'] = FrameworkWeightConfig(**config_dict['framework_weights'])
        config_dict['learning'] = LearningConfig(**config_dict.get('learning', {}))
        config_dict['monitoring'] = MonitoringConfig(**config_dict.get('monitoring', {}))
        config_dict['human_oversight'] = HumanOversightConfig(**config_dict.get('human_oversight', {}))
        config_dict['security'] = SecurityConfig(**config_dict.get('security', {}))
        config_dict['performance'] = PerformanceConfig(**config_dict.get('performance', {}))
        config_dict['cultural'] = CulturalConfig(**config_dict.get('cultural', {}))

        return EthicalFrameworkConfig(**config_dict)

    def _validate_config(self):
        """Validate configuration values"""
        errors = []

        # Validate threshold ranges
        if not 0 <= self.config.ethical_threshold <= 1:
            errors.append("ethical_threshold must be between 0 and 1")

        if not 0 <= self.config.human_oversight_threshold <= 1:
            errors.append("human_oversight_threshold must be between 0 and 1")

        if not 0 <= self.config.blocking_threshold <= 1:
            errors.append("blocking_threshold must be between 0 and 1")

        # Validate framework weights sum to 1
        weight_sum = sum([
            self.config.framework_weights.utilitarian,
            self.config.framework_weights.deontological,
            self.config.framework_weights.virtue_ethics,
            self.config.framework_weights.care_ethics,
            self.config.framework_weights.principle_based,
            self.config.framework_weights.precautionary,
            self.config.framework_weights.cultural_relativist
        ])

        if abs(weight_sum - 1.0) > 0.01:
            errors.append(f"Framework weights must sum to 1.0, current sum: {weight_sum}")

        # Validate file paths
        if self.config.log_storage_path:
            log_dir = os.path.dirname(self.config.log_storage_path)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create log directory {log_dir}: {e}")

        if self.config.audit_log_path:
            audit_dir = os.path.dirname(self.config.audit_log_path)
            if audit_dir and not os.path.exists(audit_dir):
                try:
                    os.makedirs(audit_dir, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create audit directory {audit_dir}: {e}")

        # Validate threshold relationships
        if self.config.blocking_threshold >= self.config.human_oversight_threshold:
            errors.append("blocking_threshold should be less than human_oversight_threshold")

        if self.config.human_oversight_threshold >= self.config.ethical_threshold:
            errors.append("human_oversight_threshold should be less than ethical_threshold")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    def get_config(self) -> EthicalFrameworkConfig:
        """Get current configuration"""
        return self.config

    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with runtime changes"""
        try:
            # Convert config to dict, update, and recreate
            config_dict = self._config_to_dict(self.config)
            config_dict = self._merge_configs(config_dict, updates)

            # Create new config object
            new_config = self._create_config_from_dict(config_dict)
            self._validate_config()

            self.config = new_config

        except Exception as e:
            raise Exception(f"Failed to update configuration: {e}")

    def _config_to_dict(self, config: EthicalFrameworkConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary"""
        return {
            "agent_name": config.agent_name,
            "version": config.version,
            "environment": config.environment,
            "log_level": config.log_level.value if config.log_level else None,
            "enabled_frameworks": [f.value for f in config.enabled_frameworks],
            "framework_weights": {
                "utilitarian": config.framework_weights.utilitarian,
                "deontological": config.framework_weights.deontological,
                "virtue_ethics": config.framework_weights.virtue_ethics,
                "care_ethics": config.framework_weights.care_ethics,
                "principle_based": config.framework_weights.principle_based,
                "precautionary": config.framework_weights.precautionary,
                "cultural_relativist": config.framework_weights.cultural_relativist
            },
            "default_decision_mode": config.default_decision_mode.value if config.default_decision_mode else None,
            "ethical_threshold": config.ethical_threshold,
            "human_oversight_threshold": config.human_oversight_threshold,
            "blocking_threshold": config.blocking_threshold,
            "max_constraint_violations": config.max_constraint_violations,
            "confidence_threshold": config.confidence_threshold,
            "learning": {
                "enabled": config.learning.enabled,
                "learning_rate": config.learning.learning_rate,
                "batch_size": config.learning.batch_size,
                "max_history_size": config.learning.max_history_size,
                "feedback_integration": config.learning.feedback_integration,
                "pattern_update_interval": config.learning.pattern_update_interval,
                "constraint_adaptation_threshold": config.learning.constraint_adaptation_threshold
            },
            "monitoring": {
                "real_time_monitoring": config.monitoring.real_time_monitoring,
                "monitoring_interval": config.monitoring.monitoring_interval,
                "compliance_check_interval": config.monitoring.compliance_check_interval,
                "reporting_interval": config.monitoring.reporting_interval,
                "audit_trail_enabled": config.monitoring.audit_trail_enabled,
                "alert_thresholds": config.monitoring.alert_thresholds
            },
            "human_oversight": {
                "enabled": config.human_oversight.enabled,
                "oversight_threshold": config.human_oversight.oversight_threshold,
                "escalation_threshold": config.human_oversight.escalation_threshold,
                "response_timeout": config.human_oversight.response_timeout,
                "automatic_escalation": config.human_oversight.automatic_escalation,
                "multi_reviewer_required": config.human_oversight.multi_reviewer_required,
                "stakeholder_notification": config.human_oversight.stakeholder_notification
            },
            "security": {
                "encryption_enabled": config.security.encryption_enabled,
                "audit_log_encryption": config.security.audit_log_encryption,
                "secure_data_storage": config.security.secure_data_storage,
                "access_control_enabled": config.security.access_control_enabled,
                "data_retention_period": config.security.data_retention_period,
                "anonymization_enabled": config.security.anonymization_enabled
            },
            "performance": {
                "max_concurrent_assessments": config.performance.max_concurrent_assessments,
                "assessment_timeout": config.performance.assessment_timeout,
                "cache_enabled": config.performance.cache_enabled,
                "cache_size": config.performance.cache_size,
                "cache_ttl": config.performance.cache_ttl,
                "optimization_level": config.performance.optimization_level
            },
            "cultural": {
                "enabled": config.cultural.enabled,
                "cultural_frameworks": config.cultural.cultural_frameworks,
                "cross_cultural_validation": config.cultural.cross_cultural_validation,
                "cultural_adaptation_enabled": config.cultural.cultural_adaptation_enabled,
                "stakeholder_diversity_weight": config.cultural.stakeholder_diversity_weight
            },
            "integration_mode": config.integration_mode.value if config.integration_mode else None,
            "integration_level": config.integration_level.value if config.integration_level else None,
            "real_time_integration": config.real_time_integration,
            "database_url": config.database_url,
            "log_storage_path": config.log_storage_path,
            "audit_log_path": config.audit_log_path,
            "backup_enabled": config.backup_enabled,
            "backup_interval": config.backup_interval,
            "api_enabled": config.api_enabled,
            "api_host": config.api_host,
            "api_port": config.api_port,
            "api_rate_limit": config.api_rate_limit,
            "test_mode": config.test_mode,
            "mock_assessments": config.mock_assessments,
            "debug_mode": config.debug_mode,
            "compliance_standards": config.compliance_standards,
            "reporting_formats": config.reporting_formats,
            "automated_reporting": config.automated_reporting,
            "report_distribution": config.report_distribution,
            "custom_constraints": config.custom_constraints,
            "custom_frameworks": config.custom_frameworks,
            "environment_variables": config.environment_variables
        }

    def save_config(self, path: str = None):
        """Save current configuration to file"""
        save_path = path or self.config_path

        try:
            config_dict = self._config_to_dict(self.config)

            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        except Exception as e:
            raise Exception(f"Failed to save configuration: {e}")

    def get_environment_config(self) -> Dict[str, str]:
        """Get environment-specific configuration"""
        env_configs = {
            "development": {
                "test_mode": True,
                "debug_mode": True,
                "log_level": LogLevel.DEBUG,
                "monitoring": {"real_time_monitoring": True, "monitoring_interval": 30}
            },
            "staging": {
                "test_mode": False,
                "debug_mode": True,
                "log_level": LogLevel.INFO,
                "ethical_threshold": 0.75,
                "human_oversight_threshold": 0.6
            },
            "production": {
                "test_mode": False,
                "debug_mode": False,
                "log_level": LogLevel.WARNING,
                "ethical_threshold": 0.8,
                "human_oversight_threshold": 0.7,
                "security": {"encryption_enabled": True, "access_control_enabled": True}
            }
        }

        return env_configs.get(self.config.environment, {})

    def create_default_config_file(self, path: str = None):
        """Create a default configuration file"""
        default_path = path or "ethical_framework_config_default.yaml"

        default_config = EthicalFrameworkConfig()

        try:
            config_dict = self._config_to_dict(default_config)

            with open(default_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            return default_path

        except Exception as e:
            raise Exception(f"Failed to create default configuration file: {e}")

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        return {
            "agent_name": self.config.agent_name,
            "version": self.config.version,
            "environment": self.config.environment,
            "log_level": self.config.log_level.value if self.config.log_level else None,
            "enabled_frameworks": len(self.config.enabled_frameworks),
            "ethical_threshold": self.config.ethical_threshold,
            "human_oversight_threshold": self.config.human_oversight_threshold,
            "integration_mode": self.config.integration_mode.value if self.config.integration_mode else None,
            "integration_level": self.config.integration_level.value if self.config.integration_level else None,
            "learning_enabled": self.config.learning.enabled,
            "monitoring_enabled": self.config.monitoring.real_time_monitoring,
            "human_oversight_enabled": self.config.human_oversight.enabled,
            "cultural_considerations_enabled": self.config.cultural.enabled,
            "api_enabled": self.config.api_enabled,
            "test_mode": self.config.test_mode
        }


# Global configuration instance
_config_manager = None

def get_config_manager(config_path: str = None) -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager

def get_config() -> EthicalFrameworkConfig:
    """Get current configuration"""
    return get_config_manager().get_config()

def update_config(updates: Dict[str, Any]):
    """Update global configuration"""
    get_config_manager().update_config(updates)