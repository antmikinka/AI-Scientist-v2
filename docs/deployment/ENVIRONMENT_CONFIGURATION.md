# Environment Configuration Guide

## Overview

This guide details the complete environment configuration required for deploying and running the AI-Scientist-v2 multi-agent system in various environments including development, staging, and production.

## Table of Contents

1. [Environment Types](#environment-types)
2. [Configuration Management](#configuration-management)
3. [Development Environment](#development-environment)
4. [Staging Environment](#staging-environment)
5. [Production Environment](#production-environment)
6. [Environment Variables](#environment-variables)
7. [Configuration Files](#configuration-files)
8. [Security Configuration](#security-configuration)
9. [Database Configuration](#database-configuration)
10. [Service Configuration](#service-configuration)

## Environment Types

### Development Environment
- **Purpose**: Local development and testing
- **Resources**: Minimal resource allocation
- **Security**: Basic security measures
- **Monitoring**: Local logging and debugging
- **Data**: Mock/synthetic data

### Staging Environment
- **Purpose**: Pre-production testing and validation
- **Resources**: Production-like resource allocation
- **Security**: Production-grade security configuration
- **Monitoring**: Full monitoring and alerting
- **Data**: Anonymized production data

### Production Environment
- **Purpose**: Live production deployment
- **Resources**: Optimized resource allocation
- **Security**: Maximum security configuration
- **Monitoring**: Comprehensive monitoring and alerting
- **Data**: Real user data with full backup

## Configuration Management

### Configuration File Structure

```
ai_scientist/
├── config/
│   ├── __init__.py
│   ├── base.yaml              # Base configuration
│   ├── development.yaml       # Development overrides
│   ├── staging.yaml          # Staging overrides
│   ├── production.yaml       # Production overrides
│   ├── local.yaml            # Local overrides (gitignored)
│   └── secrets.yaml          # Secrets (gitignored)
├── core/
│   ├── config_manager.py     # Configuration manager
│   └── environment.py        # Environment utilities
└── scripts/
    ├── setup_environment.py  # Environment setup script
    └── validate_config.py    # Configuration validation
```

### Configuration Manager

```python
# ai_scientist/core/config_manager.py
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

class ConfigManager:
    """Centralized configuration management system"""

    def __init__(self, env: Optional[str] = None):
        self.env = env or os.getenv('ENVIRONMENT', 'development')
        self.config_path = Path(__file__).parent.parent / 'config'
        self.config = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML files"""
        # Load base configuration
        base_config = self._load_yaml_file('base.yaml')
        self.config.update(base_config)

        # Load environment-specific configuration
        env_config = self._load_yaml_file(f'{self.env}.yaml')
        self._deep_merge(self.config, env_config)

        # Load local configuration (if exists)
        local_config = self._load_yaml_file('local.yaml', required=False)
        if local_config:
            self._deep_merge(self.config, local_config)

        # Load secrets (if exists)
        secrets = self._load_yaml_file('secrets.yaml', required=False)
        if secrets:
            self._deep_merge(self.config, secrets)

        # Override with environment variables
        self._load_env_overrides()

    def _load_yaml_file(self, filename: str, required: bool = True) -> Dict[str, Any]:
        """Load YAML configuration file"""
        file_path = self.config_path / filename
        if not file_path.exists():
            if required:
                raise FileNotFoundError(f"Required configuration file not found: {file_path}")
            return {}

        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise ValueError(f"Error loading configuration file {filename}: {e}")

    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge two dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _load_env_overrides(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'DATABASE_URL': ['database', 'url'],
            'REDIS_URL': ['redis', 'url'],
            'OPENROUTER_API_KEY': ['openrouter', 'api_key'],
            'JWT_SECRET': ['security', 'jwt_secret'],
            'LOG_LEVEL': ['logging', 'level'],
            'ENVIRONMENT': ['environment', 'name']
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested_value(self.config, config_path, value)

    def _set_nested_value(self, config: Dict, path: list, value: Any):
        """Set nested configuration value"""
        for key in path[:-1]:
            config = config.setdefault(key, {})
        config[path[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.get('database', {})

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        return self.get('redis', {})

    def get_openrouter_config(self) -> Dict[str, Any]:
        """Get OpenRouter configuration"""
        return self.get('openrouter', {})

    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return self.get('security', {})

    def validate(self) -> bool:
        """Validate configuration"""
        required_keys = [
            'database.url',
            'redis.url',
            'openrouter.api_key',
            'security.jwt_secret'
        ]

        for key in required_keys:
            if not self.get(key):
                logging.error(f"Required configuration key missing: {key}")
                return False

        return True

# Global configuration instance
config = ConfigManager()
```

## Development Environment

### Development Configuration

```yaml
# config/development.yaml
environment:
  name: development
  debug: true
  log_level: DEBUG

database:
  url: postgresql://postgres:password@localhost:5432/ai_scientist_dev
  echo: true
  pool_size: 5
  max_overflow: 10

redis:
  url: redis://localhost:6379/0
  decode_responses: true
  max_connections: 10

openrouter:
  api_key: ${OPENROUTER_API_KEY}
  base_url: https://openrouter.ai/api/v1
  timeout: 30
  max_retries: 3

security:
  jwt_secret: ${JWT_SECRET}
  jwt_algorithm: HS256
  jwt_expiration: 86400
  password_hash_rounds: 12

logging:
  level: DEBUG
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/development.log
  max_size: 10MB
  backup_count: 5

api:
  host: 0.0.0.0
  port: 8080
  workers: 1
  reload: true

agents:
  max_concurrent: 5
  timeout: 300
  retry_attempts: 2

monitoring:
  enabled: true
  metrics_port: 9090
  health_check_interval: 30

features:
  debug_mode: true
  mock_external_apis: false
  enable_profiling: true
  enable_query_debug: true
```

### Development Setup Script

```python
# scripts/setup_environment.py
import os
import subprocess
import sys
from pathlib import Path

def setup_development_environment():
    """Setup development environment"""
    print("Setting up development environment...")

    # Create virtual environment
    if not Path("venv").exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])

    # Activate virtual environment and install dependencies
    if sys.platform == "win32":
        pip_path = "venv/Scripts/pip"
        python_path = "venv/Scripts/python"
    else:
        pip_path = "venv/bin/pip"
        python_path = "venv/bin/python"

    print("Installing dependencies...")
    subprocess.run([pip_path, "install", "-r", "requirements.txt"])
    subprocess.run([pip_path, "install", "-r", "requirements_openrouter.txt"])
    subprocess.run([pip_path, "install", "-r", "requirements_phase1.txt"])
    subprocess.run([pip_path, "install", "-r", "requirements_dev.txt"])

    # Setup environment file
    if not Path(".env").exists():
        print("Creating .env file...")
        env_content = """
# Development Environment Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/ai_scientist_dev

# Redis
REDIS_URL=redis://localhost:6379/0

# OpenRouter
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Security
JWT_SECRET=your_development_jwt_secret_at_least_32_characters

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
"""
        with open(".env", "w") as f:
            f.write(env_content.strip())

    # Create necessary directories
    directories = ["logs", "data", "cache", "uploads"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

    # Setup development database
    print("Setting up development database...")
    setup_result = subprocess.run([
        python_path, "-m", "ai_scientist.database.migrations", "upgrade", "head"
    ], capture_output=True, text=True)

    if setup_result.returncode != 0:
        print(f"Database setup failed: {setup_result.stderr}")
        return False

    print("Development environment setup completed successfully!")
    print("\nNext steps:")
    print("1. Update .env file with your OpenRouter API key")
    print("2. Run: python -m ai_scientist.web.app")
    print("3. Visit: http://localhost:8080")

    return True

if __name__ == "__main__":
    if setup_development_environment():
        sys.exit(0)
    else:
        sys.exit(1)
```

## Staging Environment

### Staging Configuration

```yaml
# config/staging.yaml
environment:
  name: staging
  debug: false
  log_level: INFO

database:
  url: ${STAGING_DATABASE_URL}
  echo: false
  pool_size: 10
  max_overflow: 20
  ssl_mode: require

redis:
  url: ${STAGING_REDIS_URL}
  decode_responses: true
  max_connections: 20
  ssl: true

openrouter:
  api_key: ${STAGING_OPENROUTER_API_KEY}
  base_url: https://openrouter.ai/api/v1
  timeout: 60
  max_retries: 5

security:
  jwt_secret: ${STAGING_JWT_SECRET}
  jwt_algorithm: HS256
  jwt_expiration: 3600
  password_hash_rounds: 14
  cors_origins: ["https://staging.your-domain.com"]

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/staging.log
  max_size: 50MB
  backup_count: 10
  external_logging: true
  sentry_dsn: ${STAGING_SENTRY_DSN}

api:
  host: 0.0.0.0
  port: 8080
  workers: 4
  reload: false

agents:
  max_concurrent: 20
  timeout: 600
  retry_attempts: 3

monitoring:
  enabled: true
  metrics_port: 9090
  health_check_interval: 60
  alert_thresholds:
    error_rate: 0.05
    response_time: 2000
    memory_usage: 0.8

features:
  debug_mode: false
  mock_external_apis: false
  enable_profiling: false
  enable_query_debug: false
  enable_audit_logging: true
```

### Staging Deployment Script

```bash
#!/bin/bash
# scripts/deploy_staging.sh

set -e

echo "Deploying to staging environment..."

# Set environment
export ENVIRONMENT=staging

# Pull latest code
git pull origin main

# Build and deploy
docker-compose -f docker-compose.staging.yml down
docker-compose -f docker-compose.staging.yml build --no-cache
docker-compose -f docker-compose.staging.yml up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 60

# Run health checks
echo "Running health checks..."
python scripts/health_checks.py --environment staging

# Run integration tests
echo "Running integration tests..."
python -m pytest tests/integration/ -v

# Run smoke tests
echo "Running smoke tests..."
python scripts/smoke_tests.py --environment staging

echo "Staging deployment completed successfully!"
```

## Production Environment

### Production Configuration

```yaml
# config/production.yaml
environment:
  name: production
  debug: false
  log_level: WARNING

database:
  url: ${PRODUCTION_DATABASE_URL}
  echo: false
  pool_size: 20
  max_overflow: 40
  ssl_mode: require
  connect_timeout: 10
  command_timeout: 30

redis:
  url: ${PRODUCTION_REDIS_URL}
  decode_responses: true
  max_connections: 50
  ssl: true
  socket_timeout: 5
  socket_connect_timeout: 5

openrouter:
  api_key: ${PRODUCTION_OPENROUTER_API_KEY}
  base_url: https://openrouter.ai/api/v1
  timeout: 120
  max_retries: 5
  rate_limit: 100

security:
  jwt_secret: ${PRODUCTION_JWT_SECRET}
  jwt_algorithm: HS256
  jwt_expiration: 1800
  password_hash_rounds: 16
  cors_origins: ["https://your-domain.com"]
  rate_limiting: true
  max_requests_per_minute: 60

logging:
  level: WARNING
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/production.log
  max_size: 100MB
  backup_count: 20
  external_logging: true
  sentry_dsn: ${PRODUCTION_SENTRY_DSN}
  loggly_token: ${PRODUCTION_LOGGLY_TOKEN}

api:
  host: 0.0.0.0
  port: 8080
  workers: 8
  reload: false
  access_log: true

agents:
  max_concurrent: 50
  timeout: 1200
  retry_attempts: 5
  resource_limits:
    memory: 8192  # MB
    cpu: 4.0

monitoring:
  enabled: true
  metrics_port: 9090
  health_check_interval: 30
  alert_thresholds:
    error_rate: 0.01
    response_time: 1000
    memory_usage: 0.7
    cpu_usage: 0.8

backup:
  enabled: true
  schedule: "0 2 * * *"  # Daily at 2 AM
  retention_days: 30
  s3_bucket: ${PRODUCTION_BACKUP_BUCKET}

features:
  debug_mode: false
  mock_external_apis: false
  enable_profiling: false
  enable_query_debug: false
  enable_audit_logging: true
  enable_performance_monitoring: true
  enable_security_monitoring: true
```

### Production Deployment Script

```bash
#!/bin/bash
# scripts/deploy_production.sh

set -e

echo "Deploying to production environment..."

# Safety checks
if [ "$ENVIRONMENT" != "production" ]; then
    echo "Error: ENVIRONMENT must be set to 'production'"
    exit 1
fi

# Backup current deployment
echo "Creating backup..."
./scripts/backup_production.sh

# Run pre-deployment checks
echo "Running pre-deployment checks..."
python scripts/pre_deployment_checks.py

# Blue-green deployment
echo "Starting blue-green deployment..."

# Deploy to green environment
docker-compose -f docker-compose.prod.yml -p ai-scientist-green down
docker-compose -f docker-compose.prod.yml -p ai-scientist-green pull
docker-compose -f docker-compose.prod.yml -p ai-scientist-green up -d

# Wait for green environment to be ready
echo "Waiting for green environment..."
sleep 120

# Health checks on green
echo "Running health checks on green environment..."
python scripts/health_checks.py --environment production --port 8081

# Run smoke tests on green
echo "Running smoke tests on green environment..."
python scripts/smoke_tests.py --environment production --port 8081

# Switch traffic to green
echo "Switching traffic to green environment..."
./scripts/switch_traffic.sh green

# Wait and monitor
echo "Monitoring new deployment..."
sleep 300

# Final health checks
echo "Running final health checks..."
python scripts/health_checks.py --environment production

if [ $? -eq 0 ]; then
    echo "Production deployment completed successfully!"

    # Clean up blue environment
    docker-compose -f docker-compose.prod.yml -p ai-scientist-blue down
else
    echo "Production deployment failed! Rolling back..."
    ./scripts/switch_traffic.sh blue
    docker-compose -f docker-compose.prod.yml -p ai-scientist-green down
    exit 1
fi
```

## Environment Variables

### Required Environment Variables

```bash
# Environment Configuration
ENVIRONMENT=development|staging|production
DEBUG=true|false
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR

# Database Configuration
DATABASE_URL=postgresql://user:password@host:port/database
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Redis Configuration
REDIS_URL=redis://host:port/database
REDIS_MAX_CONNECTIONS=20

# OpenRouter Configuration
OPENROUTER_API_KEY=your_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_TIMEOUT=30

# Security Configuration
JWT_SECRET=your_jwt_secret_at_least_32_characters
JWT_ALGORITHM=HS256
JWT_EXPIRATION=86400
ENCRYPTION_KEY=your_32_character_encryption_key

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
API_WORKERS=4
CORS_ORIGINS=https://your-domain.com

# Monitoring Configuration
SENTRY_DSN=your_sentry_dsn
PROMETHEUS_PORT=9090
HEALTH_CHECK_INTERVAL=30

# Backup Configuration
BACKUP_S3_BUCKET=your_backup_bucket
BACKUP_SCHEDULE=0 2 * * *
BACKUP_RETENTION_DAYS=30

# Feature Flags
ENABLE_METRICS_COLLECTION=true
ENABLE_AUDIT_LOGGING=true
ENABLE_PERFORMANCE_MONITORING=true
ENABLE_SECURITY_MONITORING=true
```

### Environment-specific Variable Files

```bash
# .env.development
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://postgres:password@localhost:5432/ai_scientist_dev
REDIS_URL=redis://localhost:6379/0
API_PORT=8080

# .env.staging
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=postgresql://user:password@staging-db:5432/ai_scientist_staging
REDIS_URL=redis://staging-redis:6379/1
API_PORT=8080

# .env.production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
DATABASE_URL=postgresql://user:password@prod-db:5432/ai_scientist_prod
REDIS_URL=redis://prod-redis:6379/2
API_PORT=8080
```

## Security Configuration

### Security Configuration Management

```python
# ai_scientist/core/security_config.py
import os
import hashlib
import secrets
from cryptography.fernet import Fernet
from typing import Dict, Any, Optional

class SecurityConfig:
    """Security configuration management"""

    def __init__(self, config_manager):
        self.config = config_manager
        self._init_encryption()

    def _init_encryption(self):
        """Initialize encryption"""
        encryption_key = self.config.get('security.encryption_key')
        if not encryption_key:
            # Generate encryption key if not exists
            encryption_key = Fernet.generate_key().decode()
            # Save to secure storage
            self._save_encryption_key(encryption_key)

        self.cipher_suite = Fernet(encryption_key.encode())

    def _save_encryption_key(self, key: str):
        """Save encryption key to secure storage"""
        # Implementation depends on your secret management system
        # This could be AWS Secrets Manager, HashiCorp Vault, etc.
        pass

    def encrypt_value(self, value: str) -> str:
        """Encrypt configuration value"""
        return self.cipher_suite.encrypt(value.encode()).decode()

    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt configuration value"""
        return self.cipher_suite.decrypt(encrypted_value.encode()).decode()

    def validate_security_config(self) -> bool:
        """Validate security configuration"""
        required_security_keys = [
            'security.jwt_secret',
            'security.encryption_key',
            'security.password_hash_rounds'
        ]

        for key in required_security_keys:
            if not self.config.get(key):
                return False

        return True

    def get_secure_config(self, key: str) -> Optional[str]:
        """Get and decrypt secure configuration value"""
        encrypted_value = self.config.get(key)
        if encrypted_value and encrypted_value.startswith('encrypted:'):
            return self.decrypt_value(encrypted_value[10:])
        return encrypted_value
```

## Service Configuration

### Service Discovery Configuration

```yaml
# config/services.yaml
services:
  api_gateway:
    host: api-gateway
    port: 8081
    health_check: /health
    timeout: 30
    retries: 3

  web_app:
    host: web-app
    port: 8080
    health_check: /health
    timeout: 30
    retries: 3

  agent_service:
    host: agent-service
    port: 8082
    health_check: /health
    timeout: 60
    retries: 5

  research_orchestrator:
    host: research-orchestrator
    port: 8083
    health_check: /health
    timeout: 120
    retries: 3

  ethical_framework:
    host: ethical-framework
    port: 8084
    health_check: /health
    timeout: 30
    retries: 3

load_balancer:
  algorithm: round_robin
  health_check_interval: 30
  unhealthy_threshold: 3
  healthy_threshold: 2
```

### Performance Configuration

```yaml
# config/performance.yaml
performance:
  caching:
    enabled: true
    backend: redis
    ttl: 3600
    max_size: 1000

  connection_pooling:
    database:
      min_connections: 5
      max_connections: 20
      connection_timeout: 30
    redis:
      min_connections: 10
      max_connections: 50
      connection_timeout: 10

  rate_limiting:
    enabled: true
    requests_per_minute: 60
    burst_size: 10
    storage: redis

  timeouts:
    api_request: 30
    agent_execution: 600
    database_query: 10
    redis_operation: 5

  memory_management:
    max_memory_usage: 8192  # MB
    gc_threshold: 0.8
    memory_check_interval: 60
```

## Configuration Validation

### Configuration Validator

```python
# scripts/validate_config.py
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List

class ConfigurationValidator:
    """Configuration validation utility"""

    def __init__(self, environment: str):
        self.environment = environment
        self.config_path = Path(__file__).parent.parent / 'config'
        self.errors = []
        self.warnings = []

    def validate_all(self) -> bool:
        """Validate all configuration aspects"""
        print(f"Validating configuration for environment: {self.environment}")

        # Validate required files exist
        self._validate_required_files()

        # Load and validate configuration structure
        config = self._load_configuration()
        if config:
            self._validate_configuration_structure(config)
            self._validate_required_keys(config)
            self._validate_data_types(config)
            self._validate_value_ranges(config)
            self._validate_security_settings(config)

        # Print results
        self._print_results()

        return len(self.errors) == 0

    def _validate_required_files(self):
        """Validate required configuration files exist"""
        required_files = [
            f'{self.environment}.yaml',
            'base.yaml'
        ]

        for file in required_files:
            file_path = self.config_path / file
            if not file_path.exists():
                self.errors.append(f"Required configuration file missing: {file}")

    def _load_configuration(self) -> Dict[str, Any]:
        """Load and merge configuration"""
        try:
            # Load base configuration
            with open(self.config_path / 'base.yaml', 'r') as f:
                config = yaml.safe_load(f) or {}

            # Load environment configuration
            with open(self.config_path / f'{self.environment}.yaml', 'r') as f:
                env_config = yaml.safe_load(f) or {}

            # Merge configurations
            self._deep_merge(config, env_config)
            return config

        except Exception as e:
            self.errors.append(f"Error loading configuration: {e}")
            return {}

    def _validate_configuration_structure(self, config: Dict[str, Any]):
        """Validate configuration structure"""
        required_sections = [
            'environment',
            'database',
            'redis',
            'openrouter',
            'security',
            'logging'
        ]

        for section in required_sections:
            if section not in config:
                self.errors.append(f"Required configuration section missing: {section}")

    def _validate_required_keys(self, config: Dict[str, Any]):
        """Validate required configuration keys"""
        required_keys = {
            'environment.name': str,
            'database.url': str,
            'redis.url': str,
            'openrouter.api_key': str,
            'security.jwt_secret': str,
            'api.host': str,
            'api.port': int
        }

        for key, expected_type in required_keys.items():
            value = self._get_nested_value(config, key.split('.'))
            if value is None:
                self.errors.append(f"Required configuration key missing: {key}")
            elif not isinstance(value, expected_type):
                self.errors.append(f"Invalid type for {key}: expected {expected_type.__name__}, got {type(value).__name__}")

    def _validate_data_types(self, config: Dict[str, Any]):
        """Validate data types"""
        type_validations = {
            'api.workers': (int, lambda x: x > 0),
            'agents.max_concurrent': (int, lambda x: x > 0),
            'security.jwt_expiration': (int, lambda x: x > 0),
            'database.pool_size': (int, lambda x: x > 0)
        }

        for key, (expected_type, validator) in type_validations.items():
            value = self._get_nested_value(config, key.split('.'))
            if value is not None:
                if not isinstance(value, expected_type):
                    self.errors.append(f"Invalid type for {key}: expected {expected_type.__name__}")
                elif not validator(value):
                    self.errors.append(f"Invalid value for {key}: {value}")

    def _validate_value_ranges(self, config: Dict[str, Any]):
        """Validate value ranges"""
        range_validations = {
            'api.port': (1, 65535),
            'security.jwt_expiration': (300, 86400),  # 5 minutes to 24 hours
            'agents.timeout': (60, 3600),  # 1 minute to 1 hour
        }

        for key, (min_val, max_val) in range_validations.items():
            value = self._get_nested_value(config, key.split('.'))
            if value is not None and not (min_val <= value <= max_val):
                self.errors.append(f"Value for {key} out of range: {value} (expected {min_val}-{max_val})")

    def _validate_security_settings(self, config: Dict[str, Any]):
        """Validate security settings"""
        security_config = config.get('security', {})

        # JWT secret length
        jwt_secret = security_config.get('jwt_secret', '')
        if len(jwt_secret) < 32:
            self.errors.append("JWT secret must be at least 32 characters long")

        # Password hash rounds
        hash_rounds = security_config.get('password_hash_rounds', 12)
        if not (10 <= hash_rounds <= 20):
            self.warnings.append("Password hash rounds should be between 10 and 20")

        # CORS configuration in production
        if self.environment == 'production':
            cors_origins = security_config.get('cors_origins', [])
            if not cors_origins or cors_origins == ['*']:
                self.errors.append("CORS origins must be explicitly set in production")

    def _get_nested_value(self, config: Dict, keys: List[str]):
        """Get nested value from dictionary"""
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _print_results(self):
        """Print validation results"""
        print("\n" + "="*50)
        print("CONFIGURATION VALIDATION RESULTS")
        print("="*50)

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ All configuration validations passed!")
        elif not self.errors:
            print("\n✅ Configuration valid (with warnings)")

if __name__ == "__main__":
    environment = sys.argv[1] if len(sys.argv) > 1 else 'development'
    validator = ConfigurationValidator(environment)
    success = validator.validate_all()
    sys.exit(0 if success else 1)
```

## Usage Examples

### Loading Configuration in Application

```python
# ai_scientist/main.py
from ai_scientist.core.config_manager import config

def main():
    # Load configuration
    if not config.validate():
        print("Configuration validation failed!")
        return

    # Use configuration
    database_config = config.get_database_config()
    redis_config = config.get_redis_config()
    security_config = config.get_security_config()

    print(f"Starting application in {config.get('environment.name')} mode")
    print(f"Database: {database_config['url']}")
    print(f"Redis: {redis_config['url']}")

    # Start application
    start_application()

if __name__ == "__main__":
    main()
```

### Environment-specific Startup

```bash
# Development
ENVIRONMENT=development python -m ai_scientist.main

# Staging
ENVIRONMENT=staging python -m ai_scientist.main

# Production
ENVIRONMENT=production python -m ai_scientist.main
```

This comprehensive environment configuration guide provides a robust foundation for managing different deployment environments with proper security, validation, and operational procedures.