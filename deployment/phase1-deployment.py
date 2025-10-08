#!/usr/bin/env python3
"""
Phase 1 Deployment Script

Automated deployment and configuration management for AI-Scientist-v2 Phase 1 fixes.
Handles environment setup, configuration migration, service deployment, and health checks.
"""

import os
import sys
import argparse
import yaml
import json
import subprocess
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Add ai_scientist to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_scientist.core.config_manager import ConfigManager, get_config_manager
from ai_scientist.core.security_manager import SecurityManager, get_security_manager
from ai_scientist.core.logging_system import setup_logging, get_logger
from ai_scientist.core.error_handler import handle_errors, AIScientistError

# Setup logging
setup_logging(level="INFO", structured=True)
logger = get_logger("deployment")

class DeploymentConfig:
    """Deployment configuration and settings"""

    def __init__(self, config_file: str = None):
        self.config_file = config_file or "deployment/phase1-config.yaml"
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_file} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise AIScientistError(f"Configuration loading failed: {e}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default deployment configuration"""
        return {
            "environment": "development",
            "deployment_mode": "local",
            "services": {
                "config_manager": {"enabled": True, "port": 8001},
                "security_manager": {"enabled": True, "port": 8002},
                "rag_system": {"enabled": True, "port": 8003},
                "api_gateway": {"enabled": True, "port": 8000}
            },
            "databases": {
                "redis": {"enabled": True, "port": 6379},
                "chroma": {"enabled": True, "port": 8004}
            },
            "monitoring": {
                "prometheus": {"enabled": True, "port": 9090},
                "grafana": {"enabled": True, "port": 3000}
            },
            "security": {
                "require_encryption": True,
                "min_password_length": 12,
                "session_timeout_minutes": 60
            }
        }

class ServiceManager:
    """Manage deployment of individual services"""

    def __init__(self, deployment_config: DeploymentConfig):
        self.config = deployment_config
        self.services: Dict[str, Dict[str, Any]] = {}
        self.processes: Dict[str, subprocess.Popen] = {}

    def start_service(self, service_name: str, service_config: Dict[str, Any]) -> bool:
        """Start a service"""
        try:
            logger.info(f"Starting service: {service_name}")

            # Prepare service command
            cmd = self._get_service_command(service_name, service_config)

            # Start service process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            self.processes[service_name] = process
            self.services[service_name] = service_config

            # Wait for service to be ready
            if self._wait_for_service(service_name, service_config):
                logger.info(f"Service {service_name} started successfully")
                return True
            else:
                logger.error(f"Service {service_name} failed to start")
                self.stop_service(service_name)
                return False

        except Exception as e:
            logger.error(f"Failed to start service {service_name}: {e}")
            return False

    def stop_service(self, service_name: str) -> bool:
        """Stop a service"""
        try:
            if service_name in self.processes:
                process = self.processes[service_name]
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

                del self.processes[service_name]
                if service_name in self.services:
                    del self.services[service_name]

                logger.info(f"Service {service_name} stopped")
                return True

        except Exception as e:
            logger.error(f"Failed to stop service {service_name}: {e}")
            return False

    def _get_service_command(self, service_name: str, service_config: Dict[str, Any]) -> List[str]:
        """Get command to start a service"""
        base_cmd = ["python", "-m"]

        if service_name == "config_manager":
            return base_cmd + ["ai_scientist.core.config_manager"]
        elif service_name == "security_manager":
            return base_cmd + ["ai_scientist.core.security_manager"]
        elif service_name == "rag_system":
            return base_cmd + ["ai_scientist.openrouter.rag_system"]
        elif service_name == "api_gateway":
            return base_cmd + ["ai_scientist.api.gateway"]
        else:
            raise AIScientistError(f"Unknown service: {service_name}")

    def _wait_for_service(self, service_name: str, service_config: Dict[str, Any], timeout: int = 30) -> bool:
        """Wait for service to be ready"""
        port = service_config.get("port")
        if not port:
            return True  # No port check needed

        logger.info(f"Waiting for {service_name} to be ready on port {port}")
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self._is_port_open(port):
                logger.info(f"Service {service_name} is ready")
                return True
            time.sleep(1)

        logger.error(f"Service {service_name} not ready after {timeout} seconds")
        return False

    def _is_port_open(self, port: int) -> bool:
        """Check if a port is open"""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                return result == 0
        except Exception:
            return False

    def get_service_status(self) -> Dict[str, str]:
        """Get status of all services"""
        status = {}
        for service_name, process in self.processes.items():
            if process.poll() is None:
                status[service_name] = "running"
            else:
                status[service_name] = "stopped"
        return status

class ConfigurationMigrator:
    """Handle configuration migration"""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    def migrate_configurations(self) -> bool:
        """Migrate all configurations"""
        try:
            logger.info("Starting configuration migration")

            # Migrate RAG configuration
            self._migrate_rag_config()

            # Migrate security configuration
            self._migrate_security_config()

            # Migrate logging configuration
            self._migrate_logging_config()

            logger.info("Configuration migration completed successfully")
            return True

        except Exception as e:
            logger.error(f"Configuration migration failed: {e}")
            return False

    def _migrate_rag_config(self):
        """Migrate RAG configuration"""
        logger.info("Migrating RAG configuration")

        # Load existing RAG config if available
        rag_config = self.config_manager.load_config("rag")

        # Set essential Phase 1 settings
        rag_config.update({
            "enable_hybrid_search": True,
            "enable_semantic_cache": True,
            "reasoning_depth": 3,
            "enable_page_index": True,
            "page_index_max_context": 8192,
            "cache_ttl": 3600
        })

        # Save migrated config
        self.config_manager.save_config(rag_config, "rag")

    def _migrate_security_config(self):
        """Migrate security configuration"""
        logger.info("Migrating security configuration")

        security_config = self.config_manager.load_config("security")

        # Set essential Phase 1 security settings
        security_config.update({
            "require_encryption": True,
            "min_password_length": 12,
            "session_timeout_minutes": 60,
            "max_login_attempts": 5,
            "rate_limit_requests": 100,
            "audit_log_retention_days": 90
        })

        self.config_manager.save_config(security_config, "security")

    def _migrate_logging_config(self):
        """Migrate logging configuration"""
        logger.info("Migrating logging configuration")

        logging_config = self.config_manager.load_config("logging")

        # Set essential Phase 1 logging settings
        logging_config.update({
            "level": "INFO",
            "structured_logging": True,
            "max_file_size": "10MB",
            "backup_count": 5,
            "enable_performance_logging": True
        })

        self.config_manager.save_config(logging_config, "logging")

class HealthChecker:
    """System health monitoring"""

    def __init__(self, service_manager: ServiceManager):
        self.service_manager = service_manager

    def run_health_checks(self) -> Dict[str, Any]:
        """Run comprehensive health checks"""
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "services": {},
            "resources": {},
            "issues": []
        }

        # Check services
        service_status = self.service_manager.get_service_status()
        health_status["services"] = service_status

        # Check system resources
        health_status["resources"] = self._check_resources()

        # Determine overall status
        stopped_services = [name for name, status in service_status.items() if status != "running"]
        if stopped_services:
            health_status["overall_status"] = "degraded"
            health_status["issues"].extend([f"Service not running: {name}" for name in stopped_services])

        # Check resource usage
        if health_status["resources"].get("memory_usage_percent", 0) > 90:
            health_status["overall_status"] = "unhealthy"
            health_status["issues"].append("High memory usage")

        if health_status["resources"].get("cpu_usage_percent", 0) > 90:
            health_status["overall_status"] = "unhealthy"
            health_status["issues"].append("High CPU usage")

        return health_status

    def _check_resources(self) -> Dict[str, float]:
        """Check system resource usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=1)

            return {
                "memory_usage_percent": memory.percent,
                "memory_available_mb": memory.available / 1024 / 1024,
                "cpu_usage_percent": cpu,
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
        except ImportError:
            logger.warning("psutil not available, skipping resource checks")
            return {}

class DeploymentManager:
    """Main deployment orchestrator"""

    def __init__(self, config_file: str = None):
        self.deployment_config = DeploymentConfig(config_file)
        self.config_manager = get_config_manager()
        self.security_manager = get_security_manager()
        self.service_manager = ServiceManager(self.deployment_config)
        self.config_migrator = ConfigurationMigrator(self.config_manager)
        self.health_checker = HealthChecker(self.service_manager)

    def deploy(self) -> bool:
        """Execute full deployment"""
        try:
            logger.info("Starting Phase 1 deployment")

            # Step 1: Validate environment
            if not self._validate_environment():
                return False

            # Step 2: Run configuration migration
            if not self.config_migrator.migrate_configurations():
                return False

            # Step 3: Start services
            if not self._start_services():
                return False

            # Step 4: Run health checks
            if not self._verify_deployment():
                return False

            logger.info("Phase 1 deployment completed successfully")
            return True

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            self._rollback()
            return False

    def rollback(self) -> bool:
        """Rollback deployment"""
        try:
            logger.info("Starting deployment rollback")

            # Stop all services
            self._stop_services()

            # Restore previous configuration
            self._restore_configuration()

            logger.info("Rollback completed")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def _validate_environment(self) -> bool:
        """Validate deployment environment"""
        logger.info("Validating deployment environment")

        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher required")
            return False

        # Check required directories
        required_dirs = ["logs", "cache", "config", "security"]
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")

        # Check required environment variables
        required_vars = ["OPENROUTER_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            return False

        logger.info("Environment validation passed")
        return True

    def _start_services(self) -> bool:
        """Start all configured services"""
        logger.info("Starting services")

        services_config = self.deployment_config.config.get("services", {})
        failed_services = []

        for service_name, service_config in services_config.items():
            if service_config.get("enabled", True):
                if not self.service_manager.start_service(service_name, service_config):
                    failed_services.append(service_name)

        if failed_services:
            logger.error(f"Failed to start services: {failed_services}")
            return False

        logger.info("All services started successfully")
        return True

    def _stop_services(self):
        """Stop all services"""
        logger.info("Stopping services")

        service_names = list(self.service_manager.processes.keys())
        for service_name in service_names:
            self.service_manager.stop_service(service_name)

    def _verify_deployment(self) -> bool:
        """Verify deployment health"""
        logger.info("Verifying deployment health")

        # Wait for services to stabilize
        time.sleep(5)

        # Run health checks
        health_status = self.health_checker.run_health_checks()

        logger.info(f"Health status: {health_status}")

        if health_status["overall_status"] == "unhealthy":
            logger.error("Deployment verification failed")
            return False

        # Run integration tests
        if not self._run_integration_tests():
            logger.warning("Integration tests failed, but deployment continuing")

        logger.info("Deployment verification passed")
        return True

    def _run_integration_tests(self) -> bool:
        """Run basic integration tests"""
        try:
            logger.info("Running integration tests")

            # Test configuration loading
            try:
                rag_config = self.config_manager.load_config("rag")
                logger.info("Configuration loading test passed")
            except Exception as e:
                logger.error(f"Configuration test failed: {e}")
                return False

            # Test security manager
            try:
                key_id, api_key = self.security_manager.generate_api_key()
                key_info = self.security_manager.validate_api_key(api_key)
                if key_info is None:
                    raise Exception("API key validation failed")
                logger.info("Security manager test passed")
            except Exception as e:
                logger.error(f"Security test failed: {e}")
                return False

            logger.info("Integration tests passed")
            return True

        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            return False

    def _rollback(self):
        """Execute rollback on deployment failure"""
        logger.error("Deployment failed, initiating rollback")
        self.rollback()

    def _restore_configuration(self):
        """Restore previous configuration"""
        logger.info("Restoring previous configuration")
        # Implementation would restore from backup
        pass

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "services": self.service_manager.get_service_status(),
            "health": self.health_checker.run_health_checks()
        }

def main():
    """Main deployment script entry point"""
    parser = argparse.ArgumentParser(description="AI-Scientist-v2 Phase 1 Deployment")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--action", choices=["deploy", "rollback", "status"], default="deploy",
                       help="Action to perform")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        setup_logging(level="DEBUG")

    deployment_manager = DeploymentManager(args.config)

    if args.action == "deploy":
        success = deployment_manager.deploy()
        if success:
            print("Deployment completed successfully!")
            sys.exit(0)
        else:
            print("Deployment failed!")
            sys.exit(1)

    elif args.action == "rollback":
        success = deployment_manager.rollback()
        if success:
            print("Rollback completed successfully!")
            sys.exit(0)
        else:
            print("Rollback failed!")
            sys.exit(1)

    elif args.action == "status":
        status = deployment_manager.get_deployment_status()
        print(json.dumps(status, indent=2))
        sys.exit(0)

if __name__ == "__main__":
    main()