"""
Core Integration Tests for Phase 1 Critical Fixes

Comprehensive test suite covering configuration management,
security, error handling, logging, and performance optimization.
"""

import os
import sys
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import pytest
import json
import yaml

# Add the ai_scientist directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_scientist.core.config_manager import (
    ConfigManager, CoreConfig, RAGConfig, get_config_manager,
    load_config, get_rag_config, get_core_config, validate_environment
)

from ai_scientist.core.security_manager import (
    SecurityManager, SecurityPolicy, get_security_manager,
    validate_api_key_format, sanitize_sensitive_data
)

from ai_scientist.core.error_handler import (
    AIScientistError, ConfigurationError, APIError, SecurityError,
    ValidationError, handle_errors, retry_with_backoff, circuit_breaker,
    get_error_metrics, get_error_summary
)

from ai_scientist.core.logging_system import (
    setup_logging, get_logger, LogContext, LogLevel,
    PerformanceMonitor, HealthChecker, get_performance_monitor,
    get_health_checker, performance_monitor
)

from ai_scientist.core.performance_optimizer import (
    PerformanceOptimizer, CacheConfig, PerformanceConfig,
    get_performance_optimizer, cache_result
)

class TestConfigurationManager:
    """Test suite for configuration management"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_manager = ConfigManager(self.temp_dir / "config")

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_default_config(self):
        """Test loading default configuration"""
        config = self.config_manager.load_config(CoreConfig)
        assert isinstance(config, CoreConfig)
        assert config.environment == "development"
        assert config.debug is False

    def test_load_rag_config(self):
        """Test loading RAG configuration"""
        config = self.config_manager.load_config(RAGConfig, "rag")
        assert isinstance(config, RAGConfig)
        assert config.enabled is True
        assert config.chunk_size == 1000
        assert config.similarity_threshold == 0.7

    def test_save_and_load_config(self):
        """Test saving and loading configuration"""
        original_config = CoreConfig(
            environment="production",
            debug=True,
            log_level="DEBUG"
        )

        self.config_manager.save_config(original_config, "test")
        loaded_config = self.config_manager.load_config(CoreConfig, "test")

        assert loaded_config.environment == "production"
        assert loaded_config.debug is True
        assert loaded_config.log_level == "DEBUG"

    def test_config_validation(self):
        """Test configuration validation"""
        # Valid configuration
        valid_config = RAGConfig(
            chunk_size=1000,
            similarity_threshold=0.7
        )
        errors = self.config_manager.validate_config(valid_config)
        assert len(errors) == 0

        # Invalid configuration
        invalid_config = RAGConfig(
            chunk_size=-1,  # Invalid
            similarity_threshold=1.5  # Invalid
        )
        errors = self.config_manager.validate_config(invalid_config)
        assert len(errors) > 0

    def test_environment_override(self):
        """Test environment variable override"""
        os.environ["AI_SCIENTIST_RAG_CHUNK_SIZE"] = "2000"
        os.environ["AI_SCIENTIST_RAG_ENABLED"] = "false"

        try:
            config = self.config_manager.load_config(RAGConfig, "rag")
            assert config.chunk_size == 2000
            assert config.enabled is False
        finally:
            # Cleanup environment variables
            os.environ.pop("AI_SCIENTIST_RAG_CHUNK_SIZE", None)
            os.environ.pop("AI_SCIENTIST_RAG_ENABLED", None)

    def test_config_migration(self):
        """Test configuration migration from legacy files"""
        # Create legacy config file
        legacy_config = {
            "rag": {
                "chunk_size": 1500,
                "similarity_threshold": 0.8,
                "vector_store": "chroma"
            }
        }
        legacy_file = self.temp_dir / "legacy_rag.yaml"
        with open(legacy_file, 'w') as f:
            yaml.dump(legacy_config, f)

        # Mock current working directory
        with patch('pathlib.Path.cwd', return_value=self.temp_dir):
            config_manager = ConfigManager(self.temp_dir / "config")
            config = config_manager.load_config(RAGConfig, "rag")

        assert config.chunk_size == 1500
        assert config.similarity_threshold == 0.8

class TestSecurityManager:
    """Test suite for security management"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.security_manager = SecurityManager(self.temp_dir / "security")

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_api_key(self):
        """Test API key generation"""
        key_id, api_key = self.security_manager.generate_api_key(
            description="Test API key",
            permissions=["read", "write"],
            rate_limit=100
        )

        assert key_id.startswith("ask_")
        assert api_key.startswith("sk-or-")
        assert len(api_key) > 32

    def test_validate_api_key(self):
        """Test API key validation"""
        key_id, api_key = self.security_manager.generate_api_key()

        # Valid key
        key_info = self.security_manager.validate_api_key(api_key)
        assert key_info is not None
        assert key_info.key_id == key_id
        assert key_info.is_active is True

        # Invalid key
        invalid_key_info = self.security_manager.validate_api_key("invalid_key")
        assert invalid_key_info is None

    def test_revoke_api_key(self):
        """Test API key revocation"""
        key_id, api_key = self.security_manager.generate_api_key()

        # Key should be valid initially
        assert self.security_manager.validate_api_key(api_key) is not None

        # Revoke key
        assert self.security_manager.revoke_api_key(key_id) is True

        # Key should now be invalid
        assert self.security_manager.validate_api_key(api_key) is None

    def test_encryption_decryption(self):
        """Test data encryption and decryption"""
        test_data = "sensitive information"

        # Encrypt data
        encrypted = self.security_manager.encrypt_sensitive_data(test_data)
        assert encrypted is not None
        assert encrypted != test_data

        # Decrypt data
        decrypted = self.security_manager.decrypt_sensitive_data(encrypted)
        assert decrypted == test_data

    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        client_id = "test_client"
        limit = 3

        # First 3 requests should pass
        for i in range(limit):
            assert self.security_manager.check_rate_limit(client_id, limit) is True

        # 4th request should fail
        assert self.security_manager.check_rate_limit(client_id, limit) is False

    def test_security_audit_logging(self):
        """Test security audit logging"""
        audit_log_file = self.temp_dir / "security" / "audit.log"

        # Generate some audit events
        self.security_manager.generate_api_key("test key")
        self.security_manager.revoke_api_key("ask_test")

        # Check that audit log was created and contains events
        assert audit_log_file.exists()
        with open(audit_log_file, 'r') as f:
            log_content = f.read()
            assert "api_key_created" in log_content
            assert "api_key_revoked" in log_content

class TestErrorHandler:
    """Test suite for error handling"""

    def test_custom_exceptions(self):
        """Test custom exception classes"""
        # Test ConfigurationError
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("Invalid configuration")
        assert exc_info.value.category.name == "CONFIGURATION"

        # Test APIError
        with pytest.raises(APIError) as exc_info:
            raise APIError("API call failed", status_code=500)
        assert exc_info.value.category.name == "API"
        assert exc_info.value.status_code == 500

        # Test SecurityError
        with pytest.raises(SecurityError) as exc_info:
            raise SecurityError("Security violation")
        assert exc_info.value.category.name == "SECURITY"

    def test_error_decorator(self):
        """Test error handling decorator"""
        @handle_errors(reraise=False, default_return="fallback")
        def failing_function():
            raise ValueError("Function failed")

        result = failing_function()
        assert result == "fallback"

    def test_retry_decorator(self):
        """Test retry with backoff decorator"""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.1)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Still failing")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 3

    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        call_count = 0

        @circuit_breaker(failure_threshold=2, recovery_timeout=1)
        def unreliable_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("Service unavailable")
            return "recovered"

        # First two calls should fail
        with pytest.raises(AIScientistError):
            unreliable_function()
        with pytest.raises(AIScientistError):
            unreliable_function()

        # Third call should be blocked by circuit breaker
        with pytest.raises(AIScientistError) as exc_info:
            unreliable_function()
        assert "Circuit breaker is OPEN" in str(exc_info.value)

    def test_error_metrics(self):
        """Test error metrics collection"""
        # Generate some errors
        try:
            raise ConfigurationError("Config error")
        except AIScientistError as e:
            pass  # Error is tracked automatically

        try:
            raise APIError("API error")
        except AIScientistError as e:
            pass  # Error is tracked automatically

        metrics = get_error_metrics()
        assert metrics["total_errors"] >= 2
        assert "CONFIGURATION" in metrics["errors_by_category"]
        assert "API" in metrics["errors_by_category"]

class TestLoggingSystem:
    """Test suite for logging system"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.log_file = self.temp_dir / "test.log"
        setup_logging(
            level="DEBUG",
            log_file=str(self.log_file),
            structured=True
        )

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_structured_logger(self):
        """Test structured logger functionality"""
        logger = get_logger("test_logger")

        # Test basic logging
        logger.info("Test message")
        logger.error("Error message", additional_fields={"error_code": 500})

        # Check log file
        assert self.log_file.exists()
        with open(self.log_file, 'r') as f:
            log_content = f.read()
            assert "Test message" in log_content
            assert "error_code" in log_content

    def test_log_context(self):
        """Test log context functionality"""
        context = LogContext(
            component="test_component",
            operation="test_operation",
            user_id="test_user",
            additional_fields={"custom_field": "custom_value"}
        )

        logger = get_logger("test_logger", context)
        logger.info("Message with context")

        # Check that context is included in log
        with open(self.log_file, 'r') as f:
            log_content = f.read()
            assert "test_component" in log_content
            assert "test_operation" in log_content
            assert "test_user" in log_content
            assert "custom_field" in log_content

    def test_performance_monitoring(self):
        """Test performance monitoring"""
        monitor = get_performance_monitor()

        # Simulate some operations
        monitor.record_operation("test_operation", 0.5, True)
        monitor.record_operation("test_operation", 0.7, True)
        monitor.record_operation("test_operation", 1.2, False)

        metrics = monitor.get_metrics("test_operation")
        assert metrics["count"] == 3
        assert metrics["avg_duration"] == pytest.approx(0.8, rel=0.1)
        assert metrics["error_rate"] == pytest.approx(0.333, rel=0.1)

    @performance_monitor
    def monitored_function(self):
        """Test function with performance monitoring decorator"""
        time.sleep(0.1)
        return "result"

    def test_performance_decorator(self):
        """Test performance monitoring decorator"""
        result = self.monitored_function()
        assert result == "result"

        # Check that performance was recorded
        metrics = get_performance_monitor().get_metrics("monitored_function")
        assert metrics["count"] >= 1

    def test_health_checks(self):
        """Test health check functionality"""
        health_checker = get_health_checker()

        # Register some health checks
        health_checker.register_health_check("database", lambda: True)
        health_checker.register_health_check("cache", lambda: False)

        # Run health checks
        status = health_checker.run_health_checks()
        assert status.status == "degraded"  # One check failed
        assert status.components["database"] == "healthy"
        assert status.components["cache"] == "unhealthy"

class TestPerformanceOptimizer:
    """Test suite for performance optimization"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        cache_config = CacheConfig(
            max_size=100,
            ttl_seconds=60,
            disk_cache_dir=str(self.temp_dir / "cache")
        )
        self.optimizer = PerformanceOptimizer(cache_config)

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_memory_cache(self):
        """Test memory caching functionality"""
        # Set cache entry
        self.optimizer.set_cache("test_key", "test_value", 10, "memory")

        # Get cache entry
        value = self.optimizer.get_from_cache("test_key", "memory")
        assert value == "test_value"

        # Test non-existent key
        value = self.optimizer.get_from_cache("non_existent", "memory")
        assert value is None

    def test_disk_cache(self):
        """Test disk caching functionality"""
        # Set cache entry
        self.optimizer.set_cache("test_key", {"complex": "data"}, 10, "disk")

        # Get cache entry
        value = self.optimizer.get_from_cache("test_key", "disk")
        assert value == {"complex": "data"}

    def test_cache_decorator(self):
        """Test caching decorator"""
        call_count = 0

        @cache_result(ttl_seconds=10, cache_type="memory")
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call should execute function
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment

    @pytest.mark.asyncio
    async def test_async_processing(self):
        """Test async processing functionality"""
        async def async_task(x):
            await asyncio.sleep(0.1)
            return x * 2

        def sync_task(x):
            time.sleep(0.1)
            return x + 1

        tasks = [async_task, sync_task, async_task]
        results = await self.optimizer.process_async(tasks)

        assert len(results) == 3
        assert results[0] == 2  # async_task(1)
        assert results[1] == 2  # sync_task(1)
        assert results[2] == 4  # async_task(2)

    def test_performance_stats(self):
        """Test performance statistics"""
        # Generate some cache activity
        for i in range(10):
            self.optimizer.set_cache(f"key_{i}", f"value_{i}")
            self.optimizer.get_from_cache(f"key_{i}")

        stats = self.optimizer.get_performance_stats()
        assert "memory_cache" in stats
        assert stats["memory_cache"]["size"] > 0

class TestEndToEndIntegration:
    """End-to-end integration tests"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        os.environ["AI_SCIENTIST_TEST_MODE"] = "true"

        # Initialize all components with test directories
        self.config_manager = ConfigManager(self.temp_dir / "config")
        self.security_manager = SecurityManager(self.temp_dir / "security")
        self.optimizer = PerformanceOptimizer(
            CacheConfig(disk_cache_dir=str(self.temp_dir / "cache"))
        )

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        os.environ.pop("AI_SCIENTIST_TEST_MODE", None)

    def test_full_workflow(self):
        """Test complete workflow with all components"""
        # Load configuration
        config = self.config_manager.load_config(CoreConfig)
        assert config is not None

        # Generate API key
        key_id, api_key = self.security_manager.generate_api_key()
        assert api_key is not None

        # Validate API key
        key_info = self.security_manager.validate_api_key(api_key)
        assert key_info is not None

        # Use logging
        logger = get_logger("integration_test")
        logger.info("Integration test started")

        # Test error handling
        @handle_errors(reraise=False, default_return="error_handled")
        def error_prone_function():
            raise ValueError("Test error")

        result = error_prone_function()
        assert result == "error_handled"

        # Test caching
        @cache_result(ttl_seconds=60)
        def cached_function(x):
            return x ** 2

        result = cached_function(5)
        assert result == 25

        # Check performance stats
        perf_stats = self.optimizer.get_performance_stats()
        assert "memory_cache" in perf_stats

        logger.info("Integration test completed successfully")

    def test_error_recovery(self):
        """Test error recovery mechanisms"""
        logger = get_logger("error_recovery_test")

        # Test configuration error recovery
        with patch.object(self.config_manager, '_load_from_file', side_effect=Exception("Load failed")):
            config = self.config_manager.load_config(CoreConfig)
            assert config is not None  # Should fallback to default

        # Test API error recovery with retry
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.1)
        def api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIError("API unavailable", status_code=503)
            return "success"

        result = api_call()
        assert result == "success"
        assert call_count == 3

        logger.info("Error recovery test completed")

    def test_security_integration(self):
        """Test security integration with other components"""
        # Test secure configuration storage
        sensitive_config = {"api_key": "secret123", "database_url": "db://secret"}
        encrypted = self.security_manager.encrypt_sensitive_data(str(sensitive_config))
        assert encrypted is not None

        decrypted = self.security_manager.decrypt_sensitive_data(encrypted)
        assert decrypted == str(sensitive_config)

        # Test audit logging for security events
        audit_log_file = self.temp_dir / "security" / "audit.log"
        self.security_manager.generate_api_key("integration_test")

        assert audit_log_file.exists()
        with open(audit_log_file, 'r') as f:
            log_content = f.read()
            assert "api_key_created" in log_content

    @pytest.mark.asyncio
    async def test_async_integration(self):
        """Test async integration with all components"""
        logger = get_logger("async_integration_test")

        async def async_task_1():
            await asyncio.sleep(0.1)
            return "task1_result"

        async def async_task_2():
            await asyncio.sleep(0.1)
            raise ValueError("Task 2 failed")

        def sync_task():
            time.sleep(0.1)
            return "sync_result"

        # Process tasks asynchronously
        tasks = [async_task_1, async_task_2, sync_task]
        results = await self.optimizer.process_async(tasks)

        assert len(results) == 3
        assert results[0] == "task1_result"
        assert isinstance(results[1], ValueError)
        assert results[2] == "sync_result"

        logger.info("Async integration test completed")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])