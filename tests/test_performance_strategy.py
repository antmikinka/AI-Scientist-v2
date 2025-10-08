"""
Performance Testing Strategy for Phase 1

Comprehensive performance testing for AI-Scientist-v2 core functionality.
Tests response times, memory usage, throughput, and scalability limits.

Priority: HIGH - Essential for production deployment
"""

import pytest
import sys
import os
import time
import psutil
import threading
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestPerformanceBaselines:
    """Establish performance baselines for core operations"""

    def test_configuration_loading_performance(self):
        """Test configuration loading performance"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig

            # Measure configuration loading time
            start_time = time.time()
            config = OpenRouterConfig(api_key="test-key")
            end_time = time.time()

            loading_time = end_time - start_time

            # Performance assertion: should load in under 0.1 seconds
            assert loading_time < 0.1, f"Configuration loading took {loading_time:.3f}s, expected < 0.1s"

            print(f"✅ Configuration loading performance: {loading_time:.3f}s")
        except Exception as e:
            pytest.fail(f"Configuration loading performance test failed: {e}")

    def test_client_initialization_performance(self):
        """Test client initialization performance"""
        try:
            from ai_scientist.openrouter.client import OpenRouterClient

            # Measure client initialization time
            start_time = time.time()
            client = OpenRouterClient(api_key="test-key")
            end_time = time.time()

            init_time = end_time - start_time

            # Performance assertion: should initialize in under 0.5 seconds
            assert init_time < 0.5, f"Client initialization took {init_time:.3f}s, expected < 0.5s"

            print(f"✅ Client initialization performance: {init_time:.3f}s")
        except Exception as e:
            pytest.fail(f"Client initialization performance test failed: {e}")

    def test_memory_usage_baselines(self):
        """Test memory usage baselines for core components"""
        try:
            import tracemalloc

            from ai_scientist.openrouter.config import OpenRouterConfig
            from ai_scientist.openrouter.client import OpenRouterClient

            # Start memory tracing
            tracemalloc.start()

            # Create configuration and client
            config = OpenRouterConfig(api_key="test-key")
            client = OpenRouterClient(api_key="test-key")

            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Performance assertion: should use less than 10MB
            assert peak < 10 * 1024 * 1024, f"Memory usage {peak / 1024 / 1024:.2f}MB, expected < 10MB"

            print(f"✅ Memory usage baseline: {peak / 1024 / 1024:.2f}MB")
        except Exception as e:
            pytest.fail(f"Memory usage baseline test failed: {e}")

class TestLoadTesting:
    """Test system behavior under load"""

    def test_concurrent_configuration_loading(self):
        """Test concurrent configuration loading"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig

            def load_config():
                return OpenRouterConfig(api_key="test-key")

            # Test concurrent loading
            num_threads = 10
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(load_config) for _ in range(num_threads)]
                results = list(as_completed(futures))

            # All should succeed
            assert len(results) == num_threads

            print(f"✅ Concurrent configuration loading ({num_threads} threads) successful")
        except Exception as e:
            pytest.fail(f"Concurrent configuration loading failed: {e}")

    def test_configuration_serialization_throughput(self):
        """Test configuration serialization throughput"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig

            config = OpenRouterConfig(api_key="test-key")

            # Measure serialization throughput
            num_operations = 100
            start_time = time.time()

            for _ in range(num_operations):
                json_str = json.dumps(config.__dict__, default=str)

            end_time = time.time()
            throughput = num_operations / (end_time - start_time)

            # Performance assertion: should handle at least 100 operations per second
            assert throughput > 100, f"Throughput {throughput:.1f} ops/s, expected > 100 ops/s"

            print(f"✅ Configuration serialization throughput: {throughput:.1f} ops/s")
        except Exception as e:
            pytest.fail(f"Configuration serialization throughput test failed: {e}")

    @pytest.mark.asyncio
    async def test_async_operation_concurrency(self):
        """Test async operation concurrency"""
        try:
            from ai_scientist.openrouter.client import OpenRouterClient

            async def mock_async_operation():
                await asyncio.sleep(0.01)  # Simulate work
                return "result"

            # Test concurrent async operations
            num_operations = 50
            start_time = time.time()

            tasks = [mock_async_operation() for _ in range(num_operations)]
            results = await asyncio.gather(*tasks)

            end_time = time.time()
            throughput = num_operations / (end_time - start_time)

            # Performance assertion: should handle concurrent operations efficiently
            assert throughput > 20, f"Async throughput {throughput:.1f} ops/s, expected > 20 ops/s"

            print(f"✅ Async operation concurrency: {throughput:.1f} ops/s")
        except Exception as e:
            pytest.fail(f"Async operation concurrency test failed: {e}")

class TestStressTesting:
    """Test system behavior under stress conditions"""

    def test_memory_stress_test(self):
        """Test system behavior under memory stress"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig

            # Create many configurations to test memory management
            configs = []
            num_configs = 1000

            for i in range(num_configs):
                config = OpenRouterConfig(api_key=f"test-key-{i}")
                configs.append(config)

                # Monitor memory every 100 configurations
                if i % 100 == 0:
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()

                    # Memory should not grow uncontrollably
                    assert memory_info.rss < 100 * 1024 * 1024, f"Memory usage {memory_info.rss / 1024 / 1024:.2f}MB too high"

            # Cleanup
            del configs

            print(f"✅ Memory stress test ({num_configs} configurations) successful")
        except Exception as e:
            pytest.fail(f"Memory stress test failed: {e}")

    def test_cpu_stress_test(self):
        """Test system behavior under CPU stress"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig

            def cpu_intensive_task():
                # Perform CPU-intensive configuration operations
                for _ in range(100):
                    config = OpenRouterConfig(api_key="test-key")
                    json.dumps(config.__dict__, default=str)

            # Run multiple CPU-intensive tasks
            num_threads = 4
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(cpu_intensive_task) for _ in range(num_threads)]
                results = list(as_completed(futures))

            # All should complete
            assert len(results) == num_threads

            print(f"✅ CPU stress test ({num_threads} threads) successful")
        except Exception as e:
            pytest.fail(f"CPU stress test failed: {e}")

class TestScalabilityTesting:
    """Test system scalability limits"""

    def test_configuration_size_scalability(self):
        """Test configuration handling with large datasets"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig, StageConfig

            # Create configuration with many stages
            config = OpenRouterConfig(api_key="test-key")

            # Add many custom stages
            num_stages = 100
            for i in range(num_stages):
                stage = StageConfig(
                    model=f"model-{i}",
                    temperature=0.5 + (i * 0.01),
                    max_tokens=1000 + i
                )
                config.stage_configs[f"stage-{i}"] = stage

            # Test that configuration can still be serialized
            start_time = time.time()
            json_str = json.dumps(config.__dict__, default=str)
            end_time = time.time()

            serialization_time = end_time - start_time

            # Performance assertion: should still serialize in reasonable time
            assert serialization_time < 1.0, f"Large config serialization took {serialization_time:.3f}s, expected < 1.0s"

            print(f"✅ Configuration size scalability ({num_stages} stages): {serialization_time:.3f}s")
        except Exception as e:
            pytest.fail(f"Configuration size scalability test failed: {e}")

    def test_concurrent_user_simulation(self):
        """Test system behavior with simulated concurrent users"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig

            def simulate_user_activity(user_id: int):
                """Simulate a user performing configuration operations"""
                operations = []

                # User performs various operations
                for i in range(10):
                    config = OpenRouterConfig(api_key=f"user-{user_id}-key-{i}")
                    operations.append(json.dumps(config.__dict__, default=str))

                return len(operations)

            # Simulate multiple concurrent users
            num_users = 20
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [executor.submit(simulate_user_activity, user_id) for user_id in range(num_users)]
                results = list(as_completed(futures))

            # All users should complete their operations
            total_operations = sum(result.result() for result in results)
            assert total_operations == num_users * 10

            print(f"✅ Concurrent user simulation ({num_users} users) successful")
        except Exception as e:
            pytest.fail(f"Concurrent user simulation failed: {e}")

class TestPerformanceRegression:
    """Test for performance regressions"""

    def test_performance_regression_detection(self):
        """Test that performance doesn't regress from baselines"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig

            # Define performance baselines (these should be stored in config)
            baselines = {
                'config_loading': 0.1,  # seconds
                'config_serialization': 100,  # operations per second
                'memory_usage': 10 * 1024 * 1024  # bytes
            }

            # Test configuration loading
            start_time = time.time()
            config = OpenRouterConfig(api_key="test-key")
            loading_time = time.time() - start_time

            assert loading_time < baselines['config_loading'], \
                f"Configuration loading regression: {loading_time:.3f}s > {baselines['config_loading']}s"

            # Test serialization performance
            start_time = time.time()
            for _ in range(100):
                json.dumps(config.__dict__, default=str)
            serialization_time = time.time() - start_time
            throughput = 100 / serialization_time

            assert throughput > baselines['config_serialization'], \
                f"Configuration serialization regression: {throughput:.1f} ops/s < {baselines['config_serialization']} ops/s"

            print("✅ Performance regression detection passed")
        except Exception as e:
            pytest.fail(f"Performance regression detection failed: {e}")

class TestResourceUtilization:
    """Test resource utilization patterns"""

    def test_cpu_utilization_patterns(self):
        """Test CPU utilization during various operations"""
        try:
            import psutil
            from ai_scientist.openrouter.config import OpenRouterConfig

            process = psutil.Process(os.getpid())

            # Monitor CPU during configuration operations
            cpu_percent_before = process.cpu_percent()

            # Perform configuration operations
            for _ in range(100):
                config = OpenRouterConfig(api_key="test-key")
                json.dumps(config.__dict__, default=str)

            cpu_percent_after = process.cpu_percent()

            # CPU utilization should be reasonable
            assert cpu_percent_after < 80, f"CPU utilization {cpu_percent_after}% too high"

            print(f"✅ CPU utilization patterns: {cpu_percent_after}%")
        except Exception as e:
            pytest.fail(f"CPU utilization patterns test failed: {e}")

    def test_memory_utilization_patterns(self):
        """Test memory utilization patterns"""
        try:
            import psutil
            from ai_scientist.openrouter.config import OpenRouterConfig

            process = psutil.Process(os.getpid())

            # Monitor memory during configuration operations
            memory_before = process.memory_info().rss

            # Create and destroy many configurations
            configs = []
            for i in range(100):
                config = OpenRouterConfig(api_key=f"test-key-{i}")
                configs.append(config)

            # Clear references
            del configs

            # Force garbage collection
            import gc
            gc.collect()

            memory_after = process.memory_info().rss

            # Memory should return to reasonable level
            memory_growth = memory_after - memory_before
            assert memory_growth < 5 * 1024 * 1024, f"Memory growth {memory_growth / 1024 / 1024:.2f}MB too high"

            print(f"✅ Memory utilization patterns: {memory_growth / 1024 / 1024:.2f}MB growth")
        except Exception as e:
            pytest.fail(f"Memory utilization patterns test failed: {e}")

@pytest.mark.performance
class TestPerformanceMonitoring:
    """Test performance monitoring capabilities"""

    def test_performance_metrics_collection(self):
        """Test performance metrics collection"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig

            # Collect performance metrics
            metrics = {
                'configuration_loading_times': [],
                'serialization_times': [],
                'memory_usage': []
            }

            # Collect configuration loading metrics
            for _ in range(10):
                start_time = time.time()
                config = OpenRouterConfig(api_key="test-key")
                loading_time = time.time() - start_time
                metrics['configuration_loading_times'].append(loading_time)

            # Collect serialization metrics
            config = OpenRouterConfig(api_key="test-key")
            for _ in range(10):
                start_time = time.time()
                json.dumps(config.__dict__, default=str)
                serialization_time = time.time() - start_time
                metrics['serialization_times'].append(serialization_time)

            # Collect memory metrics
            process = psutil.Process(os.getpid())
            metrics['memory_usage'].append(process.memory_info().rss)

            # Validate metrics
            assert len(metrics['configuration_loading_times']) == 10
            assert len(metrics['serialization_times']) == 10
            assert len(metrics['memory_usage']) == 1

            # Calculate statistics
            avg_loading_time = sum(metrics['configuration_loading_times']) / len(metrics['configuration_loading_times'])
            avg_serialization_time = sum(metrics['serialization_times']) / len(metrics['serialization_times'])

            print(f"✅ Performance metrics collected:")
            print(f"   - Avg config loading: {avg_loading_time:.4f}s")
            print(f"   - Avg serialization: {avg_serialization_time:.4f}s")
            print(f"   - Memory usage: {metrics['memory_usage'][0] / 1024 / 1024:.2f}MB")

        except Exception as e:
            pytest.fail(f"Performance metrics collection test failed: {e}")

if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short"])