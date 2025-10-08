"""
Performance Optimization System

Provides caching, connection pooling, async processing,
and resource optimization for AI-Scientist-v2.
"""

import asyncio
import threading
import time
import functools
import hashlib
import pickle
import json
from typing import Dict, Any, Optional, List, Callable, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref
import gc
from contextlib import contextmanager

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .logging_system import get_logger, performance_monitor
from .error_handler import handle_errors, AIScientistError

logger = get_logger("performance_optimizer")
T = TypeVar('T')

@dataclass
class CacheConfig:
    """Cache configuration settings"""
    max_size: int = 1000
    ttl_seconds: int = 3600
    cleanup_interval: int = 300
    enable_disk_cache: bool = True
    disk_cache_dir: str = "./cache"
    max_memory_mb: int = 512

@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    max_workers: int = 4
    connection_pool_size: int = 10
    enable_async: bool = True
    memory_limit_mb: int = 2048
    enable_compression: bool = True
    batch_size: int = 100
    timeout_seconds: int = 30

class CacheEntry:
    """Cache entry with metadata"""

    def __init__(self, key: str, value: Any, ttl_seconds: int = None):
        self.key = key
        self.value = value
        self.created_at = datetime.utcnow()
        self.expires_at = (
            self.created_at + timedelta(seconds=ttl_seconds)
            if ttl_seconds else None
        )
        self.access_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def access(self) -> Any:
        """Access the cache entry"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
        return self.value

class MemoryCache:
    """In-memory LRU cache with TTL support"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._access_order: List[str] = []

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if entry.is_expired():
                self._remove_entry(key)
                return None

            # Update access order for LRU
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            return entry.access()

    def set(self, key: str, value: Any, ttl_seconds: int = None) -> None:
        """Set value in cache"""
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)

            # Check if we need to evict entries
            if len(self._cache) >= self.config.max_size:
                self._evict_lru()

            # Create new entry
            ttl = ttl_seconds or self.config.ttl_seconds
            entry = CacheEntry(key, value, ttl)
            self._cache[key] = entry
            self._access_order.append(key)

    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache"""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)

    def _evict_lru(self) -> None:
        """Evict least recently used entries"""
        if len(self._access_order) > 0:
            # Remove 20% of least recently used entries
            evict_count = max(1, len(self._access_order) // 5)
            for _ in range(evict_count):
                if self._access_order:
                    key = self._access_order.pop(0)
                    self._remove_entry(key)

    def _cleanup_worker(self) -> None:
        """Background cleanup worker thread"""
        while True:
            time.sleep(self.config.cleanup_interval)
            self._cleanup_expired()

    def _cleanup_expired(self) -> None:
        """Clean up expired entries"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                self._remove_entry(key)

            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.config.max_size,
                "expired_entries": sum(1 for entry in self._cache.values() if entry.is_expired()),
                "memory_usage_mb": self._estimate_memory_usage(),
                "hit_rate": self._calculate_hit_rate()
            }

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        try:
            total_size = sum(
                len(pickle.dumps(entry.value))
                for entry in self._cache.values()
            )
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)"""
        # This would be enhanced with actual hit/miss tracking
        return 0.8  # Placeholder

class DiskCache:
    """Disk-based cache for large objects"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.disk_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        cache_file = self._get_cache_file(key)

        if not cache_file.exists():
            return None

        try:
            # Check if file is expired
            stat = cache_file.stat()
            if self.config.ttl_seconds > 0:
                age = time.time() - stat.st_mtime
                if age > self.config.ttl_seconds:
                    cache_file.unlink()
                    return None

            # Load value
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
            if cache_file.exists():
                cache_file.unlink()
            return None

    def set(self, key: str, value: Any) -> None:
        """Set value in disk cache"""
        cache_file = self._get_cache_file(key)

        try:
            # Write to temporary file first, then move
            temp_file = cache_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(value, f)

            temp_file.replace(cache_file)

        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key"""
        # Hash key to get safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def cleanup(self) -> None:
        """Clean up expired cache files"""
        if self.config.ttl_seconds <= 0:
            return

        try:
            cutoff_time = time.time() - self.config.ttl_seconds
            removed_count = 0

            for cache_file in self.cache_dir.glob("*.cache"):
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    removed_count += 1

            logger.info(f"Cleaned up {removed_count} expired disk cache files")

        except Exception as e:
            logger.warning(f"Disk cache cleanup failed: {e}")

class RedisCache:
    """Redis-based distributed cache"""

    def __init__(self, config: CacheConfig, redis_url: str = "redis://localhost:6379"):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is required for RedisCache")

        self.config = config
        self.redis_client = redis.from_url(redis_url)
        self._test_connection()

    def _test_connection(self):
        """Test Redis connection"""
        try:
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            value = self.redis_client.get(key)
            if value is not None:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None

    def set(self, key: str, value: Any, ttl_seconds: int = None) -> None:
        """Set value in Redis cache"""
        try:
            ttl = ttl_seconds or self.config.ttl_seconds
            pickled_value = pickle.dumps(value)

            if ttl > 0:
                self.redis_client.setex(key, ttl, pickled_value)
            else:
                self.redis_client.set(key, pickled_value)

        except Exception as e:
            logger.warning(f"Redis set failed: {e}")

class ConnectionPool:
    """Generic connection pool with automatic management"""

    def __init__(
        self,
        connection_factory: Callable,
        max_size: int = 10,
        max_idle_time: int = 300,
        health_check: Callable = None
    ):
        self.connection_factory = connection_factory
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.health_check = health_check

        self._pool: List[Any] = []
        self._pool_lock = threading.Lock()
        self._connection_stats: Dict[str, Any] = {
            "created": 0,
            "acquired": 0,
            "released": 0,
            "closed": 0
        }

    def get_connection(self) -> Any:
        """Get connection from pool"""
        with self._pool_lock:
            # Clean up idle connections
            self._cleanup_idle_connections()

            # Check for available connection
            if self._pool:
                connection = self._pool.pop()
                if self._is_connection_healthy(connection):
                    self._connection_stats["acquired"] += 1
                    return connection
                else:
                    self._close_connection(connection)

            # Create new connection if pool not full
            if len(self._pool) < self.max_size:
                connection = self.connection_factory()
                self._connection_stats["created"] += 1
                self._connection_stats["acquired"] += 1
                return connection

            # Pool is full, wait and retry
            raise AIScientistError("Connection pool exhausted")

    def release_connection(self, connection: Any) -> None:
        """Release connection back to pool"""
        with self._pool_lock:
            if self._is_connection_healthy(connection) and len(self._pool) < self.max_size:
                self._pool.append(connection)
                self._connection_stats["released"] += 1
            else:
                self._close_connection(connection)

    def _is_connection_healthy(self, connection: Any) -> bool:
        """Check if connection is healthy"""
        if self.health_check:
            try:
                return self.health_check(connection)
            except Exception:
                return False
        return True

    def _cleanup_idle_connections(self) -> None:
        """Clean up idle connections"""
        current_time = time.time()
        healthy_connections = []

        for connection in self._pool:
            # This is simplified - in practice, you'd track connection creation time
            if self._is_connection_healthy(connection):
                healthy_connections.append(connection)
            else:
                self._close_connection(connection)

        self._pool = healthy_connections

    def _close_connection(self, connection: Any) -> None:
        """Close connection"""
        try:
            if hasattr(connection, 'close'):
                connection.close()
            self._connection_stats["closed"] += 1
        except Exception as e:
            logger.warning(f"Failed to close connection: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        with self._pool_lock:
            return {
                "pool_size": len(self._pool),
                "max_size": self.max_size,
                **self._connection_stats
            }

class AsyncProcessor:
    """Async task processor with batching and queuing"""

    def __init__(self, max_workers: int = 4, batch_size: int = 10):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self._queue = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._running = False

    async def start(self):
        """Start the async processor"""
        if self._running:
            return

        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_workers)
        ]

        logger.info(f"Started async processor with {self.max_workers} workers")

    async def stop(self):
        """Stop the async processor"""
        if not self._running:
            return

        self._running = False

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)

        logger.info("Stopped async processor")

    async def submit_task(self, task: Callable, *args, **kwargs) -> asyncio.Future:
        """Submit task for processing"""
        if not self._running:
            raise AIScientistError("Async processor is not running")

        future = asyncio.Future()
        await self._queue.put((task, args, kwargs, future))
        return future

    async def _worker(self, worker_name: str):
        """Worker coroutine"""
        logger.debug(f"Worker {worker_name} started")

        while self._running:
            try:
                # Get task from queue with timeout
                task_item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                task, args, kwargs, future = task_item

                # Execute task
                try:
                    result = await task(*args, **kwargs) if asyncio.iscoroutinefunction(task) else task(*args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)

            except asyncio.TimeoutError:
                # No task available, continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")

        logger.debug(f"Worker {worker_name} stopped")

class ResourceMonitor:
    """Monitor and manage system resources"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self._monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        self._running = False

    def start(self):
        """Start resource monitoring"""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, resource monitoring disabled")
            return

        self._running = True
        self._monitor_thread.start()
        logger.info("Started resource monitoring")

    def stop(self):
        """Stop resource monitoring"""
        self._running = False
        logger.info("Stopped resource monitoring")

    def _monitor_worker(self):
        """Resource monitoring worker thread"""
        while self._running:
            try:
                self._check_resources()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")

    def _check_resources(self):
        """Check system resources and take action if needed"""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024

        # Check memory limit
        if memory_mb > self.config.memory_limit_mb:
            logger.warning(f"Memory usage ({memory_mb:.1f}MB) exceeds limit ({self.config.memory_limit_mb}MB)")
            self._reduce_memory_usage()

        # Check CPU usage
        cpu_percent = process.cpu_percent()
        if cpu_percent > 80:
            logger.warning(f"High CPU usage: {cpu_percent:.1f}%")

    def _reduce_memory_usage(self):
        """Reduce memory usage"""
        # Force garbage collection
        gc.collect()

        # Clear caches if available
        # This would be implemented with cache manager integration

        logger.info("Reduced memory usage through garbage collection")

class PerformanceOptimizer:
    """Main performance optimization system"""

    def __init__(self, cache_config: CacheConfig = None, perf_config: PerformanceConfig = None):
        self.cache_config = cache_config or CacheConfig()
        self.perf_config = perf_config or PerformanceConfig()

        # Initialize caches
        self.memory_cache = MemoryCache(self.cache_config)
        self.disk_cache = DiskCache(self.cache_config) if self.cache_config.enable_disk_cache else None

        # Initialize Redis if available
        self.redis_cache = None
        if REDIS_AVAILABLE:
            try:
                self.redis_cache = RedisCache(self.cache_config)
            except Exception:
                logger.warning("Redis not available, falling back to local caches")

        # Initialize other components
        self.async_processor = AsyncProcessor(self.perf_config.max_workers, self.perf_config.batch_size)
        self.resource_monitor = ResourceMonitor(self.perf_config)

        # Thread pool for CPU-bound tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=self.perf_config.max_workers)

    @contextmanager
    def connection_pool(self, connection_factory: Callable, pool_name: str = "default"):
        """Context manager for connection pools"""
        pool = ConnectionPool(
            connection_factory,
            max_size=self.perf_config.connection_pool_size
        )

        try:
            yield pool
        finally:
            # Pool will be cleaned up when no longer referenced
            pass

    def cache_result(self, ttl_seconds: int = None, cache_type: str = "memory"):
        """Decorator for caching function results"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                key_data = {
                    "func": func.__name__,
                    "args": args,
                    "kwargs": kwargs
                }
                cache_key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

                # Try to get from cache
                cached_value = self.get_from_cache(cache_key, cache_type)
                if cached_value is not None:
                    return cached_value

                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set_cache(cache_key, result, ttl_seconds, cache_type)
                return result

            return wrapper
        return decorator

    def get_from_cache(self, key: str, cache_type: str = "memory") -> Optional[Any]:
        """Get value from specified cache"""
        try:
            if cache_type == "redis" and self.redis_cache:
                return self.redis_cache.get(key)
            elif cache_type == "disk" and self.disk_cache:
                return self.disk_cache.get(key)
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            logger.warning(f"Cache get failed for {cache_type}: {e}")
            return None

    def set_cache(self, key: str, value: Any, ttl_seconds: int = None, cache_type: str = "memory"):
        """Set value in specified cache"""
        try:
            if cache_type == "redis" and self.redis_cache:
                self.redis_cache.set(key, value, ttl_seconds)
            elif cache_type == "disk" and self.disk_cache:
                self.disk_cache.set(key, value)
            else:
                self.memory_cache.set(key, value, ttl_seconds)
        except Exception as e:
            logger.warning(f"Cache set failed for {cache_type}: {e}")

    async def process_async(self, tasks: List[Callable]) -> List[Any]:
        """Process tasks asynchronously"""
        if not self.async_processor._running:
            await self.async_processor.start()

        futures = []
        for task in tasks:
            if asyncio.iscoroutinefunction(task):
                future = await self.async_processor.submit_task(task)
            else:
                # Run CPU-bound tasks in thread pool
                future = asyncio.get_event_loop().run_in_executor(self.thread_pool, task)
            futures.append(future)

        results = await asyncio.gather(*futures, return_exceptions=True)
        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            "memory_cache": self.memory_cache.get_stats(),
            "disk_cache": self.disk_cache.get_stats() if self.disk_cache else None,
            "redis_cache": self.redis_cache.get_stats() if self.redis_cache else None,
            "async_processor": {
                "running": self.async_processor._running,
                "max_workers": self.async_processor.max_workers,
                "batch_size": self.async_processor.batch_size
            },
            "resource_monitor": {
                "running": self.resource_monitor._running,
                "memory_limit_mb": self.perf_config.memory_limit_mb
            }
        }

    async def cleanup(self):
        """Cleanup resources"""
        await self.async_processor.stop()
        self.resource_monitor.stop()
        self.thread_pool.shutdown(wait=True)

        if self.disk_cache:
            self.disk_cache.cleanup()

        logger.info("Performance optimizer cleanup completed")

# Global performance optimizer instance
_performance_optimizer = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance"""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer

# Convenience functions
def cache_result(ttl_seconds: int = None, cache_type: str = "memory"):
    """Convenience decorator for caching"""
    optimizer = get_performance_optimizer()
    return optimizer.cache_result(ttl_seconds, cache_type)

async def process_tasks_async(tasks: List[Callable]) -> List[Any]:
    """Convenience function for async processing"""
    optimizer = get_performance_optimizer()
    return await optimizer.process_async(tasks)