"""
Advanced Logging and Monitoring System

Provides structured logging, performance monitoring, health checks,
and alerting for the AI-Scientist-v2 system.
"""

import logging
import logging.handlers
import json
import time
import threading
import psutil
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
import asyncio
from collections import defaultdict, deque

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

class LogLevel(Enum):
    """Enhanced log levels with performance and security context"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    AUDIT = "AUDIT"

@dataclass
class LogContext:
    """Structured logging context"""
    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    additional_fields: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics"""
    operation_count: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    error_count: int = 0
    recent_operations: deque = field(default_factory=lambda: deque(maxlen=1000))

@dataclass
class HealthStatus:
    """System health status"""
    status: str = "healthy"  # healthy, degraded, unhealthy
    components: Dict[str, str] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.utcnow)
    issues: List[str] = field(default_factory=list)

class LogFormatter:
    """Enhanced log formatter with JSON and structured output"""

    def __init__(self, include_timestamp: bool = True, structured: bool = True):
        self.include_timestamp = include_timestamp
        self.structured = structured

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with enhanced context"""
        log_data = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat()

        # Add context if available
        if hasattr(record, 'context'):
            context = record.context
            if isinstance(context, LogContext):
                log_data.update({
                    "component": context.component,
                    "operation": context.operation,
                    "user_id": context.user_id,
                    "session_id": context.session_id,
                    "request_id": context.request_id,
                    **context.additional_fields
                })

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self._format_traceback(record.exc_info[2])
            }

        # Add performance metrics if present
        if hasattr(record, 'performance'):
            log_data["performance"] = record.performance

        if self.structured:
            return json.dumps(log_data, default=str)
        else:
            return self._format_human_readable(log_data)

    def _format_traceback(self, tb) -> List[str]:
        """Format traceback for JSON output"""
        import traceback
        return traceback.format_list(traceback.extract_tb(tb))

    def _format_human_readable(self, log_data: Dict[str, Any]) -> str:
        """Format log data in human-readable format"""
        parts = []
        if self.include_timestamp:
            parts.append(f"[{log_data['timestamp']}]")

        parts.append(f"[{log_data['level']}]")
        parts.append(f"[{log_data['logger']}]")

        if 'component' in log_data:
            parts.append(f"[{log_data['component']}]")

        if 'operation' in log_data:
            parts.append(f"[{log_data['operation']}]")

        parts.append(log_data['message'])

        if 'performance' in log_data:
            perf = log_data['performance']
            if 'duration' in perf:
                parts.append(f"(duration: {perf['duration']:.3f}s)")

        return " ".join(parts)

class PerformanceMonitor:
    """Real-time performance monitoring"""

    def __init__(self):
        self._metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self._lock = threading.Lock()
        self._system_metrics = {}

    def record_operation(self, operation: str, duration: float, success: bool = True):
        """Record operation performance metrics"""
        with self._lock:
            metrics = self._metrics[operation]
            metrics.operation_count += 1
            metrics.total_duration += duration
            metrics.min_duration = min(metrics.min_duration, duration)
            metrics.max_duration = max(metrics.max_duration, duration)

            if not success:
                metrics.error_count += 1

            metrics.recent_operations.append({
                "timestamp": datetime.utcnow().isoformat(),
                "duration": duration,
                "success": success
            })

    def get_metrics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics"""
        with self._lock:
            if operation:
                metrics = self._metrics.get(operation)
                if not metrics:
                    return {}

                return {
                    "operation": operation,
                    "count": metrics.operation_count,
                    "avg_duration": metrics.total_duration / max(1, metrics.operation_count),
                    "min_duration": metrics.min_duration if metrics.min_duration != float('inf') else 0,
                    "max_duration": metrics.max_duration,
                    "error_rate": metrics.error_count / max(1, metrics.operation_count),
                    "recent_operations": list(metrics.recent_operations)
                }
            else:
                return {
                    op: self.get_metrics(op)
                    for op in self._metrics.keys()
                }

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()

            return {
                "memory_usage_mb": memory_info.rss / 1024 / 1024,
                "cpu_percent": cpu_percent,
                "thread_count": process.num_threads(),
                "open_files": len(process.open_files()),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}

class HealthChecker:
    """System health monitoring and checks"""

    def __init__(self):
        self._health_checks: Dict[str, Callable] = {}
        self._health_status = HealthStatus()
        self._lock = threading.Lock()

    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function"""
        self._health_checks[name] = check_func

    def run_health_checks(self) -> HealthStatus:
        """Run all registered health checks"""
        with self._lock:
            status = HealthStatus()
            issues = []

            for name, check_func in self._health_checks.items():
                try:
                    is_healthy = check_func()
                    status.components[name] = "healthy" if is_healthy else "unhealthy"

                    if not is_healthy:
                        issues.append(f"Health check failed: {name}")

                except Exception as e:
                    status.components[name] = "error"
                    issues.append(f"Health check error for {name}: {str(e)}")

            status.issues = issues

            # Determine overall status
            if not issues:
                status.status = "healthy"
            elif len(issues) <= len(self._health_checks) * 0.3:
                status.status = "degraded"
            else:
                status.status = "unhealthy"

            self._health_status = status
            return status

    def get_health_status(self) -> HealthStatus:
        """Get current health status"""
        return self._health_status

class StructuredLogger:
    """Enhanced structured logger with context and performance tracking"""

    def __init__(self, name: str, context: Optional[LogContext] = None):
        self.logger = logging.getLogger(name)
        self.context = context or LogContext(component=name, operation="general")
        self.performance_monitor = get_performance_monitor()

    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal logging method with enhanced context"""
        # Update context timestamp
        self.context.timestamp = datetime.utcnow()

        # Add additional fields to context
        additional_fields = kwargs.pop('additional_fields', {})
        self.context.additional_fields.update(additional_fields)

        # Create log record
        log_record = self.logger.makeRecord(
            self.logger.name,
            getattr(logging, level.value),
            __file__,
            0,
            message,
            (),
            None
        )

        # Add enhanced context
        log_record.context = self.context
        log_record.levelname = level.value

        # Add performance metrics if provided
        if 'performance' in kwargs:
            log_record.performance = kwargs['performance']

        # Handle the log record
        self.logger.handle(log_record)

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log(LogLevel.CRITICAL, message, **kwargs)

    def security(self, message: str, **kwargs):
        """Log security event"""
        self._log(LogLevel.SECURITY, message, **kwargs)

    def performance(self, message: str, **kwargs):
        """Log performance event"""
        self._log(LogLevel.PERFORMANCE, message, **kwargs)

    def audit(self, message: str, **kwargs):
        """Log audit event"""
        self._log(LogLevel.AUDIT, message, **kwargs)

    def with_context(self, **kwargs) -> 'StructuredLogger':
        """Create new logger with updated context"""
        new_context = LogContext(
            component=kwargs.get('component', self.context.component),
            operation=kwargs.get('operation', self.context.operation),
            user_id=kwargs.get('user_id', self.context.user_id),
            session_id=kwargs.get('session_id', self.context.session_id),
            request_id=kwargs.get('request_id', self.context.request_id),
            additional_fields={**self.context.additional_fields, **kwargs.get('additional_fields', {})}
        )
        return StructuredLogger(self.logger.name, new_context)

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: str = "10MB",
    backup_count: int = 5,
    structured: bool = True,
    enable_prometheus: bool = False,
    prometheus_port: int = 8000
) -> None:
    """Setup comprehensive logging configuration"""
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatters
    formatter = LogFormatter(structured=structured)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Parse max file size
        max_size_bytes = _parse_size(max_file_size)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_size_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Setup Prometheus if available and enabled
    if enable_prometheus and PROMETHEUS_AVAILABLE:
        try:
            start_http_server(prometheus_port)
            print(f"Prometheus metrics available on port {prometheus_port}")
        except Exception as e:
            print(f"Failed to start Prometheus server: {e}")

    # Setup structured logging if available
    if STRUCTLOG_AVAILABLE:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

def _parse_size(size_str: str) -> int:
    """Parse size string like '10MB' to bytes"""
    size_str = size_str.upper()
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)

def get_logger(name: str, context: Optional[LogContext] = None) -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(name, context)

def performance_monitor(operation: str = None):
    """Decorator for performance monitoring"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = e
                raise
            finally:
                duration = time.time() - start_time
                monitor = get_performance_monitor()
                monitor.record_operation(operation or func.__name__, duration, success)

                # Log performance if it takes too long
                if duration > 1.0:  # Log operations taking more than 1 second
                    logger = get_logger("performance")
                    logger.performance(
                        f"Slow operation: {operation or func.__name__}",
                        additional_fields={
                            "duration": duration,
                            "success": success,
                            "operation": operation or func.__name__
                        }
                    )

        return wrapper
    return decorator

def audit_log(event_type: str, details: Dict[str, Any]):
    """Convenience function for audit logging"""
    logger = get_logger("audit")
    logger.audit(
        f"Audit event: {event_type}",
        additional_fields={
            "event_type": event_type,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Global instances
_performance_monitor = None
_health_checker = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def get_health_checker() -> HealthChecker:
    """Get global health checker instance"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker

# System monitoring functions
def monitor_system_resources():
    """Monitor system resource usage"""
    monitor = get_performance_monitor()
    metrics = monitor.get_system_metrics()

    logger = get_logger("system")
    logger.info(
        "System resource metrics",
        additional_fields=metrics
    )

    return metrics

def get_logging_summary() -> Dict[str, Any]:
    """Get summary of logging and monitoring status"""
    return {
        "structured_logging_enabled": STRUCTLOG_AVAILABLE,
        "prometheus_enabled": PROMETHEUS_AVAILABLE,
        "performance_monitor": get_performance_monitor().get_metrics(),
        "health_status": get_health_checker().get_health_status().__dict__,
        "loggers": list(logging.Logger.manager.loggerDict.keys())
    }