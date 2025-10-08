"""
Advanced Error Handling and Fault Tolerance System

Provides comprehensive error management, retry mechanisms,
circuit breakers, and graceful degradation for AI-Scientist-v2.
"""

import logging
import traceback
import time
import functools
from typing import Dict, Any, Optional, Callable, Type, Union, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5

class ErrorCategory(Enum):
    """Error categories for classification"""
    NETWORK = "network"
    DATABASE = "database"
    API = "api"
    VALIDATION = "validation"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    BUSINESS_LOGIC = "business_logic"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """Context information for error tracking"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    component: str = ""
    operation: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorMetrics:
    """Error metrics and statistics"""
    total_errors: int = 0
    errors_by_category: Dict[ErrorCategory, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_severity: Dict[ErrorSeverity, int] = field(default_factory=lambda: defaultdict(int))
    recent_errors: List[Dict[str, Any]] = field(default_factory=list)
    error_rates: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

class AIScientistError(Exception):
    """Base exception class for AI-Scientist-v2 system"""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.original_error = original_error
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization"""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.name,
            "timestamp": self.timestamp.isoformat(),
            "context": {
                "component": self.context.component,
                "operation": self.context.operation,
                "user_id": self.context.user_id,
                "session_id": self.context.session_id,
                "request_id": self.context.request_id,
                "additional_info": self.context.additional_info
            },
            "stack_trace": traceback.format_exc() if self.original_error else None
        }

class ConfigurationError(AIScientistError):
    """Configuration-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            **kwargs
        )

class APIError(AIScientistError):
    """External API errors"""
    def __init__(self, message: str, status_code: int = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.API,
            **kwargs
        )
        self.status_code = status_code

class SecurityError(AIScientistError):
    """Security-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )

class ValidationError(AIScientistError):
    """Data validation errors"""
    def __init__(self, message: str, field: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            **kwargs
        )
        self.field = field

class NetworkError(AIScientistError):
    """Network-related errors"""
    def __init__(self, message: str, endpoint: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            **kwargs
        )
        self.endpoint = endpoint

class ResourceError(AIScientistError):
    """Resource-related errors (memory, disk, etc.)"""
    def __init__(self, message: str, resource_type: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )
        self.resource_type = resource_type

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._failure_count = 0
        self._last_failure_time = None
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper

    def execute(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self._state == "OPEN":
                if self._should_attempt_reset():
                    self._state = "HALF_OPEN"
                else:
                    raise AIScientistError(
                        f"Circuit breaker is OPEN for {func.__name__}",
                        category=ErrorCategory.NETWORK,
                        severity=ErrorSeverity.WARNING
                    )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self._last_failure_time is None:
            return True

        time_since_failure = (datetime.utcnow() - self._last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout

    def _on_success(self):
        """Handle successful execution"""
        with self._lock:
            self._failure_count = 0
            self._state = "CLOSED"

    def _on_failure(self):
        """Handle failed execution"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.utcnow()

            if self._failure_count >= self.failure_threshold:
                self._state = "OPEN"

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        with self._lock:
            return {
                "state": self._state,
                "failure_count": self._failure_count,
                "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None
            }

class RetryManager:
    """Retry mechanism with exponential backoff"""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper

    def execute(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic"""
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt == self.max_attempts - 1:
                    break

                # Calculate delay
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_attempts} failed for {func.__name__}. "
                    f"Retrying in {delay:.2f}s. Error: {str(e)}"
                )
                time.sleep(delay)

        # All attempts failed
        raise AIScientistError(
            f"Function {func.__name__} failed after {self.max_attempts} attempts. Last error: {str(last_exception)}",
            original_error=last_exception
        )

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            delay *= (0.5 + (0.5 * time.time() % 1))  # Random jitter

        return delay

class ErrorTracker:
    """Error tracking and metrics collection"""

    def __init__(self):
        self._metrics = ErrorMetrics()
        self._lock = threading.Lock()

    def track_error(self, error: AIScientistError):
        """Track an error for metrics collection"""
        with self._lock:
            self._metrics.total_errors += 1
            self._metrics.errors_by_category[error.category] += 1
            self._metrics.errors_by_severity[error.severity] += 1

            # Add to recent errors (keep last 100)
            error_dict = error.to_dict()
            self._metrics.recent_errors.append(error_dict)
            if len(self._metrics.recent_errors) > 100:
                self._metrics.recent_errors.pop(0)

            # Update error rates
            component = error.context.component or "unknown"
            self._metrics.error_rates[component] = self._calculate_error_rate(component)

    def _calculate_error_rate(self, component: str) -> float:
        """Calculate error rate for a component"""
        # Simple calculation: errors in last hour / total requests
        # This would be enhanced with actual request tracking
        recent_errors = [
            e for e in self._metrics.recent_errors
            if e["context"]["component"] == component
        ]
        return len(recent_errors) / 3600.0  # errors per second

    def get_metrics(self) -> Dict[str, Any]:
        """Get current error metrics"""
        with self._lock:
            return {
                "total_errors": self._metrics.total_errors,
                "errors_by_category": {
                    cat.value: count for cat, count in self._metrics.errors_by_category.items()
                },
                "errors_by_severity": {
                    sev.name: count for sev, count in self._metrics.errors_by_severity.items()
                },
                "recent_errors_count": len(self._metrics.recent_errors),
                "error_rates": dict(self._metrics.error_rates)
            }

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        with self._lock:
            if not self._metrics.recent_errors:
                return {"message": "No recent errors"}

            # Group errors by category
            by_category = defaultdict(list)
            for error in self._metrics.recent_errors:
                by_category[error["category"]].append(error)

            return {
                "total_recent_errors": len(self._metrics.recent_errors),
                "by_category": {
                    cat: len(errors) for cat, errors in by_category.items()
                },
                "most_common_error": max(
                    by_category.items(),
                    key=lambda x: len(x[1])
                )[0] if by_category else None
            }

# Global error tracking instance
_error_tracker = ErrorTracker()

def handle_errors(
    reraise: bool = True,
    default_return: Any = None,
    log_level: int = logging.ERROR,
    categories: Optional[List[ErrorCategory]] = None
):
    """Decorator for comprehensive error handling"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Convert to AIScientistError if needed
                if not isinstance(e, AIScientistError):
                    error = AIScientistError(
                        f"Unexpected error in {func.__name__}: {str(e)}",
                        original_error=e
                    )
                else:
                    error = e

                # Filter by category if specified
                if categories and error.category not in categories:
                    raise

                # Track error
                _error_tracker.track_error(error)

                # Log error
                logger.log(
                    log_level,
                    f"Error in {func.__name__}: {error.message}",
                    extra={"error": error.to_dict()}
                )

                if reraise:
                    raise
                return default_return

        return wrapper
    return decorator

def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """Convenience decorator for retry with backoff"""
    retry_manager = RetryManager(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay
    )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return retry_manager.execute(func, *args, **kwargs)
        return wrapper
    return decorator

def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    exceptions: tuple = (Exception,)
):
    """Convenience decorator for circuit breaker"""
    cb = CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=exceptions
    )

    def decorator(func):
        return cb(func)
    return decorator

def get_error_metrics() -> Dict[str, Any]:
    """Get current error metrics"""
    return _error_tracker.get_metrics()

def get_error_summary() -> Dict[str, Any]:
    """Get summary of recent errors"""
    return _error_tracker.get_error_summary()

def graceful_degradation(fallback_func: Callable):
    """Decorator for graceful degradation with fallback"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Primary function {func.__name__} failed, using fallback. Error: {str(e)}"
                )
                try:
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback function also failed: {str(fallback_error)}")
                    raise AIScientistError(
                        f"Both primary and fallback functions failed for {func.__name__}",
                        original_error=fallback_error
                    )
        return wrapper
    return decorator

# Context manager for error handling
class ErrorHandler:
    """Context manager for error handling"""

    def __init__(self, operation: str, component: str = None, reraise: bool = False):
        self.operation = operation
        self.component = component
        self.reraise = reraise
        self.error = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error = AIScientistError(
                f"Error in {self.operation}: {str(exc_val)}",
                component=self.component,
                original_error=exc_val
            )

            _error_tracker.track_error(self.error)

            logger.error(
                f"Error in {self.operation}: {str(exc_val)}",
                extra={"error": self.error.to_dict()}
            )

            if self.reraise:
                return False  # Re-raise the exception
            return True  # Suppress the exception

        return True

# Async error handling
class AsyncErrorHandler:
    """Async context manager for error handling"""

    def __init__(self, operation: str, component: str = None, reraise: bool = False):
        self.operation = operation
        self.component = component
        self.reraise = reraise
        self.error = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error = AIScientistError(
                f"Error in {self.operation}: {str(exc_val)}",
                component=self.component,
                original_error=exc_val
            )

            _error_tracker.track_error(self.error)

            logger.error(
                f"Error in {self.operation}: {str(exc_val)}",
                extra={"error": self.error.to_dict()}
            )

            if self.reraise:
                return False
            return True

        return True