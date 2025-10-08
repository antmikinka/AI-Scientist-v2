"""
Security Management System for AI-Scientist-v2

Provides secure API key management, encryption, access control,
and audit logging for the entire system.
"""

import os
import logging
import json
import hashlib
import hmac
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import secrets
import threading
from functools import wraps

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    require_encryption: bool = True
    min_password_length: int = 12
    session_timeout_minutes: int = 60
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    audit_log_retention_days: int = 90
    api_key_rotation_days: int = 90
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 100

@dataclass
class APIKeyInfo:
    """API key metadata and management"""
    key_id: str
    key_hash: str
    created_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0
    is_active: bool = True
    permissions: List[str] = field(default_factory=list)
    rate_limit: Optional[int] = None
    description: Optional[str] = None

class SecurityManager:
    """
    Comprehensive security management for API keys, encryption,
    and access control.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path.home() / ".ai_scientist" / "security"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Security state
        self._api_keys: Dict[str, APIKeyInfo] = {}
        self._session_tokens: Dict[str, Dict[str, Any]] = {}
        self._rate_limit_counters: Dict[str, List[datetime]] = {}
        self._lockout_timers: Dict[str, datetime] = {}

        # Thread safety
        self._lock = threading.Lock()

        # Encryption setup
        self._encryption_key = self._load_or_create_encryption_key()
        self._fernet = None
        if CRYPTOGRAPHY_AVAILABLE and self._encryption_key:
            self._fernet = Fernet(self._encryption_key)

        # Load existing API keys
        self._load_api_keys()

        # Initialize audit logging
        self._audit_logger = self._setup_audit_logging()

    def _load_or_create_encryption_key(self) -> Optional[bytes]:
        """Load or create encryption key for sensitive data."""
        key_file = self.config_dir / "encryption.key"

        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to load encryption key: {e}")
                return None

        # Generate new key
        if CRYPTOGRAPHY_AVAILABLE:
            try:
                key = Fernet.generate_key()
                # Secure the key file permissions
                key_file.touch(mode=0o600)
                with open(key_file, 'wb') as f:
                    f.write(key)
                logger.info("Generated new encryption key")
                return key
            except Exception as e:
                logger.error(f"Failed to generate encryption key: {e}")
                return None

        return None

    def _setup_audit_logging(self):
        """Setup audit logging for security events."""
        audit_log_file = self.config_dir / "audit.log"

        # Create audit logger
        audit_logger = logging.getLogger('ai_scientist.audit')
        audit_logger.setLevel(logging.INFO)

        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler(
            audit_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )

        # JSON formatter for structured logging
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"event": "%(message)s", "source": "%(name)s"}'
        )
        handler.setFormatter(formatter)
        audit_logger.addHandler(handler)

        return audit_logger

    def _log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log security audit event."""
        event = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            **details
        }
        self._audit_logger.info(json.dumps(event))

    def generate_api_key(self, description: str = None,
                        permissions: List[str] = None,
                        rate_limit: int = None) -> tuple[str, str]:
        """
        Generate a new API key.

        Returns:
            Tuple of (key_id, api_key)
        """
        # Generate secure random key
        key_id = f"ask_{secrets.token_urlsafe(16)}"
        api_key = f"sk-or-{secrets.token_urlsafe(32)}"

        # Hash the key for storage
        key_hash = self._hash_api_key(api_key)

        with self._lock:
            api_key_info = APIKeyInfo(
                key_id=key_id,
                key_hash=key_hash,
                created_at=datetime.utcnow(),
                permissions=permissions or [],
                rate_limit=rate_limit,
                description=description
            )
            self._api_keys[key_id] = api_key_info
            self._save_api_keys()

        self._log_audit_event("api_key_created", {
            "key_id": key_id,
            "description": description,
            "permissions": permissions
        })

        logger.info(f"Generated new API key: {key_id}")
        return key_id, api_key

    def validate_api_key(self, api_key: str) -> Optional[APIKeyInfo]:
        """Validate API key and return key info if valid."""
        key_hash = self._hash_api_key(api_key)

        with self._lock:
            for key_info in self._api_keys.values():
                if key_info.key_hash == key_hash and key_info.is_active:
                    # Update usage stats
                    key_info.last_used = datetime.utcnow()
                    key_info.usage_count += 1
                    self._save_api_keys()

                    self._log_audit_event("api_key_used", {
                        "key_id": key_info.key_id,
                        "usage_count": key_info.usage_count
                    })

                    return key_info

        return None

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        with self._lock:
            if key_id in self._api_keys:
                self._api_keys[key_id].is_active = False
                self._save_api_keys()

                self._log_audit_event("api_key_revoked", {
                    "key_id": key_id
                })

                logger.info(f"Revoked API key: {key_id}")
                return True

        return False

    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def _load_api_keys(self):
        """Load API keys from secure storage."""
        keys_file = self.config_dir / "api_keys.json"

        if not keys_file.exists():
            return

        try:
            with open(keys_file, 'r') as f:
                data = json.load(f)

            with self._lock:
                for key_data in data.get('api_keys', []):
                    key_info = APIKeyInfo(
                        key_id=key_data['key_id'],
                        key_hash=key_data['key_hash'],
                        created_at=datetime.fromisoformat(key_data['created_at']),
                        last_used=datetime.fromisoformat(key_data['last_used']) if key_data.get('last_used') else None,
                        usage_count=key_data.get('usage_count', 0),
                        is_active=key_data.get('is_active', True),
                        permissions=key_data.get('permissions', []),
                        rate_limit=key_data.get('rate_limit'),
                        description=key_data.get('description')
                    )
                    self._api_keys[key_info.key_id] = key_info

        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")

    def _save_api_keys(self):
        """Save API keys to secure storage."""
        keys_file = self.config_dir / "api_keys.json"

        try:
            with self._lock:
                data = {
                    'api_keys': [
                        {
                            'key_id': info.key_id,
                            'key_hash': info.key_hash,
                            'created_at': info.created_at.isoformat(),
                            'last_used': info.last_used.isoformat() if info.last_used else None,
                            'usage_count': info.usage_count,
                            'is_active': info.is_active,
                            'permissions': info.permissions,
                            'rate_limit': info.rate_limit,
                            'description': info.description
                        }
                        for info in self._api_keys.values()
                    ]
                }

                # Write to temporary file first, then move
                temp_file = keys_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(data, f, indent=2)

                # Secure file permissions
                temp_file.chmod(0o600)
                temp_file.replace(keys_file)

        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")

    def encrypt_sensitive_data(self, data: Union[str, bytes]) -> Optional[str]:
        """Encrypt sensitive data for storage."""
        if not self._fernet:
            logger.warning("Encryption not available")
            return None

        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            encrypted = self._fernet.encrypt(data)
            return base64.b64encode(encrypted).decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return None

    def decrypt_sensitive_data(self, encrypted_data: str) -> Optional[str]:
        """Decrypt sensitive data."""
        if not self._fernet:
            logger.warning("Decryption not available")
            return None

        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted = self._fernet.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None

    def check_rate_limit(self, client_id: str, limit: int = None) -> bool:
        """Check if client has exceeded rate limit."""
        if not limit:
            limit = 100  # Default limit

        now = datetime.utcnow()
        with self._lock:
            if client_id not in self._rate_limit_counters:
                self._rate_limit_counters[client_id] = []

            # Clean old entries (older than 1 minute)
            self._rate_limit_counters[client_id] = [
                timestamp for timestamp in self._rate_limit_counters[client_id]
                if (now - timestamp).total_seconds() < 60
            ]

            # Check if limit exceeded
            if len(self._rate_limit_counters[client_id]) >= limit:
                self._log_audit_event("rate_limit_exceeded", {
                    "client_id": client_id,
                    "request_count": len(self._rate_limit_counters[client_id])
                })
                return False

            # Add current request
            self._rate_limit_counters[client_id].append(now)
            return True

    def check_login_attempts(self, username: str) -> bool:
        """Check if user is locked out due to failed attempts."""
        with self._lock:
            if username in self._lockout_timers:
                if datetime.utcnow() < self._lockout_timers[username]:
                    return False
                else:
                    del self._lockout_timers[username]

        return True

    def record_failed_login(self, username: str):
        """Record failed login attempt."""
        # This would be expanded with actual attempt counting
        self._log_audit_event("failed_login_attempt", {
            "username": username
        })

    def rotate_api_key(self, key_id: str) -> Optional[tuple[str, str]]:
        """Rotate an existing API key."""
        with self._lock:
            if key_id not in self._api_keys:
                return None

            old_key_info = self._api_keys[key_id]

            # Generate new key
            new_key_id, new_api_key = self.generate_api_key(
                description=old_key_info.description,
                permissions=old_key_info.permissions,
                rate_limit=old_key_info.rate_limit
            )

            # Revoke old key
            self.revoke_api_key(key_id)

            return new_key_id, new_api_key

    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status and metrics."""
        with self._lock:
            active_keys = sum(1 for info in self._api_keys.values() if info.is_active)
            total_usage = sum(info.usage_count for info in self._api_keys.values())

            return {
                "encryption_enabled": self._fernet is not None,
                "active_api_keys": active_keys,
                "total_api_key_usage": total_usage,
                "rate_limited_clients": len(self._rate_limit_counters),
                "locked_out_users": len(self._lockout_timers),
                "security_config_dir": str(self.config_dir),
                "audit_logging_enabled": True
            }

# Security decorator for API endpoints
def require_api_key(permission: str = None):
    """Decorator to require valid API key for endpoint access."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract API key from request (implementation depends on framework)
            api_key = _extract_api_key_from_request()

            if not api_key:
                raise SecurityError("API key required")

            security_manager = get_security_manager()
            key_info = security_manager.validate_api_key(api_key)

            if not key_info:
                raise SecurityError("Invalid API key")

            if permission and permission not in key_info.permissions:
                raise SecurityError(f"Insufficient permissions: {permission} required")

            # Check rate limit
            client_id = key_info.key_id
            rate_limit = key_info.rate_limit or 100
            if not security_manager.check_rate_limit(client_id, rate_limit):
                raise SecurityError("Rate limit exceeded")

            return func(*args, **kwargs)
        return wrapper
    return decorator

def _extract_api_key_from_request() -> Optional[str]:
    """Extract API key from request context."""
    # This would be implemented based on the web framework being used
    # For now, check environment variable for testing
    return os.getenv("OPENROUTER_API_KEY")

# Global security manager instance
_security_manager = None

def get_security_manager() -> SecurityManager:
    """Get the global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager

# Security validation functions
def validate_api_key_format(api_key: str) -> bool:
    """Validate API key format."""
    if not api_key:
        return False

    # Basic format validation
    if len(api_key) < 32:
        return False

    # Check for common patterns
    if not api_key.startswith(('sk-or-', 'ask_')):
        return False

    return True

def sanitize_sensitive_data(data: Dict[str, Any], sensitive_keys: List[str]) -> Dict[str, Any]:
    """Remove sensitive data from dictionaries for logging."""
    sanitized = data.copy()
    for key in sensitive_keys:
        if key in sanitized:
            sanitized[key] = "***REDACTED***"
    return sanitized

# Security audit helper functions
def audit_security_event(event_type: str, details: Dict[str, Any]):
    """Convenience function to log security audit events."""
    security_manager = get_security_manager()
    security_manager._log_audit_event(event_type, details)

def check_security_health() -> Dict[str, Any]:
    """Perform security health check."""
    security_manager = get_security_manager()
    status = security_manager.get_security_status()

    # Additional health checks
    health_issues = []

    # Check encryption
    if not status["encryption_enabled"]:
        health_issues.append("Encryption not available - install cryptography")

    # Check API key rotation
    # This would check for old API keys that need rotation

    # Check file permissions
    if not status["security_config_dir"]:
        health_issues.append("Security configuration directory not accessible")

    return {
        "status": status,
        "health_issues": health_issues,
        "overall_health": len(health_issues) == 0
    }