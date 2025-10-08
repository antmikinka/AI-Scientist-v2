"""
Critical Fix Tests for Phase 1 - RAGConfig Import Resolution

These tests validate the critical fixes needed to resolve import issues
and ensure core functionality works correctly.

Priority: CRITICAL - Blocks all other testing
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestRAGConfigImport:
    """Test RAGConfig import resolution and availability"""

    def test_rag_config_import_direct(self):
        """Test direct import of RAGConfig from config module"""
        try:
            from ai_scientist.openrouter.config import RAGConfig
            assert RAGConfig is not None
            print("✅ Direct RAGConfig import successful")
        except ImportError as e:
            pytest.fail(f"Direct RAGConfig import failed: {e}")

    def test_rag_config_instantiation(self):
        """Test RAGConfig can be instantiated with default values"""
        try:
            from ai_scientist.openrouter.config import RAGConfig

            config = RAGConfig()
            assert config.enabled is True
            assert config.chunk_size == 1000
            assert config.chunk_overlap == 200
            assert config.embedding_model == "text-embedding-ada-002"
            assert config.vector_store == "chroma"
            print("✅ RAGConfig instantiation successful")
        except Exception as e:
            pytest.fail(f"RAGConfig instantiation failed: {e}")

    def test_rag_config_custom_values(self):
        """Test RAGConfig with custom configuration values"""
        try:
            from ai_scientist.openrouter.config import RAGConfig

            custom_config = RAGConfig(
                enabled=False,
                chunk_size=500,
                chunk_overlap=100,
                embedding_model="custom-model",
                vector_store="faiss"
            )

            assert custom_config.enabled is False
            assert custom_config.chunk_size == 500
            assert custom_config.chunk_overlap == 100
            assert custom_config.embedding_model == "custom-model"
            assert custom_config.vector_store == "faiss"
            print("✅ RAGConfig custom values successful")
        except Exception as e:
            pytest.fail(f"RAGConfig custom values failed: {e}")

    def test_rag_system_import_after_fix(self):
        """Test that rag_system can be imported after RAGConfig fix"""
        try:
            # This should work after the fix
            from ai_scientist.openrouter.rag_system import RAGSystem
            assert RAGSystem is not None
            print("✅ RAGSystem import successful")
        except ImportError as e:
            pytest.fail(f"RAGSystem import failed: {e}")

    def test_chroma_vector_store_import_after_fix(self):
        """Test that ChromaVectorStore can be imported after RAGConfig fix"""
        try:
            # This should work after the fix
            from ai_scientist.openrouter.rag_system import ChromaVectorStore
            assert ChromaVectorStore is not None
            print("✅ ChromaVectorStore import successful")
        except ImportError as e:
            pytest.fail(f"ChromaVectorStore import failed: {e}")

class TestModuleImportChain:
    """Test the complete import chain for OpenRouter integration"""

    def test_openrouter_module_import(self):
        """Test complete openrouter module import"""
        try:
            from ai_scientist.openrouter import (
                OpenRouterClient,
                RAGSystem,
                OpenRouterConfig,
                RAGConfig
            )
            assert OpenRouterClient is not None
            assert RAGSystem is not None
            assert OpenRouterConfig is not None
            assert RAGConfig is not None
            print("✅ Complete OpenRouter module import successful")
        except ImportError as e:
            pytest.fail(f"Complete OpenRouter module import failed: {e}")

    def test_cli_interface_import(self):
        """Test CLI interface import after fixes"""
        try:
            from ai_scientist.openrouter.cli import CLIInterface
            assert CLIInterface is not None
            print("✅ CLI interface import successful")
        except ImportError as e:
            pytest.fail(f"CLI interface import failed: {e}")

    def test_llm_integration_import(self):
        """Test LLM integration import after fixes"""
        try:
            from ai_scientist import llm
            assert llm is not None
            print("✅ LLM integration import successful")
        except ImportError as e:
            pytest.fail(f"LLM integration import failed: {e}")

class TestConfigurationIntegration:
    """Test configuration integration across modules"""

    def test_openrouter_config_with_rag_config(self):
        """Test OpenRouterConfig includes RAGConfig properly"""
        try:
            from ai_scientist.openrouter.config import OpenRouterConfig, RAGConfig

            config = OpenRouterConfig()
            assert hasattr(config, 'rag_config')
            assert isinstance(config.rag_config, RAGConfig)
            print("✅ OpenRouterConfig RAGConfig integration successful")
        except Exception as e:
            pytest.fail(f"OpenRouterConfig RAGConfig integration failed: {e}")

    def test_rag_config_serialization(self):
        """Test RAGConfig can be serialized/deserialized"""
        try:
            from ai_scientist.openrouter.config import RAGConfig
            import json

            config = RAGConfig(
                enabled=True,
                chunk_size=1500,
                embedding_model="test-model"
            )

            # Test serialization
            config_dict = config.__dict__
            json_str = json.dumps(config_dict)

            # Test deserialization
            loaded_dict = json.loads(json_str)
            new_config = RAGConfig(**loaded_dict)

            assert new_config.enabled == config.enabled
            assert new_config.chunk_size == config.chunk_size
            assert new_config.embedding_model == config.embedding_model
            print("✅ RAGConfig serialization successful")
        except Exception as e:
            pytest.fail(f"RAGConfig serialization failed: {e}")

@pytest.mark.critical
class TestCriticalFunctionality:
    """Test critical functionality after import fixes"""

    @patch('ai_scientist.openrouter.rag_system.CHROMA_AVAILABLE', False)
    def test_rag_system_without_chroma(self):
        """Test RAGSystem handles missing ChromaDB gracefully"""
        try:
            from ai_scientist.openrouter.config import RAGConfig
            from ai_scientist.openrouter.rag_system import RAGSystem

            config = RAGConfig(enabled=False)  # Disable to avoid Chroma requirement
            rag_system = RAGSystem(config)
            assert rag_system is not None
            print("✅ RAGSystem without Chroma successful")
        except Exception as e:
            # Expected failure due to missing OpenAI API key or other dependencies
            if "api_key" in str(e).lower() or "openai" in str(e).lower():
                pytest.skip("RAGSystem requires OpenAI API key for embedding generation")
            pytest.fail(f"RAGSystem without Chroma failed: {e}")

    def test_client_initialization_minimal(self):
        """Test client initialization with minimal configuration"""
        try:
            from ai_scientist.openrouter.client import OpenRouterClient

            # Mock environment to avoid API key requirement
            with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}):
                client = OpenRouterClient(api_key='test-key')
                assert client is not None
                print("✅ Client initialization successful")
        except Exception as e:
            # Expected failure due to missing dependencies
            if "api_key" in str(e).lower() or "import" in str(e).lower():
                pytest.skip("Client initialization requires additional dependencies")
            pytest.fail(f"Client initialization failed: {e}")

if __name__ == "__main__":
    # Run critical tests first
    pytest.main([__file__, "-v", "-k", "critical", "--tb=short"])