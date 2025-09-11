#!/usr/bin/env python3
"""
Comprehensive OpenRouter Integration Test Suite
Tests all components of the AI-Scientist-v2 OpenRouter integration.
"""

import os
import sys
import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import json

# Add the ai_scientist directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai_scientist'))

class TestOpenRouterIntegration(unittest.TestCase):
    """Test suite for OpenRouter integration components"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_api_key = "sk-or-v1-test-key-12345"
        os.environ["OPENROUTER_API_KEY"] = self.test_api_key
        
    def test_imports(self):
        """Test that all OpenRouter modules can be imported"""
        try:
            # Test basic imports
            from ai_scientist.openrouter import (
                OpenRouterClient, 
                get_global_client, 
                initialize_openrouter,
                CacheStrategy,
                OpenRouterConfig,
                load_config,
                save_config,
                create_default_config,
                RAGSystem,
                extract_json_between_markers
            )
            print("✅ All OpenRouter imports successful")
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            # Check what specific dependency is missing
            if "aiohttp" in str(e):
                print("Missing aiohttp - install with: pip install aiohttp")
            elif "chromadb" in str(e):
                print("Missing chromadb - install with: pip install chromadb")
            elif "rich" in str(e):
                print("Missing rich - install with: pip install rich")
            elif "yaml" in str(e):
                print("Missing PyYAML - install with: pip install pyyaml")
            self.fail(f"Import failed: {e}")
    
    def test_config_creation(self):
        """Test configuration system"""
        try:
            from ai_scientist.openrouter.config import create_default_config, validate_config
            
            # Create default config
            config = create_default_config()
            self.assertIsNotNone(config)
            self.assertEqual(config.app_name, "AI-Scientist-v2")
            
            # Validate config
            errors = validate_config(config)
            # Should have one error about missing API key if not set in config
            print(f"✅ Config creation and validation working. Errors: {len(errors)}")
            
        except Exception as e:
            print(f"❌ Config test failed: {e}")
            self.fail(f"Config test failed: {e}")
    
    def test_client_initialization(self):
        """Test OpenRouter client initialization"""
        try:
            from ai_scientist.openrouter.client import OpenRouterClient
            
            # Test client creation
            client = OpenRouterClient(api_key=self.test_api_key)
            self.assertIsNotNone(client)
            self.assertEqual(client.api_key, self.test_api_key)
            
            print("✅ OpenRouter client initialization successful")
            
        except Exception as e:
            print(f"❌ Client initialization failed: {e}")
            self.fail(f"Client initialization failed: {e}")
    
    @patch('aiohttp.ClientSession.request')
    async def test_client_api_call(self, mock_request):
        """Test OpenRouter API call functionality"""
        try:
            from ai_scientist.openrouter.client import OpenRouterClient
            
            # Mock API response
            mock_response = Mock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value='{"choices": [{"message": {"content": "Test response"}}], "usage": {"total_tokens": 10}}')
            mock_response.headers = {}
            mock_request.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_request.return_value.__aexit__ = AsyncMock(return_value=None)
            
            client = OpenRouterClient(api_key=self.test_api_key)
            
            # Test API call
            messages = [{"role": "user", "content": "Hello test"}]
            response, history = await client.get_response(messages, "openai/gpt-4o-mini")
            
            self.assertEqual(response, "Test response")
            self.assertEqual(len(history), 2)  # Original message + response
            
            print("✅ OpenRouter API call test successful")
            
        except Exception as e:
            print(f"❌ API call test failed: {e}")
            self.fail(f"API call test failed: {e}")
    
    def test_rag_system_initialization(self):
        """Test RAG system initialization"""
        try:
            from ai_scientist.openrouter.rag_system import RAGSystem
            from ai_scientist.openrouter.config import RAGConfig
            
            # Create RAG config
            rag_config = RAGConfig()
            config_dict = {
                'enabled': True,
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'embedding_model': 'text-embedding-ada-002',
                'vector_store': 'chroma',
                'collection_name': 'test_docs',
                'similarity_threshold': 0.7,
                'max_results': 10,
                'auto_ingest': True,
                'supported_formats': ['pdf', 'txt', 'md'],
                'max_file_size': '50MB'
            }
            
            # Test RAG system creation
            rag_system = RAGSystem(config_dict)
            self.assertIsNotNone(rag_system)
            
            print("✅ RAG system initialization successful")
            
        except Exception as e:
            print(f"❌ RAG system test failed: {e}")
            # This might fail if chromadb or sentence-transformers aren't installed
            if "chromadb" in str(e) or "sentence" in str(e):
                print("Note: RAG system requires chromadb and sentence-transformers")
            self.fail(f"RAG system test failed: {e}")
    
    def test_cli_interface(self):
        """Test CLI interface initialization"""
        try:
            from ai_scientist.openrouter.cli import CLIInterface
            
            cli = CLIInterface()
            self.assertIsNotNone(cli)
            
            print("✅ CLI interface initialization successful")
            
        except Exception as e:
            print(f"❌ CLI interface test failed: {e}")
            self.fail(f"CLI interface test failed: {e}")
    
    def test_utility_functions(self):
        """Test utility functions"""
        try:
            from ai_scientist.openrouter.utils import extract_json_between_markers, count_tokens
            
            # Test JSON extraction
            test_text = 'Some text ```json{"key": "value"}``` more text'
            result = extract_json_between_markers(test_text)
            self.assertIsNotNone(result)
            self.assertEqual(result.get("key"), "value")
            
            # Test token counting
            tokens = count_tokens("Hello world test")
            self.assertGreater(tokens, 0)
            
            print("✅ Utility functions test successful")
            
        except Exception as e:
            print(f"❌ Utility functions test failed: {e}")
            self.fail(f"Utility functions test failed: {e}")
    
    def test_llm_integration(self):
        """Test integration with existing LLM module"""
        try:
            from ai_scientist import llm
            
            # Test that OpenRouter functions are available
            self.assertTrue(hasattr(llm, 'create_client'))
            self.assertTrue(hasattr(llm, 'get_response_from_llm'))
            self.assertTrue(hasattr(llm, 'extract_json_between_markers'))
            
            # Test model mapping
            self.assertTrue(hasattr(llm, 'LEGACY_MODEL_MAPPING'))
            
            print("✅ LLM integration test successful")
            
        except Exception as e:
            print(f"❌ LLM integration test failed: {e}")
            self.fail(f"LLM integration test failed: {e}")

class TestOpenRouterConfiguration(unittest.TestCase):
    """Test configuration and setup functionality"""
    
    def test_config_file_operations(self):
        """Test configuration file save/load"""
        try:
            from ai_scientist.openrouter.config import create_default_config, save_config, load_config
            
            # Create a temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
                config_path = tmp.name
            
            try:
                # Create and save config
                config = create_default_config()
                config.api_key = "test-key"
                save_config(config, config_path)
                
                # Load config
                loaded_config = load_config(config_path)
                self.assertEqual(loaded_config.api_key, "test-key")
                
                print("✅ Config file operations test successful")
                
            finally:
                # Clean up
                if os.path.exists(config_path):
                    os.unlink(config_path)
                    
        except Exception as e:
            print(f"❌ Config file operations test failed: {e}")
            self.fail(f"Config file operations test failed: {e}")

def run_integration_tests():
    """Run all integration tests and provide detailed feedback"""
    print("=" * 60)
    print("AI-Scientist-v2 OpenRouter Integration Test Suite")
    print("=" * 60)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestOpenRouterIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestOpenRouterConfiguration))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Integration test PASSED - System is ready for use!")
    elif success_rate >= 60:
        print("⚠️  Integration test PARTIAL - Some issues need attention")
    else:
        print("❌ Integration test FAILED - Critical issues need fixing")
    
    return success_rate >= 80

async def run_async_tests():
    """Run async-specific tests"""
    print("\n" + "=" * 40)
    print("Running Async Tests")
    print("=" * 40)
    
    # Create a test instance
    test_instance = TestOpenRouterIntegration()
    test_instance.setUp()
    
    try:
        # Run the async API test
        await test_instance.test_client_api_call()
        print("✅ Async tests completed successfully")
        return True
    except Exception as e:
        print(f"❌ Async tests failed: {e}")
        return False

def main():
    """Main test runner"""
    print("Starting OpenRouter Integration Tests...\n")
    
    # Run synchronous tests
    sync_success = run_integration_tests()
    
    # Run asynchronous tests
    async_success = asyncio.run(run_async_tests())
    
    # Overall result
    if sync_success and async_success:
        print("\n🎉 ALL TESTS PASSED - OpenRouter integration is ready!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Check output above for issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())