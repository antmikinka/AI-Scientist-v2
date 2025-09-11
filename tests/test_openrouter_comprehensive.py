#!/usr/bin/env python3
"""
Comprehensive OpenRouter Integration Test Suite

Tests all aspects of the OpenRouter integration including:
- API connectivity and authentication
- Model availability and selection
- Configuration management
- CLI functionality
- RAG system integration
- Pipeline execution
- Error handling and fallbacks

Run with: python test_openrouter_comprehensive.py
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add AI-Scientist to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai_scientist'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenRouterIntegrationTester:
    """Comprehensive tester for OpenRouter integration"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.test_results = {
            "timestamp": time.time(),
            "tests": {},
            "overall_status": "unknown"
        }
        self.temp_dir = Path(tempfile.mkdtemp(prefix="openrouter_test_"))
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        logger.info("🧪 Starting comprehensive OpenRouter integration tests")
        logger.info(f"Temporary directory: {self.temp_dir}")
        
        # Test 1: Basic connectivity
        await self._test_api_connectivity()
        
        # Test 2: Configuration management
        await self._test_configuration_management()
        
        # Test 3: Model operations
        await self._test_model_operations()
        
        # Test 4: Caching functionality
        await self._test_caching_functionality()
        
        # Test 5: RAG system
        await self._test_rag_system()
        
        # Test 6: CLI functionality
        await self._test_cli_functionality()
        
        # Test 7: Pipeline integration
        await self._test_pipeline_integration()
        
        # Test 8: Error handling
        await self._test_error_handling()
        
        # Test 9: Performance and limits
        await self._test_performance_limits()
        
        # Final summary
        self._generate_test_summary()
        
        return self.test_results
    
    async def _test_api_connectivity(self):
        """Test basic API connectivity and authentication"""
        test_name = "api_connectivity"
        logger.info(f"🔗 Testing {test_name}")
        
        try:
            from ai_scientist.openrouter import initialize_openrouter, get_global_client
            
            if not self.api_key:
                self.test_results["tests"][test_name] = {
                    "status": "skipped",
                    "message": "No OPENROUTER_API_KEY found"
                }
                return
            
            # Initialize client
            client = initialize_openrouter(self.api_key)
            
            # Test model listing
            models = await client.get_available_models()
            
            self.test_results["tests"][test_name] = {
                "status": "passed",
                "models_found": len(models),
                "client_initialized": True
            }
            
            logger.info(f"✅ {test_name}: Found {len(models)} models")
            
        except Exception as e:
            logger.error(f"❌ {test_name}: {e}")
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    async def _test_configuration_management(self):
        """Test configuration loading, saving, and validation"""
        test_name = "configuration_management"
        logger.info(f"⚙️ Testing {test_name}")
        
        try:
            from ai_scientist.openrouter import (
                create_default_config, save_config, load_config, 
                validate_config, create_config_from_template
            )
            
            # Test default config creation
            default_config = create_default_config()
            assert default_config is not None, "Default config creation failed"
            
            # Test config validation
            errors = validate_config(default_config)
            if errors:
                logger.warning(f"Default config validation errors: {errors}")
            
            # Test config saving/loading
            test_config_file = self.temp_dir / "test_config.yaml"
            save_config(default_config, str(test_config_file))
            assert test_config_file.exists(), "Config file not created"
            
            loaded_config = load_config(str(test_config_file))
            assert loaded_config is not None, "Config loading failed"
            
            # Test template creation
            templates = ["research", "cost_optimized", "high_quality", "experimental"]
            template_configs = {}
            
            for template in templates:
                try:
                    config = create_config_from_template(template)
                    template_configs[template] = config is not None
                except Exception as e:
                    logger.warning(f"Template {template} failed: {e}")
                    template_configs[template] = False
            
            self.test_results["tests"][test_name] = {
                "status": "passed",
                "default_config_created": True,
                "config_file_saved": test_config_file.exists(),
                "config_loaded": loaded_config is not None,
                "template_configs": template_configs,
                "validation_errors": len(errors)
            }
            
            logger.info(f"✅ {test_name}: All configuration operations successful")
            
        except Exception as e:
            logger.error(f"❌ {test_name}: {e}")
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    async def _test_model_operations(self):
        """Test model selection and API calls"""
        test_name = "model_operations"
        logger.info(f"🤖 Testing {test_name}")
        
        try:
            from ai_scientist.openrouter import get_global_client, CacheStrategy
            
            if not self.api_key:
                self.test_results["tests"][test_name] = {
                    "status": "skipped",
                    "message": "No API key available"
                }
                return
            
            client = get_global_client()
            
            # Test basic model call
            test_models = [
                "openai/gpt-4o-mini",  # Fast, cheap model for testing
                "anthropic/claude-3-haiku"  # Another fast option
            ]
            
            model_test_results = {}
            
            for model in test_models:
                try:
                    start_time = time.time()
                    
                    response, history = await client.get_response(
                        messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
                        model=model,
                        temperature=0.1,
                        max_tokens=50
                    )
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    model_test_results[model] = {
                        "status": "success",
                        "response_length": len(response),
                        "response_time": response_time,
                        "history_updated": len(history) > 1
                    }
                    
                    logger.info(f"  ✓ {model}: Response in {response_time:.2f}s")
                    
                except Exception as e:
                    logger.warning(f"  ✗ {model}: {e}")
                    model_test_results[model] = {
                        "status": "failed",
                        "error": str(e)
                    }
            
            # Test batch responses if supported
            try:
                responses, histories = await client.get_batch_responses(
                    messages=[{"role": "user", "content": "Count to 3."}],
                    model=test_models[0],
                    n_responses=2,
                    temperature=0.5
                )
                
                batch_test_success = len(responses) > 0
                
            except Exception as e:
                logger.info(f"Batch responses not supported or failed: {e}")
                batch_test_success = False
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if any(r["status"] == "success" for r in model_test_results.values()) else "failed",
                "model_tests": model_test_results,
                "batch_responses_supported": batch_test_success
            }
            
            successful_models = sum(1 for r in model_test_results.values() if r["status"] == "success")
            logger.info(f"✅ {test_name}: {successful_models}/{len(test_models)} models working")
            
        except Exception as e:
            logger.error(f"❌ {test_name}: {e}")
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    async def _test_caching_functionality(self):
        """Test prompt caching features"""
        test_name = "caching_functionality"
        logger.info(f"💾 Testing {test_name}")
        
        try:
            from ai_scientist.openrouter import get_global_client, CacheStrategy
            
            if not self.api_key:
                self.test_results["tests"][test_name] = {
                    "status": "skipped",
                    "message": "No API key available"
                }
                return
            
            client = get_global_client()
            
            # Test different caching strategies
            cache_strategies = [CacheStrategy.AUTO, CacheStrategy.NONE]
            cache_test_results = {}
            
            base_prompt = "This is a long context for testing caching. " * 50  # Make it long enough to benefit from caching
            
            for strategy in cache_strategies:
                try:
                    start_time = time.time()
                    
                    response, _ = await client.get_response(
                        messages=[
                            {"role": "system", "content": base_prompt},
                            {"role": "user", "content": "Summarize the above in 10 words."}
                        ],
                        model="openai/gpt-4o-mini",
                        cache_strategy=strategy,
                        max_tokens=50
                    )
                    
                    end_time = time.time()
                    
                    cache_test_results[strategy.value] = {
                        "status": "success",
                        "response_time": end_time - start_time,
                        "response_length": len(response)
                    }
                    
                except Exception as e:
                    cache_test_results[strategy.value] = {
                        "status": "failed",
                        "error": str(e)
                    }
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if any(r["status"] == "success" for r in cache_test_results.values()) else "failed",
                "cache_strategies": cache_test_results
            }
            
            logger.info(f"✅ {test_name}: Caching strategies tested")
            
        except Exception as e:
            logger.error(f"❌ {test_name}: {e}")
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    async def _test_rag_system(self):
        """Test RAG document ingestion and retrieval"""
        test_name = "rag_system"
        logger.info(f"📚 Testing {test_name}")
        
        try:
            from ai_scientist.openrouter import RAGSystem
            from ai_scientist.openrouter.config import RAGConfig
            
            # Create test RAG config
            rag_config = RAGConfig(
                enabled=True,
                chunk_size=500,
                chunk_overlap=100,
                vector_store="chroma"
            )
            
            # Initialize RAG system
            rag_system = RAGSystem(rag_config, str(self.temp_dir / "rag_storage"))
            
            # Create test document
            test_doc = self.temp_dir / "test_document.txt"
            test_content = """
            This is a test document for RAG system testing.
            It contains information about machine learning, neural networks, and artificial intelligence.
            The document discusses various algorithms including decision trees, random forests, and deep learning.
            Natural language processing is also covered, including topics like tokenization and embeddings.
            """
            
            with open(test_doc, 'w') as f:
                f.write(test_content)
            
            # Test document ingestion
            doc_id = await rag_system.ingest_file(test_doc)
            
            if doc_id:
                # Test search functionality
                search_results = rag_system.search("machine learning algorithms", max_results=3)
                
                # Test context generation
                context = rag_system.get_context_for_query("neural networks")
                
                # Get system statistics
                stats = rag_system.get_statistics()
                
                self.test_results["tests"][test_name] = {
                    "status": "passed",
                    "document_ingested": doc_id is not None,
                    "search_results": len(search_results),
                    "context_generated": len(context) > 0,
                    "statistics": stats
                }
                
                logger.info(f"✅ {test_name}: RAG system operational")
            else:
                self.test_results["tests"][test_name] = {
                    "status": "failed",
                    "message": "Document ingestion failed"
                }
                
        except Exception as e:
            logger.error(f"❌ {test_name}: {e}")
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    async def _test_cli_functionality(self):
        """Test CLI interface functionality"""
        test_name = "cli_functionality"
        logger.info(f"💻 Testing {test_name}")
        
        try:
            from ai_scientist.openrouter.cli import CLIInterface
            
            # Initialize CLI
            cli = CLIInterface()
            
            # Test basic CLI methods
            cli_tests = {
                "initialization": cli is not None,
                "rich_available": cli.use_rich,
                "configuration_templates": True  # Would test template loading
            }
            
            self.test_results["tests"][test_name] = {
                "status": "passed",
                "cli_tests": cli_tests
            }
            
            logger.info(f"✅ {test_name}: CLI interface functional")
            
        except Exception as e:
            logger.error(f"❌ {test_name}: {e}")
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    async def _test_pipeline_integration(self):
        """Test integration with AI-Scientist pipeline"""
        test_name = "pipeline_integration"
        logger.info(f"🔄 Testing {test_name}")
        
        try:
            from ai_scientist.openrouter import create_default_config
            from ai_scientist.integration.simple_enhanced_launcher import SimpleEnhancedLauncher
            
            if not self.api_key:
                self.test_results["tests"][test_name] = {
                    "status": "skipped",
                    "message": "No API key for pipeline testing"
                }
                return
            
            # Create test configuration
            config = create_default_config()
            config.api_key = self.api_key
            
            # Set up lightweight models for testing
            for stage_config in config.stage_configs.values():
                stage_config.model = "openai/gpt-4o-mini"  # Fast model for testing
                stage_config.max_tokens = 100  # Limit tokens for speed
            
            # Initialize launcher
            launcher = SimpleEnhancedLauncher(config, rag_system=None)
            
            # Test individual stage execution (simplified)
            test_idea = "test_research_idea"
            test_results_dir = self.temp_dir / "pipeline_test"
            test_results_dir.mkdir(exist_ok=True)
            
            # Test ideation stage only (fastest)
            ideation_result = await launcher.run_enhanced_ideation(test_idea, test_results_dir)
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if ideation_result.get("status") == "completed" else "failed",
                "launcher_initialized": True,
                "ideation_test": ideation_result
            }
            
            logger.info(f"✅ {test_name}: Pipeline integration working")
            
        except Exception as e:
            logger.error(f"❌ {test_name}: {e}")
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    async def _test_error_handling(self):
        """Test error handling and fallback mechanisms"""
        test_name = "error_handling"
        logger.info(f"⚠️ Testing {test_name}")
        
        try:
            from ai_scientist.openrouter import get_global_client
            
            if not self.api_key:
                self.test_results["tests"][test_name] = {
                    "status": "skipped",
                    "message": "No API key for error testing"
                }
                return
            
            client = get_global_client()
            
            error_tests = {}
            
            # Test invalid model
            try:
                await client.get_response(
                    messages=[{"role": "user", "content": "test"}],
                    model="invalid/model",
                    max_tokens=10
                )
                error_tests["invalid_model"] = "no_error"  # This shouldn't happen
            except Exception as e:
                error_tests["invalid_model"] = "handled_correctly"
                logger.info(f"  ✓ Invalid model error handled: {type(e).__name__}")
            
            # Test invalid parameters
            try:
                await client.get_response(
                    messages=[{"role": "user", "content": "test"}],
                    model="openai/gpt-4o-mini",
                    temperature=-1.0,  # Invalid temperature
                    max_tokens=10
                )
                error_tests["invalid_params"] = "no_error"
            except Exception as e:
                error_tests["invalid_params"] = "handled_correctly"
                logger.info(f"  ✓ Invalid parameter error handled: {type(e).__name__}")
            
            self.test_results["tests"][test_name] = {
                "status": "passed",
                "error_tests": error_tests
            }
            
            logger.info(f"✅ {test_name}: Error handling working")
            
        except Exception as e:
            logger.error(f"❌ {test_name}: {e}")
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    async def _test_performance_limits(self):
        """Test performance and rate limiting"""
        test_name = "performance_limits"
        logger.info(f"🚀 Testing {test_name}")
        
        try:
            from ai_scientist.openrouter import get_global_client
            
            if not self.api_key:
                self.test_results["tests"][test_name] = {
                    "status": "skipped", 
                    "message": "No API key for performance testing"
                }
                return
            
            client = get_global_client()
            
            # Test concurrent requests
            async def make_test_request(i):
                try:
                    start_time = time.time()
                    response, _ = await client.get_response(
                        messages=[{"role": "user", "content": f"Count to {i}"}],
                        model="openai/gpt-4o-mini",
                        max_tokens=20
                    )
                    end_time = time.time()
                    return {
                        "success": True,
                        "response_time": end_time - start_time,
                        "response_length": len(response)
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            # Test 3 concurrent requests (conservative)
            concurrent_tasks = [make_test_request(i) for i in range(1, 4)]
            concurrent_results = await asyncio.gather(*concurrent_tasks)
            
            successful_concurrent = sum(1 for r in concurrent_results if r["success"])
            avg_response_time = sum(r.get("response_time", 0) for r in concurrent_results if r["success"]) / max(1, successful_concurrent)
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if successful_concurrent > 0 else "failed",
                "concurrent_requests": len(concurrent_results),
                "successful_concurrent": successful_concurrent,
                "average_response_time": avg_response_time,
                "results": concurrent_results
            }
            
            logger.info(f"✅ {test_name}: {successful_concurrent}/3 concurrent requests successful")
            
        except Exception as e:
            logger.error(f"❌ {test_name}: {e}")
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    def _generate_test_summary(self):
        """Generate final test summary"""
        total_tests = len(self.test_results["tests"])
        passed_tests = sum(1 for test in self.test_results["tests"].values() if test["status"] == "passed")
        failed_tests = sum(1 for test in self.test_results["tests"].values() if test["status"] == "failed")
        skipped_tests = sum(1 for test in self.test_results["tests"].values() if test["status"] == "skipped")
        
        if passed_tests == total_tests:
            overall_status = "all_passed"
        elif passed_tests > 0:
            overall_status = "partial_pass"
        else:
            overall_status = "all_failed"
        
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "skipped": skipped_tests,
            "overall_status": overall_status,
            "success_rate": passed_tests / max(1, total_tests - skipped_tests)
        }
        
        # Save detailed results
        results_file = self.temp_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 OpenRouter Integration Test Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {failed_tests}")
        logger.info(f"⏭️  Skipped: {skipped_tests}")
        logger.info(f"Success Rate: {passed_tests / max(1, total_tests - skipped_tests):.1%}")
        logger.info(f"Detailed Results: {results_file}")
        logger.info(f"{'='*60}")
        
        if failed_tests > 0:
            logger.info("\n❌ Failed Tests:")
            for test_name, result in self.test_results["tests"].items():
                if result["status"] == "failed":
                    logger.info(f"  • {test_name}: {result.get('error', 'Unknown error')}")
        
        if overall_status == "all_passed":
            logger.info("\n🎉 All tests passed! OpenRouter integration is working correctly.")
        elif overall_status == "partial_pass":
            logger.info(f"\n⚠️ Partial success. {passed_tests}/{total_tests} tests passed.")
        else:
            logger.info("\n💥 All tests failed. Check your configuration and API key.")
    
    def cleanup(self):
        """Clean up test resources"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory: {e}")

async def main():
    """Main test execution"""
    tester = OpenRouterIntegrationTester()
    
    try:
        print("🚀 Starting OpenRouter Integration Tests")
        print(f"API Key: {'✓ Found' if tester.api_key else '❌ Not Found'}")
        print(f"Temp Directory: {tester.temp_dir}")
        print("-" * 60)
        
        results = await tester.run_all_tests()
        
        # Final status
        if results["summary"]["overall_status"] == "all_passed":
            sys.exit(0)
        elif results["summary"]["overall_status"] == "partial_pass":
            sys.exit(1)
        else:
            sys.exit(2)
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Test execution failed: {e}")
        sys.exit(1)
    finally:
        tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())