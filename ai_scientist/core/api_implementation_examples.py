"""
Implementation Examples and Patterns for AI-Scientist-v2 API Framework

Provides comprehensive examples and best practices for using the unified
API management system, including registration, usage patterns, and integration
with existing AI-Scientist-v2 components.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import aiohttp
import requests
from pathlib import Path

# Import the unified API framework components
from .api_registry import APIRegistry, APIDefinition, APIConfiguration, APICapabilities
from .api_client import BaseAPIClient, GenericRESTClient, APIClientFactory, APIGateway
from .api_error_handling import AdaptiveRetryManager, APICircuitBreaker, FallbackManager
from .api_monitoring import APIMonitor, APITrace
from .api_security import APIKeyManager, SecurityContext
from .api_performance import IntelligentCache, RequestBatcher, LoadBalancer

logger = logging.getLogger(__name__)

class APIImplementationExample:
    """Comprehensive examples of API integration patterns"""

    def __init__(self):
        self.registry = APIRegistry()
        self.client_factory = APIClientFactory()
        self.gateway = APIGateway()
        self.monitor = APIMonitor()
        self.security_manager = APIKeyManager()

    def example_register_openai_api(self):
        """Example: Registering OpenAI API with the unified framework"""
        # Define API capabilities
        capabilities = APICapabilities(
            supports_streaming=True,
            supports_async=True,
            supports_batch=True,
            supports_tools=True,
            max_batch_size=50,
            rate_limit_requests=1000,
            rate_limit_window=60
        )

        # Create API configuration
        config = APIConfiguration(
            base_url="https://api.openai.com/v1",
            timeout=30,
            max_retries=3,
            circuit_breaker_threshold=5,
            retry_config={
                "max_attempts": 3,
                "base_delay": 1.0,
                "max_delay": 60.0,
                "strategy": "exponential_backoff"
            },
            cache_config={
                "enabled": True,
                "ttl": 3600,
                "max_size": 1000
            }
        )

        # Define metadata
        metadata = {
            "api_name": "OpenAI",
            "version": "v1",
            "description": "OpenAI API for GPT models",
            "documentation_url": "https://platform.openai.com/docs/api-reference",
            "models": ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"],
            "features": ["chat", "completion", "embedding", "fine-tuning", "image-generation"],
            "pricing": {
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}
            }
        }

        # Register the API
        api_def = APIDefinition(
            api_id="openai",
            name="OpenAI API",
            version="v1",
            description="OpenAI API for GPT models and services",
            base_url="https://api.openai.com/v1",
            capabilities=capabilities,
            configuration=config,
            metadata=metadata
        )

        self.registry.register_api(api_def)
        logger.info("Registered OpenAI API successfully")

    def example_register_anthropic_api(self):
        """Example: Registering Anthropic Claude API"""
        capabilities = APICapabilities(
            supports_streaming=True,
            supports_async=True,
            supports_batch=False,
            supports_tools=True,
            max_batch_size=1,
            rate_limit_requests=1000,
            rate_limit_window=60
        )

        config = APIConfiguration(
            base_url="https://api.anthropic.com/v1",
            timeout=30,
            max_retries=3,
            circuit_breaker_threshold=5,
            retry_config={
                "max_attempts": 3,
                "base_delay": 1.0,
                "max_delay": 60.0,
                "strategy": "adaptive"
            },
            cache_config={
                "enabled": True,
                "ttl": 1800,
                "max_size": 500
            }
        )

        metadata = {
            "api_name": "Anthropic",
            "version": "v1",
            "description": "Anthropic Claude API",
            "documentation_url": "https://docs.anthropic.com/claude/reference",
            "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "features": ["messages", "streaming", "tool-use"],
            "pricing": {
                "claude-3-opus": {"input": 0.015, "output": 0.075},
                "claude-3-sonnet": {"input": 0.003, "output": 0.015}
            }
        }

        api_def = APIDefinition(
            api_id="anthropic",
            name="Anthropic Claude API",
            version="v1",
            description="Anthropic Claude API for advanced AI",
            base_url="https://api.anthropic.com/v1",
            capabilities=capabilities,
            configuration=config,
            metadata=metadata
        )

        self.registry.register_api(api_def)
        logger.info("Registered Anthropic API successfully")

    def example_create_custom_api_client(self):
        """Example: Creating a custom API client for a specific service"""
        class WeatherAPIClient(BaseAPIClient):
            """Custom weather API client"""

            def __init__(self, api_key: str, base_url: str = "https://api.weatherapi.com/v1"):
                super().__init__(base_url)
                self.api_key = api_key
                self.session = None

            async def __aenter__(self):
                self.session = aiohttp.ClientSession(
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if self.session:
                    await self.session.close()

            async def get_weather(self, location: str) -> Dict[str, Any]:
                """Get weather for a location"""
                params = {"q": location, "key": self.api_key}

                async with APITrace("weather_api", "get_weather") as trace:
                    trace.add_metadata("location", location)

                    try:
                        async with self.session.get(
                            f"{self.base_url}/current.json",
                            params=params
                        ) as response:
                            data = await response.json()
                            trace.add_metadata("temperature", data.get("current", {}).get("temp_c"))
                            return data
                    except Exception as e:
                        trace.set_error(e)
                        raise

            async def get_forecast(self, location: str, days: int = 7) -> Dict[str, Any]:
                """Get weather forecast"""
                params = {"q": location, "days": days, "key": self.api_key}

                async with APITrace("weather_api", "get_forecast") as trace:
                    trace.add_metadata("location", location)
                    trace.add_metadata("days", days)

                    try:
                        async with self.session.get(
                            f"{self.base_url}/forecast.json",
                            params=params
                        ) as response:
                            data = await response.json()
                            return data
                    except Exception as e:
                        trace.set_error(e)
                        raise

        return WeatherAPIClient

    def example_use_generic_rest_client(self):
        """Example: Using the generic REST client for quick integration"""
        # Create a generic client
        client = GenericRESTClient(
            base_url="https://jsonplaceholder.typicode.com",
            timeout=10,
            headers={"Content-Type": "application/json"}
        )

        # Sync usage
        try:
            posts = client.get("/posts")
            print(f"Retrieved {len(posts)} posts")

            # Create a new post
            new_post = {
                "title": "New Post",
                "body": "This is a new post",
                "userId": 1
            }
            created_post = client.post("/posts", json=new_post)
            print(f"Created post with ID: {created_post['id']}")

            # Update the post
            updated_post = client.put(f"/posts/{created_post['id']}", json={
                **created_post,
                "title": "Updated Post"
            })
            print(f"Updated post: {updated_post['title']}")

        except Exception as e:
            print(f"Error: {e}")

        # Async usage
        async def async_example():
            async with client:
                try:
                    users = await client.async_get("/users")
                    print(f"Retrieved {len(users)} users")
                except Exception as e:
                    print(f"Async error: {e}")

        # Run async example
        asyncio.run(async_example())

    def example_api_gateway_routing(self):
        """Example: Using API Gateway for intelligent routing"""
        # Configure gateway with multiple APIs
        self.gateway.add_api("openai", "https://api.openai.com/v1", priority=1)
        self.gateway.add_api("anthropic", "https://api.anthropic.com/v1", priority=2)
        self.gateway.add_api("cohere", "https://api.cohere.ai/v1", priority=3)

        # Set up load balancing
        self.gateway.set_load_balancer("round_robin")

        # Define routing rules
        self.gateway.add_route_rule(
            condition=lambda request: "gpt" in request.get("model", "").lower(),
            api_id="openai"
        )

        self.gateway.add_route_rule(
            condition=lambda request: "claude" in request.get("model", "").lower(),
            api_id="anthropic"
        )

        # Make a request through the gateway
        async def make_gateway_request():
            request = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 100
            }

            try:
                response = await self.gateway.async_request("openai", "/chat/completions", request)
                print(f"Gateway response: {response}")
            except Exception as e:
                print(f"Gateway error: {e}")

        asyncio.run(make_gateway_request())

    def example_error_handling_patterns(self):
        """Example: Advanced error handling patterns"""
        # Create adaptive retry manager
        retry_manager = AdaptiveRetryManager(
            max_attempts=5,
            base_delay=1.0,
            max_delay=60.0,
            strategy="adaptive"
        )

        # Create circuit breaker
        circuit_breaker = APICircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=(Exception,)
        )

        # Create fallback manager
        fallback_manager = FallbackManager()

        # Define fallback functions
        async def fallback_api_call(request):
            print("Using fallback API")
            return {"fallback": True, "message": "Service unavailable, using fallback"}

        async def cache_fallback(request):
            print("Using cached response")
            return {"cached": True, "message": "Cached response"}

        # Register fallbacks
        fallback_manager.add_fallback(fallback_api_call)
        fallback_manager.add_fallback(cache_fallback)

        # Use with decorators
        @retry_manager.retry
        @circuit_breaker
        async def unreliable_api_call(request: Dict[str, Any]) -> Dict[str, Any]:
            # Simulate an unreliable API
            import random
            if random.random() < 0.3:  # 30% failure rate
                raise Exception("API unavailable")

            return {"success": True, "data": "API response"}

        # Test the error handling
        async def test_error_handling():
            try:
                # This will use retry and circuit breaker
                result = await unreliable_api_call({"test": "data"})
                print(f"Success: {result}")
            except Exception as e:
                print(f"Failed after retries: {e}")

                # Try with fallback
                result = await fallback_manager.execute_with_fallbacks(
                    unreliable_api_call,
                    {"test": "data"}
                )
                print(f"Fallback result: {result}")

        asyncio.run(test_error_handling())

    def example_monitoring_and_metrics(self):
        """Example: Monitoring and metrics collection"""
        # Create a monitor
        monitor = APIMonitor()

        # Start collecting metrics
        monitor.start_monitoring()

        # Record custom metrics
        monitor.record_metric(
            name="api_response_time",
            value=150.5,
            tags={"api": "openai", "endpoint": "/chat/completions"}
        )

        monitor.record_metric(
            name="api_requests_total",
            value=1,
            tags={"api": "openai", "status": "200"}
        )

        # Create an alert
        alert = monitor.create_alert(
            name="high_error_rate",
            condition=lambda metrics: metrics.get("error_rate", 0) > 0.05,
            action=lambda: print("High error rate detected!")
        )

        # Use tracing
        async def traced_api_call():
            async with APITrace("openai", "chat_completion") as trace:
                trace.add_metadata("model", "gpt-4")
                trace.add_metadata("tokens", 100)

                try:
                    # Simulate API call
                    await asyncio.sleep(0.1)
                    trace.add_metadata("response_time", 100)
                    return {"response": "API response"}
                except Exception as e:
                    trace.set_error(e)
                    raise

        # Test monitoring
        asyncio.run(traced_api_call())

        # Get metrics summary
        metrics = monitor.get_metrics_summary()
        print(f"Metrics summary: {metrics}")

    def example_security_patterns(self):
        """Example: Security and access control patterns"""
        # Create security manager
        security_manager = APIKeyManager()

        # Generate API keys
        key_id, api_key = security_manager.generate_api_key(
            description="Research API key",
            permissions=["read", "write"],
            rate_limit=100
        )

        print(f"Generated API key: {api_key[:20]}... (ID: {key_id})")

        # Create security context
        context = SecurityContext(
            api_key=api_key,
            permissions=["read", "write"],
            rate_limit=100
        )

        # Validate API key
        key_info = security_manager.validate_api_key(api_key)
        if key_info:
            print(f"API key valid: {key_info.description}")
            print(f"Permissions: {key_info.permissions}")
            print(f"Rate limit: {key_info.rate_limit}")

        # Check rate limit
        if security_manager.check_rate_limit(key_id):
            print("Rate limit check passed")
        else:
            print("Rate limit exceeded")

        # Use security decorator
        @security_manager.require_api_key("read")
        async def secure_api_operation(data: Dict[str, Any]) -> Dict[str, Any]:
            return {"success": True, "data": data}

        # Test secure operation
        asyncio.run(secure_api_operation({"test": "data"}))

    def example_performance_optimization(self):
        """Example: Performance optimization patterns"""
        # Create intelligent cache
        cache = IntelligentCache(max_size=1000, ttl=3600)

        # Create request batcher
        batcher = RequestBatcher(
            max_batch_size=10,
            max_wait_time=1.0,
            timeout=30
        )

        # Create load balancer
        load_balancer = LoadBalancer(strategy="weighted_round_robin")

        # Add servers to load balancer
        load_balancer.add_server("https://api1.example.com", weight=2)
        load_balancer.add_server("https://api2.example.com", weight=1)

        # Use cache
        @cache.cached_response(ttl=300)
        async def cached_api_call(query: str) -> Dict[str, Any]:
            print(f"Making API call for: {query}")
            await asyncio.sleep(0.1)  # Simulate API call
            return {"query": query, "result": "data"}

        # Use request batching
        @batcher.batch_requests
        async def batch_api_call(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            print(f"Processing batch of {len(requests)} requests")
            await asyncio.sleep(0.1)  # Simulate batch processing
            return [{"result": f"processed_{i}"} for i in range(len(requests))]

        # Test performance optimizations
        async def test_performance():
            # Test cache
            result1 = await cached_api_call("test query")
            result2 = await cached_api_call("test query")  # Should be cached
            print(f"Cache test: {result1 == result2}")

            # Test batching
            requests = [{"id": i, "data": f"request_{i}"} for i in range(5)]
            batch_results = await batch_api_call(requests)
            print(f"Batch test: {len(batch_results)} results")

            # Test load balancing
            for i in range(5):
                server = load_balancer.get_next_server()
                print(f"Request {i+1} routed to: {server}")

        asyncio.run(test_performance())

    def example_integration_with_existing_components(self):
        """Example: Integrating with existing AI-Scientist-v2 components"""
        # This would integrate with the existing OpenRouter client
        try:
            from ai_scientist.openrouter.client import OpenRouterClient

            # Wrap existing client with our framework
            class OpenRouterAdapter(BaseAPIClient):
                def __init__(self, openrouter_client: OpenRouterClient):
                    super().__init__(openrouter_client.base_url)
                    self.client = openrouter_client

                async def async_request(self, method: str, endpoint: str, **kwargs):
                    # Adapt to existing OpenRouter interface
                    if endpoint == "/chat/completions":
                        messages = kwargs.get("json", {}).get("messages", [])
                        model = kwargs.get("json", {}).get("model", "anthropic/claude-3-opus")

                        response = await self.client.generate(
                            messages=messages,
                            model=model
                        )

                        return {
                            "id": "chat_" + str(hash(str(messages))),
                            "object": "chat.completion",
                            "created": int(asyncio.get_event_loop().time()),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": response.get("choices", [{}])[0].get("message", {}).get("content", "")
                                },
                                "finish_reason": "stop"
                            }]
                        }

                    raise NotImplementedError(f"Endpoint {endpoint} not supported")

            # Register the adapter
            self.client_factory.register_client("openrouter", OpenRouterAdapter)

            print("Integrated OpenRouter client successfully")

        except ImportError:
            print("OpenRouter client not available for integration")

    def example_complete_workflow(self):
        """Example: Complete workflow using all components"""
        print("\n=== Complete API Integration Workflow ===\n")

        # 1. Register APIs
        self.example_register_openai_api()
        self.example_register_anthropic_api()

        # 2. Create API gateway
        self.example_api_gateway_routing()

        # 3. Set up error handling
        self.example_error_handling_patterns()

        # 4. Configure monitoring
        self.example_monitoring_and_metrics()

        # 5. Set up security
        self.example_security_patterns()

        # 6. Optimize performance
        self.example_performance_optimization()

        # 7. Integration example
        self.example_integration_with_existing_components()

        print("\n=== Workflow Complete ===")

def create_sample_api_configurations():
    """Create sample API configurations for common services"""
    configurations = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "timeout": 30,
            "max_retries": 3,
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer YOUR_API_KEY"
            },
            "endpoints": {
                "chat": "/chat/completions",
                "completion": "/completions",
                "embedding": "/embeddings",
                "models": "/models"
            }
        },
        "anthropic": {
            "base_url": "https://api.anthropic.com/v1",
            "timeout": 30,
            "max_retries": 3,
            "headers": {
                "Content-Type": "application/json",
                "x-api-key": "YOUR_API_KEY",
                "anthropic-version": "2023-06-01"
            },
            "endpoints": {
                "messages": "/messages",
                "models": "/models"
            }
        },
        "cohere": {
            "base_url": "https://api.cohere.ai/v1",
            "timeout": 30,
            "max_retries": 3,
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer YOUR_API_KEY"
            },
            "endpoints": {
                "chat": "/chat",
                "generate": "/generate",
                "embed": "/embed",
                "models": "/models"
            }
        }
    }

    return configurations

def create_migration_guide():
    """Create migration guide for existing integrations"""
    migration_guide = """
# API Integration Migration Guide

## Existing OpenRouter Integration

### Before (Current)
```python
from ai_scientist.openrouter.client import OpenRouterClient

client = OpenRouterClient(api_key="your-key")
response = client.generate(messages=[...], model="anthropic/claude-3-opus")
```

### After (Unified Framework)
```python
from ai_scientist.core.api_client import APIClientFactory
from ai_scientist.core.api_registry import APIRegistry

# Register the API (one-time setup)
registry = APIRegistry()
registry.register_api(openai_api_definition)

# Use the client factory
client_factory = APIClientFactory()
client = client_factory.create_client("openrouter")

response = await client.async_request(
    "POST",
    "/chat/completions",
    json={"messages": [...], "model": "anthropic/claude-3-opus"}
)
```

## Existing ChromaDB Integration

### Before (Current)
```python
from ai_scientist.chroma_client import ChromaClient

client = ChromaClient()
results = client.similarity_search(query, n_results=5)
```

### After (Unified Framework)
```python
from ai_scientist.core.api_client import GenericRESTClient

client = GenericRESTClient(base_url="http://localhost:8000")
results = client.post("/similarity_search", json={
    "query": query,
    "n_results": 5
})
```

## Benefits of Migration

1. **Unified Interface**: All APIs use the same client interface
2. **Enhanced Error Handling**: Automatic retries and circuit breakers
3. **Monitoring**: Built-in metrics and tracing
4. **Security**: Centralized API key management
5. **Performance**: Intelligent caching and load balancing
6. **Extensibility**: Easy to add new APIs
"""

    return migration_guide

if __name__ == "__main__":
    # Run examples
    example = APIImplementationExample()
    example.example_complete_workflow()

    # Print sample configurations
    print("\n=== Sample API Configurations ===")
    configs = create_sample_api_configurations()
    for api_name, config in configs.items():
        print(f"\n{api_name.upper()}:")
        print(json.dumps(config, indent=2))

    # Print migration guide
    print("\n=== Migration Guide ===")
    print(create_migration_guide())