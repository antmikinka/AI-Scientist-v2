"""
LLM Client Module

This module provides LLM client functionality for AI-Scientist-v2.
"""

# Mock LLM client for demonstration purposes
class MockLLMClient:
    """Mock LLM client for demonstrations"""

    def __init__(self, model=None, temperature=0.7, max_tokens=8192, **kwargs):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def agenerate(self, prompt, **kwargs):
        """Mock async generation"""
        return MockResponse(f"Mock response to: {prompt[:50]}...")

    async def __call__(self, messages, **kwargs):
        """Mock call method"""
        return MockResponse("Mock response from LLM")


class MockResponse:
    """Mock response object"""

    def __init__(self, content):
        self.content = content
        self.text = content


def create_client(**kwargs):
    """Create mock LLM client"""
    return MockLLMClient(**kwargs)


__all__ = ["create_client", "MockLLMClient"]