"""
OpenRouter Client Implementation

Comprehensive OpenRouter client with advanced features:
- 200+ AI models from multiple providers
- Prompt caching (automatic and explicit)
- Tool calling with universal support
- Reasoning tokens for O1/O3 models
- Streaming responses
- Smart fallbacks and error handling
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any, Tuple, Union, AsyncIterator
from enum import Enum
from dataclasses import dataclass, asdict
import aiohttp
import backoff

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Prompt caching strategies"""
    NONE = "none"
    AUTO = "auto"           # For OpenAI, DeepSeek, Grok models
    EPHEMERAL = "ephemeral" # For Anthropic, Gemini models
    EXPLICIT = "explicit"   # Manual cache control

@dataclass
class ToolCall:
    """Tool call structure"""
    id: str
    type: str
    function: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls"""
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function
        }

@dataclass
class ToolDefinition:
    """Tool definition for function calling"""
    name: str
    description: str
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI function format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

@dataclass
class Message:
    """Message structure for OpenRouter API"""
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

@dataclass
class OpenRouterResponse:
    """OpenRouter API response structure"""
    id: str
    choices: List[Dict[str, Any]]
    created: int
    model: str
    usage: Optional[Dict[str, Any]] = None
    system_fingerprint: Optional[str] = None

# Global client instance
_global_client: Optional['OpenRouterClient'] = None

class OpenRouterClient:
    """
    Comprehensive OpenRouter client supporting all advanced features.
    
    Features:
    - 200+ models from multiple providers
    - Prompt caching with 25-90% cost savings
    - Universal tool calling
    - Reasoning tokens for O1/O3 models
    - Streaming responses
    - Smart fallbacks and error handling
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1",
                 enable_cost_tracking: bool = True):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            base_url: OpenRouter API base URL
            enable_cost_tracking: Whether to enable cost tracking
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Configuration
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "https://ai-scientist-v2.com")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "AI-Scientist-v2")
        
        # Cache for available models
        self._models_cache: Optional[List[Dict[str, Any]]] = None
        self._cache_timestamp: Optional[float] = None
        
        # Cost tracking
        self.enable_cost_tracking = enable_cost_tracking
        if enable_cost_tracking:
            try:
                from .cost_tracker import get_global_cost_tracker
                self.cost_tracker = get_global_cost_tracker()
            except ImportError:
                logger.warning("Cost tracking not available")
                self.cost_tracker = None
                self.enable_cost_tracking = False
        else:
            self.cost_tracker = None
        
        logger.info(f"Initialized OpenRouter client for {self.app_name} (cost tracking: {self.enable_cost_tracking})")

    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def _ensure_session(self):
        """Ensure aiohttp session is created"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()

    def _get_headers(self) -> Dict[str, str]:
        """Get standard headers for OpenRouter API"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
        }

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        factor=2
    )
    async def _make_request(self, endpoint: str, data: Optional[Dict] = None, method: str = "POST") -> Dict[str, Any]:
        """Make HTTP request to OpenRouter API with retry logic"""
        await self._ensure_session()
        
        url = f"{self.base_url}/{endpoint}"
        headers = self._get_headers()
        
        try:
            async with self.session.request(method, url, headers=headers, json=data) as response:
                response_text = await response.text()
                
                if response.status == 429:
                    # Rate limit - wait and retry
                    retry_after = int(response.headers.get("Retry-After", "60"))
                    logger.warning(f"Rate limited. Waiting {retry_after} seconds.")
                    await asyncio.sleep(retry_after)
                    raise aiohttp.ClientError("Rate limited")
                
                if response.status >= 400:
                    logger.error(f"OpenRouter API error {response.status}: {response_text}")
                    raise aiohttp.ClientError(f"HTTP {response.status}: {response_text}")
                
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {response_text}")
                    raise aiohttp.ClientError(f"Invalid JSON response: {e}")
        
        except Exception as e:
            logger.error(f"Request to {url} failed: {e}")
            raise

    async def get_available_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of available models from OpenRouter.
        
        Args:
            force_refresh: Force refresh of cached models
            
        Returns:
            List of model information dictionaries
        """
        # Check cache (5 minute TTL)
        if (not force_refresh and self._models_cache and self._cache_timestamp and 
            time.time() - self._cache_timestamp < 300):
            return self._models_cache
        
        try:
            response = await self._make_request("models", method="GET")
            models = response.get("data", [])
            
            # Cache the results
            self._models_cache = models
            self._cache_timestamp = time.time()
            
            logger.info(f"Retrieved {len(models)} available models from OpenRouter")
            return models
            
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            # Return cached version if available
            if self._models_cache:
                logger.warning("Using cached models due to API error")
                return self._models_cache
            raise

    def _prepare_messages_for_caching(self, messages: List[Dict[str, Any]], 
                                    cache_strategy: CacheStrategy) -> List[Dict[str, Any]]:
        """
        Prepare messages with appropriate caching headers.
        
        Args:
            messages: List of message dictionaries
            cache_strategy: Caching strategy to use
            
        Returns:
            Messages with caching annotations
        """
        if cache_strategy == CacheStrategy.NONE:
            return messages
        
        if cache_strategy == CacheStrategy.AUTO:
            # Auto caching - no modifications needed for OpenAI, DeepSeek, Grok
            return messages
        
        if cache_strategy in [CacheStrategy.EPHEMERAL, CacheStrategy.EXPLICIT]:
            # Add cache control to appropriate messages for Anthropic/Gemini
            processed_messages = []
            
            for msg in messages:
                if msg["role"] in ["system", "user"] and isinstance(msg["content"], str):
                    # Convert string content to multipart format for caching
                    if len(msg["content"]) > 1000:  # Only cache large content
                        processed_msg = {
                            "role": msg["role"],
                            "content": [
                                {
                                    "type": "text",
                                    "text": msg["content"],
                                    "cache_control": {"type": "ephemeral"}
                                }
                            ]
                        }
                        processed_messages.append(processed_msg)
                    else:
                        processed_messages.append(msg)
                else:
                    processed_messages.append(msg)
            
            return processed_messages
        
        return messages

    async def get_response(self, messages: List[Dict[str, Any]], model: str,
                         temperature: float = 0.7, max_tokens: int = 4096,
                         tools: Optional[List[Dict[str, Any]]] = None,
                         tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                         cache_strategy: CacheStrategy = CacheStrategy.AUTO,
                         reasoning_config: Optional[Dict[str, Any]] = None,
                         **kwargs) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get single response from OpenRouter API.
        
        Args:
            messages: List of message dictionaries
            model: Model identifier (e.g., "anthropic/claude-3.5-sonnet")
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            tools: List of available tools for tool calling
            tool_choice: Tool choice strategy
            cache_strategy: Prompt caching strategy
            reasoning_config: Configuration for reasoning models (O1/O3)
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (response_content, updated_message_history)
        """
        try:
            # Prepare messages with caching if needed
            processed_messages = self._prepare_messages_for_caching(messages, cache_strategy)
            
            # Build request payload
            request_data = {
                "model": model,
                "messages": processed_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            
            # Add tools if provided
            if tools:
                request_data["tools"] = tools
                if tool_choice:
                    request_data["tool_choice"] = tool_choice
            
            # Add reasoning configuration for O1/O3 models
            if reasoning_config and ("o1" in model.lower() or "o3" in model.lower()):
                request_data.update(reasoning_config)
            
            # Make API request
            response_data = await self._make_request("chat/completions", request_data)
            
            # Extract response content
            if not response_data.get("choices"):
                raise ValueError("No choices in response")
            
            choice = response_data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")
            
            # Handle tool calls
            tool_calls = message.get("tool_calls")
            if tool_calls:
                # Add assistant message with tool calls to history
                updated_history = processed_messages + [{
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls
                }]
            else:
                # Standard response
                updated_history = processed_messages + [{
                    "role": "assistant", 
                    "content": content
                }]
            
            # Log usage information and track costs
            usage = response_data.get("usage", {})
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                
                logger.info(f"Token usage - Prompt: {prompt_tokens}, "
                          f"Completion: {completion_tokens}, "
                          f"Total: {total_tokens}")
                
                # Track costs if enabled
                if self.cost_tracker:
                    # Extract cached tokens if available (OpenRouter specific)
                    cached_tokens = usage.get("prompt_tokens_cached", 0)
                    
                    # Extract stage from kwargs or response ID
                    stage = kwargs.get("stage")
                    request_id = response_data.get("id")
                    
                    cost = self.cost_tracker.record_usage(
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cached_tokens=cached_tokens,
                        request_id=request_id,
                        stage=stage
                    )
                    
                    logger.debug(f"Recorded cost: ${cost:.6f}")
            
            return content, updated_history
            
        except Exception as e:
            logger.error(f"OpenRouter API request failed: {e}")
            raise

    async def get_batch_responses(self, messages: List[Dict[str, Any]], model: str,
                                n_responses: int = 1, temperature: float = 0.7,
                                max_tokens: int = 4096,
                                cache_strategy: CacheStrategy = CacheStrategy.AUTO,
                                **kwargs) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
        """
        Get multiple responses from OpenRouter API for ensembling.
        
        Args:
            messages: List of message dictionaries
            model: Model identifier
            n_responses: Number of responses to generate
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            cache_strategy: Prompt caching strategy
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (list_of_responses, list_of_message_histories)
        """
        if n_responses == 1:
            # Single response case
            content, history = await self.get_response(
                messages, model, temperature, max_tokens, 
                cache_strategy=cache_strategy, **kwargs
            )
            return [content], [history]
        
        # Multiple responses - some models support n parameter
        if model.startswith("openai/"):
            try:
                processed_messages = self._prepare_messages_for_caching(messages, cache_strategy)
                
                request_data = {
                    "model": model,
                    "messages": processed_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "n": n_responses,
                    **kwargs
                }
                
                response_data = await self._make_request("chat/completions", request_data)
                
                contents = []
                histories = []
                
                for choice in response_data.get("choices", []):
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    contents.append(content)
                    
                    # Each response gets its own history
                    history = processed_messages + [{
                        "role": "assistant",
                        "content": content
                    }]
                    histories.append(history)
                
                return contents, histories
                
            except Exception as e:
                logger.warning(f"Batch request failed, falling back to individual requests: {e}")
        
        # Fallback: Make individual requests
        contents = []
        histories = []
        
        tasks = []
        for _ in range(n_responses):
            task = self.get_response(
                messages, model, temperature, max_tokens,
                cache_strategy=cache_strategy, **kwargs
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Individual batch request failed: {result}")
                continue
            
            content, history = result
            contents.append(content)
            histories.append(history)
        
        return contents, histories

    async def stream_response(self, messages: List[Dict[str, Any]], model: str,
                            temperature: float = 0.7, max_tokens: int = 4096,
                            tools: Optional[List[Dict[str, Any]]] = None,
                            cache_strategy: CacheStrategy = CacheStrategy.AUTO,
                            **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream response from OpenRouter API.
        
        Args:
            messages: List of message dictionaries
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: List of available tools
            cache_strategy: Prompt caching strategy
            **kwargs: Additional parameters
            
        Yields:
            Streaming response chunks
        """
        try:
            processed_messages = self._prepare_messages_for_caching(messages, cache_strategy)
            
            request_data = {
                "model": model,
                "messages": processed_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
                **kwargs
            }
            
            if tools:
                request_data["tools"] = tools
            
            await self._ensure_session()
            url = f"{self.base_url}/chat/completions"
            headers = self._get_headers()
            
            async with self.session.post(url, headers=headers, json=request_data) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise aiohttp.ClientError(f"HTTP {response.status}: {error_text}")
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        
                        if data == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data)
                            yield chunk
                        except json.JSONDecodeError:
                            # Skip invalid JSON lines
                            continue
        
        except Exception as e:
            logger.error(f"Streaming request failed: {e}")
            raise

    def get_response_sync(self, messages: List[Dict[str, Any]], model: str,
                         temperature: float = 0.7, max_tokens: int = 4096,
                         **kwargs) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Synchronous wrapper for get_response.
        
        Args:
            messages: List of message dictionaries
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (response_content, updated_message_history)
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.get_response(messages, model, temperature, max_tokens, **kwargs)
        )

    def get_batch_responses_sync(self, messages: List[Dict[str, Any]], model: str,
                               n_responses: int = 1, temperature: float = 0.7,
                               max_tokens: int = 4096, **kwargs) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
        """
        Synchronous wrapper for get_batch_responses.
        
        Args:
            messages: List of message dictionaries
            model: Model identifier
            n_responses: Number of responses to generate
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (list_of_responses, list_of_message_histories)
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.get_batch_responses(messages, model, n_responses, temperature, max_tokens, **kwargs)
        )

    async def get_generation_info(self, generation_id: str) -> Dict[str, Any]:
        """
        Get detailed generation information including costs and native token counts.
        
        Args:
            generation_id: Generation ID from response
            
        Returns:
            Generation information dictionary
        """
        try:
            response = await self._make_request(f"generation?id={generation_id}", method="GET")
            return response
        except Exception as e:
            logger.error(f"Failed to get generation info for {generation_id}: {e}")
            raise
    
    async def call_function(self, messages: List[Dict[str, Any]], model: str,
                           tools: List[ToolDefinition], tool_choice: str = "auto",
                           max_iterations: int = 5, **kwargs) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Execute function calling workflow with automatic tool execution.
        
        Args:
            messages: List of message dictionaries
            model: Model identifier
            tools: List of available tools
            tool_choice: Tool choice strategy ("auto", "none", or specific tool)
            max_iterations: Maximum number of tool calling iterations
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (final_response, complete_message_history)
        """
        try:
            # Convert tools to API format
            api_tools = [tool.to_dict() for tool in tools]
            tool_registry = {tool.name: tool for tool in tools}
            
            current_messages = messages.copy()
            iteration = 0
            
            while iteration < max_iterations:
                # Make API call with tools
                response, updated_history = await self.get_response(
                    messages=current_messages,
                    model=model,
                    tools=api_tools,
                    tool_choice=tool_choice,
                    **kwargs
                )
                
                current_messages = updated_history
                
                # Check if there are tool calls to execute
                last_message = current_messages[-1] if current_messages else {}
                tool_calls = last_message.get("tool_calls", [])
                
                if not tool_calls:
                    # No tool calls - we're done
                    return response, current_messages
                
                # Execute tool calls
                for tool_call in tool_calls:
                    tool_name = tool_call.get("function", {}).get("name")
                    tool_args = tool_call.get("function", {}).get("arguments")
                    tool_id = tool_call.get("id")
                    
                    if tool_name in tool_registry:
                        try:
                            # Parse arguments if they're a string
                            if isinstance(tool_args, str):
                                import json
                                tool_args = json.loads(tool_args)
                            
                            # Execute the tool (placeholder - would need actual implementation)
                            tool_result = await self._execute_tool(tool_name, tool_args)
                            
                            # Add tool result to conversation
                            current_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": str(tool_result)
                            })
                            
                        except Exception as e:
                            logger.error(f"Tool execution failed for {tool_name}: {e}")
                            current_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": f"Error: Tool execution failed - {str(e)}"
                            })
                    else:
                        current_messages.append({
                            "role": "tool", 
                            "tool_call_id": tool_id,
                            "content": f"Error: Unknown tool {tool_name}"
                        })
                
                iteration += 1
            
            # If we hit max iterations, make one final call without tools
            final_response, final_history = await self.get_response(
                messages=current_messages,
                model=model,
                **kwargs
            )
            
            return final_response, final_history
            
        except Exception as e:
            logger.error(f"Function calling workflow failed: {e}")
            raise
    
    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """
        Execute a tool function (placeholder implementation).
        
        In a real implementation, this would dispatch to actual tool implementations.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        # Placeholder implementations for common tools
        if tool_name == "web_search":
            return f"Web search results for: {tool_args.get('query', 'N/A')}"
        elif tool_name == "code_execution":
            return f"Code execution result: {tool_args.get('code', 'No code provided')}"
        elif tool_name == "file_read":
            return f"File contents: {tool_args.get('filename', 'No filename provided')}"
        elif tool_name == "calculation":
            expression = tool_args.get('expression', '')
            try:
                # Safe evaluation for basic math
                import re
                if re.match(r'^[\d\+\-\*/\(\)\.\s]+$', expression):
                    result = eval(expression)
                    return f"Calculation result: {result}"
                else:
                    return "Error: Invalid expression"
            except:
                return "Error: Calculation failed"
        else:
            return f"Tool {tool_name} not implemented (args: {tool_args})"
    
    def create_tool_definitions(self) -> List[ToolDefinition]:
        """
        Create standard tool definitions for AI-Scientist use cases.
        
        Returns:
            List of tool definitions
        """
        tools = [
            ToolDefinition(
                name="web_search",
                description="Search the web for information",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            ),
            
            ToolDefinition(
                name="code_execution", 
                description="Execute Python code",
                parameters={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute"
                        }
                    },
                    "required": ["code"]
                }
            ),
            
            ToolDefinition(
                name="literature_search",
                description="Search academic literature",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string", 
                            "description": "Academic search query"
                        },
                        "source": {
                            "type": "string",
                            "description": "Source to search (arxiv, pubmed, etc.)",
                            "enum": ["arxiv", "pubmed", "scholar"]
                        }
                    },
                    "required": ["query"]
                }
            ),
            
            ToolDefinition(
                name="data_analysis",
                description="Analyze data and generate insights",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "description": "Data to analyze (JSON format)"
                        },
                        "analysis_type": {
                            "type": "string",
                            "description": "Type of analysis to perform",
                            "enum": ["descriptive", "statistical", "correlation", "regression"]
                        }
                    },
                    "required": ["data", "analysis_type"]
                }
            ),
            
            ToolDefinition(
                name="calculation",
                description="Perform mathematical calculations",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            )
        ]
        
        return tools
    
    async def get_response_with_fallback(self, messages: List[Dict[str, Any]], 
                                       primary_model: str, fallback_models: List[str],
                                       **kwargs) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get response with automatic fallback to other models on failure.
        
        Args:
            messages: List of message dictionaries
            primary_model: Primary model to try first
            fallback_models: List of fallback models to try
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (response_content, updated_message_history)
        """
        models_to_try = [primary_model] + fallback_models
        
        for i, model in enumerate(models_to_try):
            try:
                logger.info(f"Trying model {model} (attempt {i + 1}/{len(models_to_try)})")
                
                response, history = await self.get_response(
                    messages=messages,
                    model=model,
                    **kwargs
                )
                
                if i > 0:
                    logger.info(f"Success with fallback model: {model}")
                
                return response, history
                
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                
                if i == len(models_to_try) - 1:
                    # Last model failed - re-raise the error
                    logger.error(f"All models failed. Last error: {e}")
                    raise
                else:
                    # Try next model
                    continue
        
        raise RuntimeError("All models failed unexpectedly")

    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'session') and self.session and not self.session.closed:
            # Try to close session gracefully
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.session.close())
                else:
                    loop.run_until_complete(self.session.close())
            except Exception:
                pass  # Ignore cleanup errors

def initialize_openrouter(api_key: Optional[str] = None, 
                         site_url: Optional[str] = None,
                         app_name: Optional[str] = None) -> OpenRouterClient:
    """
    Initialize global OpenRouter client.
    
    Args:
        api_key: OpenRouter API key
        site_url: Site URL for rankings
        app_name: App name for rankings
        
    Returns:
        OpenRouter client instance
    """
    global _global_client
    
    # Set environment variables if provided
    if site_url:
        os.environ["OPENROUTER_SITE_URL"] = site_url
    if app_name:
        os.environ["OPENROUTER_APP_NAME"] = app_name
    
    _global_client = OpenRouterClient(api_key=api_key)
    logger.info("Initialized global OpenRouter client")
    return _global_client

def get_global_client() -> OpenRouterClient:
    """
    Get global OpenRouter client instance.
    
    Returns:
        OpenRouter client instance
        
    Raises:
        RuntimeError: If client not initialized
    """
    global _global_client
    
    if _global_client is None:
        # Try to initialize with environment variables
        try:
            _global_client = OpenRouterClient()
            logger.info("Auto-initialized global OpenRouter client")
        except ValueError as e:
            raise RuntimeError(f"OpenRouter client not initialized. Call initialize_openrouter() first. {e}")
    
    return _global_client