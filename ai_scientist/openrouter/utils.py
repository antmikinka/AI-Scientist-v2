"""
OpenRouter Utility Functions

Utility functions for OpenRouter integration including JSON parsing,
error handling, and data processing.
"""

import json
import re
import logging
from typing import Dict, Optional, Any, List
import tiktoken

logger = logging.getLogger(__name__)

def extract_json_between_markers(text: str, markers: List[str] = None) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text between markers or code blocks.
    
    Args:
        text: Text containing JSON
        markers: Custom markers to look for (defaults to common patterns)
        
    Returns:
        Parsed JSON dictionary or None if not found
    """
    if markers is None:
        markers = [
            r"```json(.*?)```",           # Standard JSON code blocks
            r"```(.*?)```",               # Any code blocks
            r"<json>(.*?)</json>",        # XML-style JSON tags
            r"\{.*?\}",                   # Any JSON-like objects
        ]
    
    for pattern in markers:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            json_string = match.strip()
            
            # Try to parse JSON
            try:
                parsed_json = json.loads(json_string)
                if isinstance(parsed_json, dict):
                    return parsed_json
            except json.JSONDecodeError:
                # Try to clean up common JSON issues
                cleaned_json = _clean_json_string(json_string)
                if cleaned_json:
                    try:
                        parsed_json = json.loads(cleaned_json)
                        if isinstance(parsed_json, dict):
                            return parsed_json
                    except json.JSONDecodeError:
                        continue
    
    return None

def _clean_json_string(json_string: str) -> Optional[str]:
    """Clean and fix common JSON formatting issues."""
    try:
        # Remove control characters
        cleaned = re.sub(r'[\x00-\x1F\x7F]', '', json_string)
        
        # Fix common trailing comma issues
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        
        # Fix unescaped quotes in strings
        # This is a simple heuristic and may not work in all cases
        cleaned = re.sub(r'(?<!\\)"(?=\w)', '\\"', cleaned)
        
        return cleaned
    except Exception as e:
        logger.debug(f"Error cleaning JSON string: {e}")
        return None

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using appropriate tokenizer.
    
    Args:
        text: Text to count tokens for
        model: Model name to determine tokenizer
        
    Returns:
        Token count
    """
    try:
        # Map model names to tokenizer encodings
        if "gpt-4" in model.lower() or "gpt-3.5" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif "claude" in model.lower():
            # Claude uses similar tokenization to GPT-4
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif "gemini" in model.lower():
            # Gemini tokenization approximation
            encoding = tiktoken.encoding_for_model("gpt-4")
        else:
            # Default to GPT-4 tokenizer
            encoding = tiktoken.encoding_for_model("gpt-4")
        
        return len(encoding.encode(str(text)))
        
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}")
        # Fallback: approximate as 4 characters per token
        return len(str(text)) // 4

def truncate_text(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """
    Truncate text to fit within token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model: Model name for tokenization
        
    Returns:
        Truncated text
    """
    current_tokens = count_tokens(text, model)
    
    if current_tokens <= max_tokens:
        return text
    
    # Binary search for optimal truncation point
    left, right = 0, len(text)
    
    while left < right:
        mid = (left + right + 1) // 2
        truncated = text[:mid]
        
        if count_tokens(truncated, model) <= max_tokens:
            left = mid
        else:
            right = mid - 1
    
    return text[:left]

def format_model_name(model: str) -> str:
    """
    Format model name for display.
    
    Args:
        model: Model identifier
        
    Returns:
        Formatted model name
    """
    # Remove provider prefix for display
    if "/" in model:
        provider, name = model.split("/", 1)
        return f"{name} ({provider})"
    return model

def parse_provider_preferences(preferences: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse and validate provider preferences.
    
    Args:
        preferences: Raw provider preferences
        
    Returns:
        Validated preferences dictionary
    """
    validated = {}
    
    # Order preferences
    if "order" in preferences:
        order = preferences["order"]
        if isinstance(order, list):
            validated["order"] = order
    
    # Allow fallbacks
    if "allow_fallbacks" in preferences:
        validated["allow_fallbacks"] = bool(preferences["allow_fallbacks"])
    
    # Quantized models preference
    if "quantized" in preferences:
        validated["quantized"] = bool(preferences["quantized"])
    
    # Data processing location
    if "data_processing" in preferences:
        location = preferences["data_processing"]
        if location in ["eu", "us", "any"]:
            validated["data_processing"] = location
    
    return validated

def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str, 
                 cached_tokens: int = 0) -> Dict[str, float]:
    """
    Estimate cost for model usage.
    
    Args:
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        model: Model identifier
        cached_tokens: Number of cached tokens (for cost reduction)
        
    Returns:
        Cost breakdown dictionary
    """
    # Simplified cost estimation - would use actual pricing in production
    cost_per_1k_tokens = {
        "openai/gpt-4o": {"input": 0.005, "output": 0.015},
        "openai/gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "openai/o1-preview": {"input": 0.015, "output": 0.06},
        "openai/o1-mini": {"input": 0.003, "output": 0.012},
        "anthropic/claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
        "anthropic/claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "google/gemini-2.0-flash": {"input": 0.000075, "output": 0.0003},
        "deepseek/deepseek-v3": {"input": 0.00014, "output": 0.00028},
    }
    
    # Default pricing if model not found
    default_pricing = {"input": 0.002, "output": 0.008}
    pricing = cost_per_1k_tokens.get(model, default_pricing)
    
    # Calculate base costs
    prompt_cost = (prompt_tokens / 1000) * pricing["input"]
    completion_cost = (completion_tokens / 1000) * pricing["output"]
    
    # Apply caching discount
    cache_discount = 0.0
    if cached_tokens > 0:
        # Typical cache discount is 50-90%
        cache_savings_rate = 0.75  # 75% savings on cached tokens
        cache_discount = (cached_tokens / 1000) * pricing["input"] * cache_savings_rate
    
    total_cost = prompt_cost + completion_cost - cache_discount
    
    return {
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "cache_discount": cache_discount,
        "total_cost": max(0.0, total_cost),
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "cached": cached_tokens,
            "total": prompt_tokens + completion_tokens
        }
    }

def validate_temperature(temperature: float) -> float:
    """
    Validate and clamp temperature value.
    
    Args:
        temperature: Raw temperature value
        
    Returns:
        Valid temperature between 0.0 and 2.0
    """
    return max(0.0, min(2.0, float(temperature)))

def validate_max_tokens(max_tokens: int, model: str = None) -> int:
    """
    Validate max tokens against model limits.
    
    Args:
        max_tokens: Requested max tokens
        model: Model identifier
        
    Returns:
        Valid max tokens value
    """
    # Model context limits (simplified)
    context_limits = {
        "openai/gpt-4o": 128000,
        "openai/gpt-4o-mini": 128000,
        "openai/o1-preview": 128000,
        "openai/o1-mini": 65000,
        "anthropic/claude-3.5-sonnet": 200000,
        "anthropic/claude-3-haiku": 200000,
        "google/gemini-2.0-flash": 1048576,
        "deepseek/deepseek-v3": 64000,
    }
    
    max_limit = context_limits.get(model, 32000)  # Default limit
    
    # Leave some room for the prompt
    safe_limit = max_limit // 2
    
    return max(1, min(max_tokens, safe_limit))

def format_error_message(error: Exception, model: str = None) -> str:
    """
    Format error message for user display.
    
    Args:
        error: Exception object
        model: Model that caused the error
        
    Returns:
        Formatted error message
    """
    error_msg = str(error).lower()
    
    if "rate limit" in error_msg:
        return f"Rate limit exceeded for {model or 'model'}. Please try again in a few moments."
    elif "insufficient" in error_msg and "credits" in error_msg:
        return "Insufficient credits. Please check your OpenRouter account balance."
    elif "api key" in error_msg or "authentication" in error_msg:
        return "API authentication failed. Please check your OpenRouter API key."
    elif "model" in error_msg and "not found" in error_msg:
        return f"Model {model or 'specified'} is not available. Please choose a different model."
    elif "timeout" in error_msg:
        return "Request timed out. Please try again with a shorter prompt or different model."
    elif "content filter" in error_msg or "content policy" in error_msg:
        return "Content filtered by safety policies. Please modify your prompt."
    else:
        return f"API request failed: {str(error)[:200]}"

def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundaries
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            sentence_break = text.rfind('.', max(start, end - 100), end)
            if sentence_break > start:
                end = sentence_break + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = max(start + 1, end - overlap)
    
    return chunks

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to merge in
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged