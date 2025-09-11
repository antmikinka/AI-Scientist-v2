"""
LLM Interface for AI-Scientist-v2 with OpenRouter Integration
Maintains backward compatibility while leveraging OpenRouter's unified API
"""

import json
import os
import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from ai_scientist.utils.token_tracker import track_token_usage

# Keep original imports for backward compatibility
import anthropic
import backoff
import openai

# Import OpenRouter integration
try:
    from ai_scientist.openrouter import (
        get_global_client, initialize_openrouter, OpenRouterConfig,
        get_available_models, extract_json_between_markers as openrouter_extract_json
    )
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

MAX_NUM_TOKENS = 4096

# Legacy model names mapped to OpenRouter equivalents
LEGACY_MODEL_MAPPING = {
    "claude-3-5-sonnet-20240620": "anthropic/claude-3.5-sonnet",
    "claude-3-5-sonnet-20241022": "anthropic/claude-3.5-sonnet",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4o-2024-05-13": "openai/gpt-4o",
    "gpt-4o-2024-08-06": "openai/gpt-4o",
    "gpt-4.1": "openai/gpt-4",
    "gpt-4.1-2025-04-14": "openai/gpt-4",
    "gpt-4.1-mini": "openai/gpt-4o-mini",
    "gpt-4.1-mini-2025-04-14": "openai/gpt-4o-mini",
    "o1": "openai/o1",
    "o1-2024-12-17": "openai/o1",
    "o1-preview-2024-09-12": "openai/o1-preview",
    "o1-mini": "openai/o1-mini",
    "o1-mini-2024-09-12": "openai/o1-mini",
    "o3-mini": "openai/o3-mini",
    "o3-mini-2025-01-31": "openai/o3-mini",
    "deepseek-coder-v2-0724": "deepseek/deepseek-coder",
    "deepcoder-14b": "deepseek/deepseek-coder",
    "llama3.1-405b": "meta-llama/llama-3.1-405b-instruct",
    "gemini-2.0-flash": "google/gemini-2.0-flash",
    "gemini-2.5-flash-preview-04-17": "google/gemini-2.5-flash",
    "gemini-2.5-pro-preview-03-25": "google/gemini-2.5-pro",
}

# Expanded available models list with OpenRouter models
AVAILABLE_LLMS = [
    # Legacy names for backward compatibility
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "o1",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
    "deepseek-coder-v2-0724",
    "deepcoder-14b",
    "llama3.1-405b",
    "gemini-2.0-flash",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-preview-03-25",
    # OpenRouter native model names
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-haiku",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/o1",
    "openai/o1-mini",
    "openai/o3-mini",
    "google/gemini-2.0-flash",
    "google/gemini-1.5-pro",
    "meta-llama/llama-3.1-405b-instruct",
    "deepseek/deepseek-v3",
    "x-ai/grok-2",
    # Bedrock models
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    # Vertex AI models
    "vertex_ai/claude-3-opus@20240229",
    "vertex_ai/claude-3-5-sonnet@20240620",
    "vertex_ai/claude-3-5-sonnet@20241022",
    "vertex_ai/claude-3-sonnet@20240229",
    "vertex_ai/claude-3-haiku@20240307",
]

logger = logging.getLogger(__name__)


def _normalize_model_name(model: str) -> str:
    """Normalize model name for OpenRouter compatibility"""
    return LEGACY_MODEL_MAPPING.get(model, model)

def _use_openrouter() -> bool:
    """Check if OpenRouter should be used"""
    return OPENROUTER_AVAILABLE and os.getenv("USE_OPENROUTER", "true").lower() == "true"

def _convert_messages_format(msg_history: List[Dict[str, Any]], system_message: str, user_message: str) -> List[Dict[str, Any]]:
    """Convert message format for OpenRouter"""
    messages = []
    
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    # Add message history
    for msg in msg_history:
        messages.append(msg)
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    return messages

# Get N responses from a single message, used for ensembling.
@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
        anthropic.RateLimitError,
    ),
)
@track_token_usage
def get_batch_responses_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
    n_responses=1,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    """Get multiple responses from LLM for ensembling"""
    
    if msg_history is None:
        msg_history = []
    
    # Use OpenRouter if available
    if _use_openrouter():
        try:
            openrouter_client = get_global_client()
            normalized_model = _normalize_model_name(model)
            
            # Convert to OpenRouter message format
            messages = _convert_messages_format(msg_history, system_message, prompt)
            
            # Get multiple responses using sync wrapper
            contents, message_histories = openrouter_client.get_batch_responses_sync(
                messages=messages,
                model=normalized_model,
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                n_responses=n_responses
            )
            
            if print_debug:
                logger.info(f"OpenRouter batch response ({n_responses} responses):")
                for i, content in enumerate(contents):
                    logger.info(f"Response {i+1}: {content[:100]}...")
            
            return contents, message_histories
            
        except Exception as e:
            logger.warning(f"OpenRouter batch request failed: {e}. Falling back to legacy implementation.")
    
    # Legacy implementation fallback
    msg = prompt
    
    if "gpt" in model or model in LEGACY_MODEL_MAPPING:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=_normalize_model_name(model) if "/" not in model else model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    else:
        # For other models, get individual responses
        content, new_msg_history = [], []
        for _ in range(n_responses):
            c, hist = get_response_from_llm(
                msg,
                client,
                model,
                system_message,
                print_debug=False,
                msg_history=msg_history,
                temperature=temperature,
            )
            content.append(c)
            new_msg_history.append(hist)

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    if "gpt" in model:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
    elif "o1" in model or "o3" in model:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": system_message},
                *prompt,
            ],
            temperature=1,
            n=1,
            seed=0,
        )
    
    else:
        raise ValueError(f"Model {model} not supported.")


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
        anthropic.RateLimitError,
    ),
)
@track_token_usage
def get_response_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
) -> tuple[str, list[dict[str, Any]]]:
    """Get single response from LLM with OpenRouter integration"""
    
    if msg_history is None:
        msg_history = []
    
    # Use OpenRouter if available
    if _use_openrouter():
        try:
            openrouter_client = get_global_client()
            normalized_model = _normalize_model_name(model)
            
            # Convert to OpenRouter message format
            messages = _convert_messages_format(msg_history, system_message, prompt)
            
            # Get response using sync wrapper
            content, new_msg_history = openrouter_client.get_response_sync(
                messages=messages,
                model=normalized_model,
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS
            )
            
            if print_debug:
                logger.info(f"OpenRouter response for model {normalized_model}:")
                logger.info(f"Content: {content[:200]}...")
            
            return content, new_msg_history
            
        except Exception as e:
            logger.warning(f"OpenRouter request failed: {e}. Falling back to legacy implementation.")
    
    # Legacy implementation fallback
    msg = prompt
    
    if "claude" in model:
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        response = client.messages.create(
            model=model,
            max_tokens=MAX_NUM_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
    elif "gpt" in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = make_llm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif "o1" in model or "o3" in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = make_llm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        # Use OpenRouter for all other models
        try:
            if OPENROUTER_AVAILABLE:
                openrouter_client = get_global_client()
                normalized_model = _normalize_model_name(model)
                
                messages = _convert_messages_format(msg_history, system_message, msg)
                content, new_msg_history = openrouter_client.get_response_sync(
                    messages=messages,
                    model=normalized_model,
                    temperature=temperature,
                    max_tokens=MAX_NUM_TOKENS
                )
            else:
                raise ValueError(f"Model {model} not supported and OpenRouter not available.")
        except Exception as e:
            logger.error(f"Failed to process model {model}: {e}")
            raise ValueError(f"Model {model} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg_item in enumerate(new_msg_history):
            role = msg_item.get("role", "unknown")
            content_item = msg_item.get("content", "")
            # Handle both string and list content formats
            if isinstance(content_item, list):
                content_str = " ".join([item.get("text", str(item)) for item in content_item])
            else:
                content_str = str(content_item)
            print(f'{j}, {role}: {content_str}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM output, with OpenRouter compatibility"""
    
    # Try OpenRouter version first if available
    if OPENROUTER_AVAILABLE:
        try:
            result = openrouter_extract_json(llm_output)
            if result:
                return result
        except Exception as e:
            logger.debug(f"OpenRouter JSON extraction failed: {e}")
    
    # Legacy implementation fallback
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found


def create_client(model: str) -> Tuple[Any, str]:
    """Create LLM client with OpenRouter integration and legacy fallback"""
    
    # Use OpenRouter if available and enabled
    if _use_openrouter():
        try:
            openrouter_client = get_global_client()
            normalized_model = _normalize_model_name(model)
            
            print(f"Using OpenRouter with model {normalized_model}.")
            return openrouter_client, normalized_model
        except Exception as e:
            logger.warning(f"OpenRouter client creation failed: {e}. Falling back to legacy clients.")
    
    # Legacy client creation for backward compatibility
    normalized_model = _normalize_model_name(model)
    
    if model.startswith("claude-") or normalized_model.startswith("anthropic/"):
        print(f"Using Anthropic API with model {model}.")
        return anthropic.Anthropic(), model
    elif model.startswith("bedrock") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Amazon Bedrock with model {client_model}.")
        return anthropic.AnthropicBedrock(), client_model
    elif model.startswith("vertex_ai") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Vertex AI with model {client_model}.")
        return anthropic.AnthropicVertex(), client_model
    elif "gpt" in model or normalized_model.startswith("openai/"):
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif "o1" in model or "o3" in model:
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif model == "deepseek-coder-v2-0724":
        print(f"Using DeepSeek API with {model}.")
        return (
            openai.OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            ),
            model,
        )
    elif model == "deepcoder-14b":
        print(f"Using HuggingFace API with {model}.")
        if "HUGGINGFACE_API_KEY" not in os.environ:
            raise ValueError("HUGGINGFACE_API_KEY environment variable not set")
        return (
            openai.OpenAI(
                api_key=os.environ["HUGGINGFACE_API_KEY"],
                base_url="https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview",
            ),
            model,
        )
    elif model == "llama3.1-405b":
        print(f"Using OpenRouter with {model}.")
        return (
            openai.OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
            ),
            "meta-llama/llama-3.1-405b-instruct",
        )
    elif 'gemini' in model:
        print(f"Using Gemini API with {model}.")
        return (
            openai.OpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            model,
        )
    else:
        # Try OpenRouter as final fallback
        if OPENROUTER_AVAILABLE and os.getenv("OPENROUTER_API_KEY"):
            try:
                openrouter_client = get_global_client()
                print(f"Using OpenRouter fallback for model {model}.")
                return openrouter_client, normalized_model
            except Exception as e:
                logger.error(f"OpenRouter fallback failed: {e}")
        
        raise ValueError(f"Model {model} not supported and no OpenRouter fallback available.")
