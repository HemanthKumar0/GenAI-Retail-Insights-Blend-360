"""
LLM Provider abstraction for multi-provider support.

This module provides a unified interface for interacting with different LLM providers
(Gemini, OpenAI) with built-in retry logic, caching, and context management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
import hashlib
import json
import time
from functools import lru_cache
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM provider."""
    content: str
    tokens_used: int
    model: str
    cached: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLMProvider(ABC):
    """Base interface for LLM providers."""
    
    def __init__(self, model: str, api_key: str, max_retries: int = 3):
        """
        Initialize LLM provider.
        
        Args:
            model: Model identifier (e.g., "gemini-pro", "gpt-4")
            api_key: API key for authentication
            max_retries: Maximum number of retry attempts
        """
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self._response_cache: Dict[str, LLMResponse] = {}
        
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7, 
                max_tokens: Optional[int] = None) -> LLMResponse:
        """
        Generate response from LLM.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse object
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Token count
        """
        pass
    
    def generate_with_cache(self, prompt: str, temperature: float = 0.7,
                           max_tokens: Optional[int] = None) -> LLMResponse:
        """
        Generate response with caching.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse object (may be cached)
        """
        # Create cache key from prompt and parameters
        cache_key = self._create_cache_key(prompt, temperature, max_tokens)
        
        # Check cache
        if cache_key in self._response_cache:
            logger.info("Cache hit for prompt")
            cached_response = self._response_cache[cache_key]
            cached_response.cached = True
            return cached_response
        
        # Generate new response
        logger.info("Cache miss, generating new response")
        response = self.generate_with_retry(prompt, temperature, max_tokens)
        
        # Store in cache
        self._response_cache[cache_key] = response
        
        return response
    
    def generate_with_retry(self, prompt: str, temperature: float = 0.7,
                           max_tokens: Optional[int] = None) -> LLMResponse:
        """
        Generate response with exponential backoff retry.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse object
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"LLM generation attempt {attempt + 1}/{self.max_retries}")
                return self.generate(prompt, temperature, max_tokens)
            except Exception as e:
                last_exception = e
                logger.warning(f"LLM generation failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        # All retries failed
        logger.error(f"All {self.max_retries} retry attempts failed")
        raise last_exception
    
    def _create_cache_key(self, prompt: str, temperature: float, 
                         max_tokens: Optional[int]) -> str:
        """Create cache key from prompt and parameters."""
        key_data = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "model": self.model
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear response cache."""
        self._response_cache.clear()
        logger.info("Response cache cleared")


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self, model: str = "gemini-pro", api_key: str = None, max_retries: int = 3):
        """
        Initialize Gemini provider.
        
        Args:
            model: Gemini model name
            api_key: Google API key
            max_retries: Maximum retry attempts
        """
        super().__init__(model, api_key, max_retries)
        
        try:
            import google.generativeai as genai
            self.genai = genai
            self.genai.configure(api_key=api_key)
            self.client = self.genai.GenerativeModel(model)
            logger.info(f"Initialized Gemini provider with model: {model}")
        except ImportError:
            raise ImportError("google-generativeai package not installed. "
                            "Install with: pip install google-generativeai")
    
    def generate(self, prompt: str, temperature: float = 0.7,
                max_tokens: Optional[int] = None) -> LLMResponse:
        """Generate response using Gemini API."""
        generation_config = {
            "temperature": temperature,
        }
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
        
        response = self.client.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Extract token usage (Gemini provides this in usage_metadata)
        tokens_used = 0
        if hasattr(response, 'usage_metadata'):
            tokens_used = (response.usage_metadata.prompt_token_count + 
                          response.usage_metadata.candidates_token_count)
        
        return LLMResponse(
            content=response.text,
            tokens_used=tokens_used,
            model=self.model,
            metadata={"finish_reason": response.candidates[0].finish_reason.name if response.candidates else None}
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's tokenizer."""
        result = self.client.count_tokens(text)
        return result.total_tokens


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, model: str = "gpt-4", api_key: str = None, max_retries: int = 3):
        """
        Initialize OpenAI provider.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key
            max_retries: Maximum retry attempts
        """
        super().__init__(model, api_key, max_retries)
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            logger.info(f"Initialized OpenAI provider with model: {model}")
        except ImportError:
            raise ImportError("openai package not installed. "
                            "Install with: pip install openai")
    
    def generate(self, prompt: str, temperature: float = 0.7,
                max_tokens: Optional[int] = None) -> LLMResponse:
        """Generate response using OpenAI API."""
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        
        response = self.client.chat.completions.create(**kwargs)
        
        return LLMResponse(
            content=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            model=self.model,
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken.
        
        Note: This is an approximation. For exact counts, use the API.
        """
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: rough approximation (1 token ≈ 4 characters)
            logger.warning("tiktoken not installed, using rough approximation")
            return len(text) // 4


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider_type: str, model: str = None, 
                       api_key: str = None, max_retries: int = 3) -> LLMProvider:
        """
        Create LLM provider instance.
        
        Args:
            provider_type: "gemini" or "openai"
            model: Model name (uses default if None)
            api_key: API key (required)
            max_retries: Maximum retry attempts
            
        Returns:
            LLMProvider instance
            
        Raises:
            ValueError: If provider_type is invalid
        """
        if not api_key:
            raise ValueError("API key is required")
        
        provider_type = provider_type.lower()
        
        if provider_type == "gemini":
            model = model or "gemini-pro"
            return GeminiProvider(model=model, api_key=api_key, max_retries=max_retries)
        elif provider_type == "openai":
            model = model or "gpt-4"
            return OpenAIProvider(model=model, api_key=api_key, max_retries=max_retries)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}. "
                           f"Supported: 'gemini', 'openai'")
