"""
Unit tests for LLM provider abstraction.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from src.llm_provider import (
    LLMProvider, LLMResponse, GeminiProvider, OpenAIProvider, 
    LLMProviderFactory
)


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, model: str = "mock-model", api_key: str = "test-key", 
                 max_retries: int = 3, fail_count: int = 0):
        super().__init__(model, api_key, max_retries)
        self.call_count = 0
        self.fail_count = fail_count  # Number of times to fail before succeeding
    
    def generate(self, prompt: str, temperature: float = 0.7,
                max_tokens: int = None) -> LLMResponse:
        """Mock generate method."""
        self.call_count += 1
        
        # Simulate failures
        if self.call_count <= self.fail_count:
            raise Exception(f"Mock failure {self.call_count}")
        
        return LLMResponse(
            content=f"Mock response to: {prompt[:20]}...",
            tokens_used=100,
            model=self.model
        )
    
    def count_tokens(self, text: str) -> int:
        """Mock token counter (1 token per 4 characters)."""
        return len(text) // 4


class TestLLMProvider:
    """Test LLMProvider base class."""
    
    def test_cache_hit(self):
        """Test that identical prompts return cached responses."""
        provider = MockLLMProvider()
        
        # First call
        response1 = provider.generate_with_cache("test prompt")
        assert not response1.cached
        assert provider.call_count == 1
        
        # Second call with same prompt should hit cache
        response2 = provider.generate_with_cache("test prompt")
        assert response2.cached
        assert provider.call_count == 1  # No additional call
        assert response1.content == response2.content
    
    def test_cache_miss_different_prompt(self):
        """Test that different prompts don't hit cache."""
        provider = MockLLMProvider()
        
        response1 = provider.generate_with_cache("prompt 1")
        response2 = provider.generate_with_cache("prompt 2")
        
        assert not response1.cached
        assert not response2.cached
        assert provider.call_count == 2
    
    def test_cache_miss_different_temperature(self):
        """Test that different temperatures don't hit cache."""
        provider = MockLLMProvider()
        
        response1 = provider.generate_with_cache("test", temperature=0.5)
        response2 = provider.generate_with_cache("test", temperature=0.7)
        
        assert not response1.cached
        assert not response2.cached
        assert provider.call_count == 2
    
    def test_clear_cache(self):
        """Test cache clearing."""
        provider = MockLLMProvider()
        
        # Generate and cache
        provider.generate_with_cache("test")
        assert provider.call_count == 1
        
        # Clear cache
        provider.clear_cache()
        
        # Should generate again
        provider.generate_with_cache("test")
        assert provider.call_count == 2
    
    def test_retry_success_after_failure(self):
        """Test retry logic succeeds after initial failures."""
        # Fail once, then succeed
        provider = MockLLMProvider(fail_count=1)
        
        response = provider.generate_with_retry("test")
        
        assert response.content.startswith("Mock response")
        assert provider.call_count == 2  # Failed once, succeeded on retry
    
    def test_retry_all_attempts_fail(self):
        """Test retry logic exhausts all attempts."""
        # Fail more times than max_retries
        provider = MockLLMProvider(max_retries=3, fail_count=5)
        
        with pytest.raises(Exception, match="Mock failure"):
            provider.generate_with_retry("test")
        
        assert provider.call_count == 3  # All retries exhausted
    
    def test_retry_exponential_backoff(self):
        """Test exponential backoff timing."""
        provider = MockLLMProvider(max_retries=3, fail_count=2)
        
        start_time = time.time()
        provider.generate_with_retry("test")
        elapsed = time.time() - start_time
        
        # Should wait 1s + 2s = 3s total (with some tolerance)
        assert elapsed >= 2.5  # Allow some tolerance
        assert elapsed < 4.0


class TestLLMProviderFactory:
    """Test LLMProviderFactory."""
    
    def test_create_gemini_provider(self):
        """Test creating Gemini provider."""
        with patch('google.generativeai.configure') as mock_configure:
            with patch('google.generativeai.GenerativeModel') as mock_model:
                provider = LLMProviderFactory.create_provider(
                    "gemini", 
                    model="gemini-pro",
                    api_key="test-key"
                )
                
                assert isinstance(provider, GeminiProvider)
                assert provider.model == "gemini-pro"
    
    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        with patch('openai.OpenAI') as mock_openai:
            provider = LLMProviderFactory.create_provider(
                "openai",
                model="gpt-4",
                api_key="test-key"
            )
            
            assert isinstance(provider, OpenAIProvider)
            assert provider.model == "gpt-4"
    
    def test_create_provider_default_models(self):
        """Test default model selection."""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                provider = LLMProviderFactory.create_provider(
                    "gemini",
                    api_key="test-key"
                )
                assert provider.model == "gemini-pro"
        
        with patch('openai.OpenAI'):
            provider = LLMProviderFactory.create_provider(
                "openai",
                api_key="test-key"
            )
            assert provider.model == "gpt-4"
    
    def test_create_provider_invalid_type(self):
        """Test error on invalid provider type."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            LLMProviderFactory.create_provider(
                "invalid",
                api_key="test-key"
            )
    
    def test_create_provider_missing_api_key(self):
        """Test error when API key is missing."""
        with pytest.raises(ValueError, match="API key is required"):
            LLMProviderFactory.create_provider("gemini")


class TestGeminiProvider:
    """Test GeminiProvider."""
    
    def test_initialization(self):
        """Test Gemini provider initialization."""
        with patch('google.generativeai.configure') as mock_configure:
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                mock_model = Mock()
                mock_model_class.return_value = mock_model
                
                provider = GeminiProvider(api_key="test-key")
                
                assert provider.model == "gemini-pro"
                mock_configure.assert_called_once_with(api_key="test-key")
    
    def test_generate(self):
        """Test Gemini generate method."""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                # Mock response
                mock_response = Mock()
                mock_response.text = "Generated text"
                mock_response.usage_metadata = Mock()
                mock_response.usage_metadata.prompt_token_count = 10
                mock_response.usage_metadata.candidates_token_count = 20
                mock_response.candidates = [Mock()]
                mock_response.candidates[0].finish_reason.name = "STOP"
                
                mock_model = Mock()
                mock_model.generate_content.return_value = mock_response
                mock_model_class.return_value = mock_model
                
                provider = GeminiProvider(api_key="test-key")
                response = provider.generate("test prompt")
                
                assert response.content == "Generated text"
                assert response.tokens_used == 30
                assert response.model == "gemini-pro"
    
    def test_count_tokens(self):
        """Test Gemini token counting."""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                mock_model = Mock()
                mock_result = Mock()
                mock_result.total_tokens = 42
                mock_model.count_tokens.return_value = mock_result
                mock_model_class.return_value = mock_model
                
                provider = GeminiProvider(api_key="test-key")
                count = provider.count_tokens("test text")
                
                assert count == 42


class TestOpenAIProvider:
    """Test OpenAIProvider."""
    
    def test_initialization(self):
        """Test OpenAI provider initialization."""
        with patch('openai.OpenAI') as mock_openai:
            provider = OpenAIProvider(api_key="test-key")
            
            assert provider.model == "gpt-4"
            mock_openai.assert_called_once_with(api_key="test-key")
    
    def test_generate(self):
        """Test OpenAI generate method."""
        with patch('openai.OpenAI') as mock_openai_class:
            # Mock response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Generated text"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = Mock()
            mock_response.usage.total_tokens = 50
            mock_response.usage.prompt_tokens = 20
            mock_response.usage.completion_tokens = 30
            
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            provider = OpenAIProvider(api_key="test-key")
            response = provider.generate("test prompt")
            
            assert response.content == "Generated text"
            assert response.tokens_used == 50
            assert response.model == "gpt-4"
    
    def test_count_tokens_with_tiktoken(self):
        """Test OpenAI token counting with tiktoken."""
        with patch('openai.OpenAI'):
            with patch('tiktoken.encoding_for_model') as mock_encoding_for_model:
                mock_encoding = Mock()
                mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
                mock_encoding_for_model.return_value = mock_encoding
                
                provider = OpenAIProvider(api_key="test-key")
                count = provider.count_tokens("test text")
                
                assert count == 5
    
    def test_count_tokens_fallback(self):
        """Test OpenAI token counting fallback without tiktoken."""
        with patch('openai.OpenAI'):
            provider = OpenAIProvider(api_key="test-key")
            
            # Mock tiktoken import failure by patching the import in count_tokens
            import sys
            original_tiktoken = sys.modules.get('tiktoken')
            sys.modules['tiktoken'] = None
            
            try:
                count = provider.count_tokens("test" * 10)  # 40 characters
                
                # Fallback: ~1 token per 4 chars
                assert count == 10
            finally:
                if original_tiktoken:
                    sys.modules['tiktoken'] = original_tiktoken
                else:
                    sys.modules.pop('tiktoken', None)


class TestLLMResponse:
    """Test LLMResponse dataclass."""
    
    def test_initialization(self):
        """Test LLMResponse initialization."""
        response = LLMResponse(
            content="test",
            tokens_used=10,
            model="test-model"
        )
        
        assert response.content == "test"
        assert response.tokens_used == 10
        assert response.model == "test-model"
        assert not response.cached
        assert response.metadata == {}
    
    def test_with_metadata(self):
        """Test LLMResponse with metadata."""
        response = LLMResponse(
            content="test",
            tokens_used=10,
            model="test-model",
            metadata={"key": "value"}
        )
        
        assert response.metadata == {"key": "value"}
