"""
Unit tests for context window management.
"""

import pytest
from unittest.mock import Mock
from datetime import datetime
from src.context_manager import (
    ContextWindow, ContextManager, truncate_text_to_tokens
)
from src.models import Message


class TestContextWindow:
    """Test ContextWindow class."""
    
    def test_initialization(self):
        """Test context window initialization."""
        window = ContextWindow(max_tokens=2000)
        
        assert window.max_tokens == 2000
        assert window.reserved_tokens == 1000
        assert len(window.messages) == 0
    
    def test_add_message(self):
        """Test adding messages to context."""
        window = ContextWindow()
        message = Message(role="user", content="test", timestamp=datetime.now())
        
        window.add_message(message)
        
        assert window.get_message_count() == 1
        assert window.messages[0] == message
    
    def test_get_message_count(self):
        """Test getting message count."""
        window = ContextWindow()
        
        assert window.get_message_count() == 0
        
        window.add_message(Message(role="user", content="test1", timestamp=datetime.now()))
        window.add_message(Message(role="assistant", content="test2", timestamp=datetime.now()))
        
        assert window.get_message_count() == 2
    
    def test_clear(self):
        """Test clearing context."""
        window = ContextWindow()
        window.add_message(Message(role="user", content="test", timestamp=datetime.now()))
        
        assert window.get_message_count() == 1
        
        window.clear()
        
        assert window.get_message_count() == 0
    
    def test_get_context_string_empty(self):
        """Test getting context string when empty."""
        window = ContextWindow()
        token_counter = lambda text: len(text) // 4
        
        context = window.get_context_string(token_counter)
        
        assert context == ""
    
    def test_get_context_string_within_limit(self):
        """Test getting context string when all messages fit."""
        window = ContextWindow(max_tokens=1000, reserved_tokens=500)
        token_counter = lambda text: len(text) // 4
        
        window.add_message(Message(role="user", content="Hello", timestamp=datetime.now()))
        window.add_message(Message(role="assistant", content="Hi there", timestamp=datetime.now()))
        
        context = window.get_context_string(token_counter)
        
        assert "user: Hello" in context
        assert "assistant: Hi there" in context
    
    def test_get_context_string_truncates_old_messages(self):
        """Test that old messages are truncated when exceeding limit."""
        window = ContextWindow(max_tokens=200, reserved_tokens=20)
        # Available tokens: 180
        token_counter = lambda text: len(text) // 4
        
        # Add messages that exceed limit
        # Each message is ~30 chars = ~7 tokens
        for i in range(30):
            window.add_message(Message(
                role="user",
                content=f"Message {i} with some content",
                timestamp=datetime.now()
            ))
        
        context = window.get_context_string(token_counter, prioritize_recent=True)
        
        # Should only include recent messages
        assert "Message 29" in context  # Most recent
        assert "Message 0" not in context  # Oldest should be truncated
    
    def test_estimate_tokens(self):
        """Test token estimation."""
        window = ContextWindow()
        token_counter = lambda text: len(text) // 4
        
        window.add_message(Message(role="user", content="1234", timestamp=datetime.now()))
        window.add_message(Message(role="assistant", content="5678", timestamp=datetime.now()))
        
        # "user: 1234" = 11 chars = 2 tokens (11//4 = 2)
        # "assistant: 5678" = 17 chars = 4 tokens (17//4 = 4)
        # But there's also a newline between them in estimate_tokens
        # Actually: estimate_tokens just sums individual message tokens
        # "user: 1234" = 11 chars = 2 tokens
        # "assistant: 5678" = 17 chars = 4 tokens  
        # Total = 6 tokens, but 11//4 = 2, 17//4 = 4, so 2+4 = 6
        # Wait, let me recalculate: 11//4 = 2, 17//4 = 4, total = 6
        # But the test is getting 5, so let me check the actual calculation
        # "user: 1234" has 10 chars (u-s-e-r-:-space-1-2-3-4) = 10 chars = 2 tokens
        # "assistant: 5678" has 16 chars = 4 tokens
        # Hmm, let me count: "user: " = 6 chars, "1234" = 4 chars = 10 total
        # "assistant: " = 11 chars, "5678" = 4 chars = 15 total
        # 10//4 = 2, 15//4 = 3, total = 5
        tokens = window.estimate_tokens(token_counter)
        
        assert tokens == 5


class TestContextManager:
    """Test ContextManager class."""
    
    def test_initialization(self):
        """Test context manager initialization."""
        manager = ContextManager(max_tokens=2000)
        
        assert manager.context_window.max_tokens == 2000
        assert manager.summary_cache is None
    
    def test_add_message(self):
        """Test adding message to context manager."""
        manager = ContextManager()
        message = Message(role="user", content="test", timestamp=datetime.now())
        
        manager.add_message(message)
        
        assert manager.context_window.get_message_count() == 1
    
    def test_get_context_without_llm_provider(self):
        """Test getting context without LLM provider."""
        manager = ContextManager()
        
        context = manager.get_context()
        
        assert context == "No context available"
    
    def test_get_context_with_llm_provider(self):
        """Test getting context with LLM provider."""
        mock_provider = Mock()
        mock_provider.count_tokens = lambda text: len(text) // 4
        
        manager = ContextManager(llm_provider=mock_provider)
        manager.add_message(Message(role="user", content="Hello", timestamp=datetime.now()))
        
        context = manager.get_context()
        
        assert "user: Hello" in context
    
    def test_get_context_with_summary(self):
        """Test getting context includes summary."""
        mock_provider = Mock()
        mock_provider.count_tokens = lambda text: len(text) // 4
        
        manager = ContextManager(llm_provider=mock_provider)
        manager.summary_cache = "Previous conversation summary"
        manager.add_message(Message(role="user", content="Hello", timestamp=datetime.now()))
        
        context = manager.get_context()
        
        assert "Previous conversation summary" in context
        assert "user: Hello" in context
    
    def test_clear(self):
        """Test clearing context manager."""
        manager = ContextManager()
        manager.add_message(Message(role="user", content="test", timestamp=datetime.now()))
        manager.summary_cache = "summary"
        
        manager.clear()
        
        assert manager.context_window.get_message_count() == 0
        assert manager.summary_cache is None
    
    def test_get_stats_without_llm(self):
        """Test getting stats without LLM provider."""
        manager = ContextManager()
        manager.add_message(Message(role="user", content="test", timestamp=datetime.now()))
        
        stats = manager.get_stats()
        
        assert stats["message_count"] == 1
        assert stats["has_summary"] is False
        assert stats["max_tokens"] == 4000
        assert "estimated_tokens" not in stats
    
    def test_get_stats_with_llm(self):
        """Test getting stats with LLM provider."""
        mock_provider = Mock()
        mock_provider.count_tokens = lambda text: len(text) // 4
        
        manager = ContextManager(llm_provider=mock_provider)
        manager.add_message(Message(role="user", content="test", timestamp=datetime.now()))
        
        stats = manager.get_stats()
        
        assert stats["message_count"] == 1
        assert "estimated_tokens" in stats
        assert "usage_ratio" in stats
    
    def test_summarization_not_triggered_below_threshold(self):
        """Test that summarization is not triggered below threshold."""
        mock_provider = Mock()
        mock_provider.count_tokens = lambda text: 10  # Low token count
        
        manager = ContextManager(max_tokens=2000, llm_provider=mock_provider)
        
        # Add a few messages
        for i in range(3):
            manager.add_message(Message(role="user", content=f"msg {i}", timestamp=datetime.now()))
        
        # Should not have summary
        assert manager.summary_cache is None
    
    def test_summarization_triggered_above_threshold(self):
        """Test that summarization is triggered above threshold."""
        mock_provider = Mock()
        # High token count to trigger summarization
        mock_provider.count_tokens = lambda text: 800
        
        # Mock LLM response for summarization
        mock_response = Mock()
        mock_response.content = "Summary of conversation"
        mock_provider.generate_with_cache = Mock(return_value=mock_response)
        
        manager = ContextManager(max_tokens=2000, llm_provider=mock_provider)
        
        # Add messages to trigger summarization
        for i in range(5):
            manager.add_message(Message(role="user", content=f"msg {i}", timestamp=datetime.now()))
        
        # Should have created summary
        assert manager.summary_cache is not None
        assert "Summary of conversation" in manager.summary_cache


class TestTruncateTextToTokens:
    """Test truncate_text_to_tokens function."""
    
    def test_truncate_text_within_limit(self):
        """Test text within limit is not truncated."""
        text = "This is a short text"
        token_counter = lambda t: len(t) // 4
        
        result = truncate_text_to_tokens(text, max_tokens=10, token_counter=token_counter)
        
        assert result == text
    
    def test_truncate_text_exceeds_limit(self):
        """Test text exceeding limit is truncated."""
        text = "This is a much longer text that needs to be truncated"
        token_counter = lambda t: len(t) // 4
        
        result = truncate_text_to_tokens(text, max_tokens=5, token_counter=token_counter)
        
        assert len(result) < len(text)
        assert result.endswith("...")
    
    def test_truncate_text_exact_limit(self):
        """Test text at exact limit is not truncated."""
        text = "1234567890123456"  # 16 chars = 4 tokens
        token_counter = lambda t: len(t) // 4
        
        result = truncate_text_to_tokens(text, max_tokens=4, token_counter=token_counter)
        
        assert result == text
