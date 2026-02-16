"""
Context window management for LLM interactions.

This module manages conversation context to stay within token limits,
with truncation and summarization strategies.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from src.core.models import Message

logger = logging.getLogger(__name__)


@dataclass
class ContextWindow:
    """Manages context window for LLM interactions."""
    max_tokens: int = 4000  # Conservative limit for most models
    reserved_tokens: int = 1000  # Reserve for prompt template and response
    messages: List[Message] = field(default_factory=list)
    
    def add_message(self, message: Message) -> None:
        """
        Add message to context.
        
        Args:
            message: Message to add
        """
        self.messages.append(message)
        logger.debug(f"Added message to context: {message.role}")
    
    def get_context_string(self, token_counter, prioritize_recent: bool = True) -> str:
        """
        Get context string within token limits.
        
        Args:
            token_counter: Function to count tokens in text
            prioritize_recent: If True, keep most recent messages
            
        Returns:
            Context string within token limits
        """
        if not self.messages:
            return ""
        
        available_tokens = self.max_tokens - self.reserved_tokens
        
        # Ensure we have at least some tokens available
        if available_tokens <= 0:
            logger.warning(f"No tokens available for context (max={self.max_tokens}, reserved={self.reserved_tokens})")
            # Return at least the most recent message
            if self.messages:
                msg = self.messages[-1]
                return f"{msg.role}: {msg.content}"
            return ""
        
        if prioritize_recent:
            return self._get_recent_context(token_counter, available_tokens)
        else:
            return self._get_balanced_context(token_counter, available_tokens)
    
    def _get_recent_context(self, token_counter, available_tokens: int) -> str:
        """
        Get most recent messages that fit within token limit.
        
        Args:
            token_counter: Function to count tokens
            available_tokens: Available token budget
            
        Returns:
            Context string with recent messages
        """
        context_parts = []
        current_tokens = 0
        
        # Iterate from most recent to oldest
        for message in reversed(self.messages):
            message_text = f"{message.role}: {message.content}"
            message_tokens = token_counter(message_text)
            
            if current_tokens + message_tokens <= available_tokens:
                context_parts.insert(0, message_text)
                current_tokens += message_tokens
            else:
                # Can't fit more messages
                logger.info(f"Context truncated: keeping {len(context_parts)} most recent messages")
                break
        
        return "\n\n".join(context_parts)
    
    def _get_balanced_context(self, token_counter, available_tokens: int) -> str:
        """
        Get balanced context with summary of old messages and recent messages.
        
        Args:
            token_counter: Function to count tokens
            available_tokens: Available token budget
            
        Returns:
            Context string with balanced content
        """
        # Reserve 30% for summary, 70% for recent messages
        summary_tokens = int(available_tokens * 0.3)
        recent_tokens = available_tokens - summary_tokens
        
        # Get recent messages
        recent_context = []
        current_tokens = 0
        
        for message in reversed(self.messages):
            message_text = f"{message.role}: {message.content}"
            message_tokens = token_counter(message_text)
            
            if current_tokens + message_tokens <= recent_tokens:
                recent_context.insert(0, message_text)
                current_tokens += message_tokens
            else:
                break
        
        # If we have older messages, create a summary placeholder
        num_recent = len(recent_context)
        num_total = len(self.messages)
        
        if num_recent < num_total:
            summary = f"[Earlier conversation: {num_total - num_recent} messages summarized]"
            return summary + "\n\n" + "\n\n".join(recent_context)
        else:
            return "\n\n".join(recent_context)
    
    def clear(self) -> None:
        """Clear all messages from context."""
        self.messages.clear()
        logger.info("Context cleared")
    
    def get_message_count(self) -> int:
        """Get number of messages in context."""
        return len(self.messages)
    
    def estimate_tokens(self, token_counter) -> int:
        """
        Estimate total tokens in current context.
        
        Args:
            token_counter: Function to count tokens
            
        Returns:
            Estimated token count
        """
        total = 0
        for message in self.messages:
            message_text = f"{message.role}: {message.content}"
            total += token_counter(message_text)
        return total


class ContextManager:
    """
    High-level context manager with summarization support.
    
    This class manages conversation context with automatic summarization
    when approaching token limits.
    """
    
    def __init__(self, max_tokens: int = 4000, llm_provider=None):
        """
        Initialize context manager.
        
        Args:
            max_tokens: Maximum tokens for context window
            llm_provider: LLM provider for summarization (optional)
        """
        self.context_window = ContextWindow(max_tokens=max_tokens)
        self.llm_provider = llm_provider
        self.summarization_threshold = 0.8  # Summarize at 80% capacity
        self.summary_cache: Optional[str] = None
    
    def add_message(self, message: Message) -> None:
        """
        Add message to context with automatic summarization.
        
        Args:
            message: Message to add
        """
        self.context_window.add_message(message)
        
        # Check if we need to summarize
        if self.llm_provider:
            self._check_and_summarize()
    
    def get_context(self, prioritize_recent: bool = True) -> str:
        """
        Get context string for LLM prompt.
        
        Args:
            prioritize_recent: If True, prioritize recent messages
            
        Returns:
            Context string
        """
        if not self.llm_provider:
            return "No context available"
        
        token_counter = self.llm_provider.count_tokens
        
        # Include summary if available
        context = self.context_window.get_context_string(token_counter, prioritize_recent)
        
        if self.summary_cache:
            context = f"{self.summary_cache}\n\n{context}"
        
        return context
    
    def _check_and_summarize(self) -> None:
        """Check if summarization is needed and perform it."""
        if not self.llm_provider:
            return
        
        token_counter = self.llm_provider.count_tokens
        current_tokens = self.context_window.estimate_tokens(token_counter)
        max_tokens = self.context_window.max_tokens - self.context_window.reserved_tokens
        
        if max_tokens <= 0:
            logger.warning("Invalid token configuration: max_tokens <= reserved_tokens")
            return
        
        usage_ratio = current_tokens / max_tokens
        
        if usage_ratio >= self.summarization_threshold:
            logger.info(f"Context usage at {usage_ratio:.1%}, triggering summarization")
            self._summarize_old_context()
    
    def _summarize_old_context(self) -> None:
        """Summarize older messages to free up context space."""
        # Keep last 3 messages, summarize the rest
        messages_to_keep = 3
        
        if len(self.context_window.messages) <= messages_to_keep:
            logger.info("Not enough messages to summarize")
            return
        
        # Get messages to summarize
        messages_to_summarize = self.context_window.messages[:-messages_to_keep]
        recent_messages = self.context_window.messages[-messages_to_keep:]
        
        # Create summary using LLM
        try:
            from src.llm.prompt_templates import PromptTemplates
            
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in messages_to_summarize
            ]
            
            prompt = PromptTemplates.format_context_summary_prompt(history)
            response = self.llm_provider.generate_with_cache(prompt, temperature=0.3)
            
            self.summary_cache = f"[Previous conversation summary]\n{response.content}"
            
            # Replace old messages with recent ones
            self.context_window.messages = recent_messages
            
            logger.info(f"Summarized {len(messages_to_summarize)} messages")
            
        except Exception as e:
            logger.error(f"Failed to summarize context: {str(e)}")
            # Fallback: just truncate old messages
            self.context_window.messages = recent_messages
    
    def clear(self) -> None:
        """Clear context and summary."""
        self.context_window.clear()
        self.summary_cache = None
        logger.info("Context manager cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get context statistics.
        
        Returns:
            Dictionary with context stats
        """
        stats = {
            "message_count": self.context_window.get_message_count(),
            "has_summary": self.summary_cache is not None,
            "max_tokens": self.context_window.max_tokens
        }
        
        if self.llm_provider:
            token_counter = self.llm_provider.count_tokens
            stats["estimated_tokens"] = self.context_window.estimate_tokens(token_counter)
            stats["usage_ratio"] = stats["estimated_tokens"] / (
                self.context_window.max_tokens - self.context_window.reserved_tokens
            )
        
        return stats


def truncate_text_to_tokens(text: str, max_tokens: int, token_counter) -> str:
    """
    Truncate text to fit within token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        token_counter: Function to count tokens
        
    Returns:
        Truncated text
    """
    current_tokens = token_counter(text)
    
    if current_tokens <= max_tokens:
        return text
    
    # Binary search for the right length
    left, right = 0, len(text)
    result = text
    
    while left < right:
        mid = (left + right + 1) // 2
        truncated = text[:mid]
        tokens = token_counter(truncated)
        
        if tokens <= max_tokens:
            result = truncated
            left = mid
        else:
            right = mid - 1
    
    return result + "..."
