"""
Unit tests for ErrorHandler and ProgressFeedback.

**Validates: Requirements 12.1, 12.2, 12.5**
"""

import pytest
import time
from unittest.mock import patch, Mock

from src.error_handler import (
    ErrorHandler, ErrorType, ProgressFeedback, ErrorContext
)


class TestErrorHandler:
    """Test suite for ErrorHandler class."""
    
    @pytest.fixture
    def error_handler(self):
        """Create an ErrorHandler instance."""
        return ErrorHandler()
    
    def test_initialization(self, error_handler):
        """Test ErrorHandler initialization."""
        assert error_handler.error_log == []
    
    def test_format_query_parsing_error(self, error_handler):
        """
        Test formatting of query parsing errors.
        
        **Validates: Requirement 12.1**
        """
        error = ValueError("Invalid query syntax")
        user_query = "What were sales?"
        
        message = error_handler.format_user_friendly_error(
            error=error,
            error_type=ErrorType.QUERY_PARSING_ERROR,
            user_query=user_query
        )
        
        # Verify user-friendly message
        assert "trouble understanding" in message.lower()
        assert user_query in message
        assert "Suggestions" in message
        assert "rephras" in message.lower()
        
        # Verify no stack trace in user message
        assert "Traceback" not in message
        assert "ValueError" not in message
    
    def test_format_data_loading_error(self, error_handler):
        """
        Test formatting of data loading errors.
        
        **Validates: Requirement 12.1**
        """
        error = FileNotFoundError("File not found")
        
        message = error_handler.format_user_friendly_error(
            error=error,
            error_type=ErrorType.DATA_LOADING_ERROR
        )
        
        assert "loading your data" in message.lower()
        assert "file exists" in message.lower()
        assert "supported" in message.lower()
    
    def test_format_rate_limit_error(self, error_handler):
        """
        Test formatting of API rate limit errors.
        
        **Validates: Requirement 12.5**
        """
        error = Exception("Rate limit exceeded")
        
        message = error_handler.format_user_friendly_error(
            error=error,
            error_type=ErrorType.API_RATE_LIMIT
        )
        
        assert "rate limit" in message.lower()
        assert "wait" in message.lower()
        assert "30-60 seconds" in message.lower()
    
    def test_format_timeout_error(self, error_handler):
        """Test formatting of timeout errors."""
        error = TimeoutError("Query timeout")
        user_query = "Show me all sales data"
        
        message = error_handler.format_user_friendly_error(
            error=error,
            error_type=ErrorType.TIMEOUT_ERROR,
            user_query=user_query
        )
        
        assert "took too long" in message.lower() or "timed out" in message.lower()
        assert user_query in message
        assert "smaller" in message.lower() or "simplif" in message.lower()
    
    def test_format_ambiguous_query_error(self, error_handler):
        """Test formatting of ambiguous query errors."""
        error = ValueError("Ambiguous query")
        user_query = "What about sales?"
        
        message = error_handler.format_user_friendly_error(
            error=error,
            error_type=ErrorType.AMBIGUOUS_QUERY,
            user_query=user_query
        )
        
        assert "ambiguous" in message.lower()
        assert "clarification" in message.lower() or "specific" in message.lower()
        assert user_query in message
    
    def test_format_unknown_error(self, error_handler):
        """Test formatting of unknown errors."""
        error = Exception("Something went wrong")
        
        message = error_handler.format_user_friendly_error(
            error=error,
            error_type=ErrorType.UNKNOWN_ERROR
        )
        
        assert "unexpected" in message.lower()
        assert "try again" in message.lower()
    
    def test_suggest_alternatives_parsing_error(self, error_handler):
        """
        Test alternative suggestions for parsing errors.
        
        **Validates: Requirement 12.2**
        """
        suggestions = error_handler.suggest_alternatives(
            failed_query="asdf qwerty",
            error_type=ErrorType.QUERY_PARSING_ERROR
        )
        
        assert len(suggestions) > 0
        assert any("Try:" in s for s in suggestions)
        assert any("sales" in s.lower() for s in suggestions)
    
    def test_suggest_alternatives_execution_error(self, error_handler):
        """
        Test alternative suggestions for execution errors.
        
        **Validates: Requirement 12.2**
        """
        suggestions = error_handler.suggest_alternatives(
            failed_query="Show me xyz data",
            error_type=ErrorType.QUERY_EXECUTION_ERROR
        )
        
        assert len(suggestions) > 0
        assert any("available" in s.lower() for s in suggestions)
    
    def test_suggest_alternatives_ambiguous_query(self, error_handler):
        """Test alternative suggestions for ambiguous queries."""
        suggestions = error_handler.suggest_alternatives(
            failed_query="What about that?",
            error_type=ErrorType.AMBIGUOUS_QUERY
        )
        
        assert len(suggestions) > 0
        assert any("specific" in s.lower() for s in suggestions)
    
    def test_suggest_alternatives_timeout(self, error_handler):
        """Test alternative suggestions for timeout errors."""
        suggestions = error_handler.suggest_alternatives(
            failed_query="Analyze all data",
            error_type=ErrorType.TIMEOUT_ERROR
        )
        
        assert len(suggestions) > 0
        assert any("time period" in s.lower() or "smaller" in s.lower() for s in suggestions)
    
    def test_create_data_quality_warning(self, error_handler):
        """Test creating data quality warnings."""
        warning = error_handler.create_data_quality_warning(
            issue_type="Missing Values",
            description="Some rows have missing sales data",
            impact="Calculations may be incomplete",
            affected_count=15
        )
        
        assert "Warning" in warning
        assert "Missing Values" in warning
        assert "missing sales data" in warning
        assert "15" in warning
        assert "impact" in warning.lower()
    
    def test_error_logging(self, error_handler):
        """
        Test that errors are logged with full details.
        
        **Validates: Requirement 12.4**
        """
        error = ValueError("Test error")
        user_query = "Test query"
        
        # Format error (which triggers logging)
        error_handler.format_user_friendly_error(
            error=error,
            error_type=ErrorType.QUERY_PARSING_ERROR,
            user_query=user_query
        )
        
        # Verify error was logged
        assert len(error_handler.error_log) == 1
        
        log_entry = error_handler.error_log[0]
        assert log_entry["error_type"] == ErrorType.QUERY_PARSING_ERROR.value
        assert log_entry["error_message"] == "Test error"
        assert log_entry["user_query"] == user_query
        assert log_entry["stack_trace"] is not None
        assert "timestamp" in log_entry
    
    def test_get_error_log(self, error_handler):
        """Test retrieving error log."""
        # Create some errors
        for i in range(3):
            error_handler.format_user_friendly_error(
                error=Exception(f"Error {i}"),
                error_type=ErrorType.UNKNOWN_ERROR
            )
        
        # Get log
        log = error_handler.get_error_log()
        assert len(log) == 3
        assert all("timestamp" in entry for entry in log)
    
    def test_clear_error_log(self, error_handler):
        """Test clearing error log."""
        # Add some errors
        error_handler.format_user_friendly_error(
            error=Exception("Test"),
            error_type=ErrorType.UNKNOWN_ERROR
        )
        
        assert len(error_handler.error_log) > 0
        
        # Clear
        error_handler.clear_error_log()
        assert len(error_handler.error_log) == 0
    
    def test_multiple_error_types(self, error_handler):
        """Test handling multiple error types."""
        error_types = [
            ErrorType.QUERY_PARSING_ERROR,
            ErrorType.DATA_LOADING_ERROR,
            ErrorType.QUERY_EXECUTION_ERROR,
            ErrorType.VALIDATION_ERROR,
            ErrorType.API_RATE_LIMIT,
            ErrorType.API_ERROR,
            ErrorType.TIMEOUT_ERROR,
            ErrorType.AMBIGUOUS_QUERY,
            ErrorType.UNKNOWN_ERROR
        ]
        
        for error_type in error_types:
            message = error_handler.format_user_friendly_error(
                error=Exception("Test error"),
                error_type=error_type,
                user_query="Test query"
            )
            
            # All messages should be user-friendly
            assert len(message) > 0
            assert "Traceback" not in message
            assert "Exception" not in message


class TestProgressFeedback:
    """Test suite for ProgressFeedback class."""
    
    def test_initialization(self):
        """Test ProgressFeedback initialization."""
        feedback = ProgressFeedback("test_operation", threshold_seconds=5.0)
        
        assert feedback.operation_name == "test_operation"
        assert feedback.threshold_seconds == 5.0
        assert feedback.feedback_shown is False
    
    def test_no_progress_before_threshold(self):
        """Test that no progress is shown before threshold."""
        feedback = ProgressFeedback("test_operation", threshold_seconds=10.0)
        
        # Check immediately
        message = feedback.check_and_show_progress()
        assert message is None
        assert not feedback.is_long_running()
    
    def test_progress_after_threshold(self):
        """Test that progress is shown after threshold."""
        feedback = ProgressFeedback("test_operation", threshold_seconds=0.1)
        
        # Wait for threshold
        time.sleep(0.15)
        
        # Check progress
        message = feedback.check_and_show_progress()
        assert message is not None
        assert "test_operation" in message
        assert feedback.is_long_running()
    
    def test_progress_shown_only_once(self):
        """Test that progress message is shown only once."""
        feedback = ProgressFeedback("test_operation", threshold_seconds=0.1)
        
        # Wait for threshold
        time.sleep(0.15)
        
        # First check
        message1 = feedback.check_and_show_progress()
        assert message1 is not None
        
        # Second check
        message2 = feedback.check_and_show_progress()
        assert message2 is None  # Should not show again
    
    def test_custom_progress_message(self):
        """Test custom progress message."""
        feedback = ProgressFeedback("test_operation", threshold_seconds=0.1)
        
        time.sleep(0.15)
        
        custom_message = "Custom progress update"
        message = feedback.check_and_show_progress(custom_message)
        
        assert message == custom_message
    
    def test_get_elapsed_time(self):
        """Test getting elapsed time."""
        feedback = ProgressFeedback("test_operation")
        
        time.sleep(0.1)
        
        elapsed = feedback.get_elapsed_time()
        assert elapsed >= 0.1
        assert elapsed < 1.0  # Should be less than 1 second
    
    def test_is_long_running(self):
        """Test checking if operation is long-running."""
        feedback = ProgressFeedback("test_operation", threshold_seconds=0.1)
        
        # Initially not long-running
        assert not feedback.is_long_running()
        
        # After threshold
        time.sleep(0.15)
        assert feedback.is_long_running()


class TestErrorContext:
    """Test suite for ErrorContext dataclass."""
    
    def test_error_context_creation(self):
        """Test creating ErrorContext."""
        error = ValueError("Test error")
        context = ErrorContext(
            error_type=ErrorType.QUERY_PARSING_ERROR,
            original_error=error,
            user_query="Test query",
            stack_trace="Test stack trace",
            additional_info={"key": "value"}
        )
        
        assert context.error_type == ErrorType.QUERY_PARSING_ERROR
        assert context.original_error == error
        assert context.user_query == "Test query"
        assert context.stack_trace == "Test stack trace"
        assert context.additional_info == {"key": "value"}
