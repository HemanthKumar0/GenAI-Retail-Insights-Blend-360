"""
Error handling and user feedback module.

This module provides user-friendly error messages, suggestions for failed queries,
data quality warnings, and comprehensive error logging.

**Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7**
"""

import logging
import traceback
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur."""
    QUERY_PARSING_ERROR = "query_parsing_error"
    DATA_LOADING_ERROR = "data_loading_error"
    QUERY_EXECUTION_ERROR = "query_execution_error"
    VALIDATION_ERROR = "validation_error"
    API_RATE_LIMIT = "api_rate_limit"
    API_ERROR = "api_error"
    DATA_QUALITY_WARNING = "data_quality_warning"
    AMBIGUOUS_QUERY = "ambiguous_query"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ErrorContext:
    """Context information for an error."""
    error_type: ErrorType
    original_error: Exception
    user_query: Optional[str] = None
    stack_trace: Optional[str] = None
    additional_info: Dict[str, Any] = None


class ErrorHandler:
    """
    Error handler for user-friendly error messages and suggestions.
    
    This class provides:
    - User-friendly error message formatting
    - Query failure suggestions
    - Data quality warnings
    - Comprehensive error logging
    - API rate limit handling
    - Ambiguity clarification
    - Progress feedback for long operations
    
    **Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7**
    """
    
    def __init__(self):
        """Initialize the error handler."""
        self.error_log: List[Dict[str, Any]] = []
        logger.info("ErrorHandler initialized")
    
    def format_user_friendly_error(
        self,
        error: Exception,
        error_type: ErrorType,
        user_query: Optional[str] = None
    ) -> str:
        """
        Format an error into a user-friendly message without technical details.
        
        Args:
            error: The original exception
            error_type: Type of error that occurred
            user_query: The user's query that caused the error (optional)
            
        Returns:
            User-friendly error message
            
        **Validates: Requirement 12.1**
        """
        # Log the full error with stack trace
        self._log_error(error, error_type, user_query)
        
        # Create user-friendly message based on error type
        if error_type == ErrorType.QUERY_PARSING_ERROR:
            return self._format_query_parsing_error(error, user_query)
        elif error_type == ErrorType.DATA_LOADING_ERROR:
            return self._format_data_loading_error(error)
        elif error_type == ErrorType.QUERY_EXECUTION_ERROR:
            return self._format_query_execution_error(error, user_query)
        elif error_type == ErrorType.VALIDATION_ERROR:
            return self._format_validation_error(error)
        elif error_type == ErrorType.API_RATE_LIMIT:
            return self._format_rate_limit_error(error)
        elif error_type == ErrorType.API_ERROR:
            return self._format_api_error(error)
        elif error_type == ErrorType.TIMEOUT_ERROR:
            return self._format_timeout_error(error, user_query)
        elif error_type == ErrorType.AMBIGUOUS_QUERY:
            return self._format_ambiguous_query_error(error, user_query)
        else:
            return self._format_unknown_error(error)
    
    def _format_query_parsing_error(
        self,
        error: Exception,
        user_query: Optional[str]
    ) -> str:
        """Format query parsing error message."""
        message = "I had trouble understanding your question."
        
        if user_query:
            message += f"\n\nYour question: \"{user_query}\""
        
        message += "\n\nSuggestions:"
        message += "\n- Try rephrasing your question more clearly"
        message += "\n- Be specific about what data you want to see"
        message += "\n- Use simpler language and avoid complex nested questions"
        
        return message
    
    def _format_data_loading_error(self, error: Exception) -> str:
        """Format data loading error message."""
        return (
            "I encountered an issue loading your data.\n\n"
            "Please check that:\n"
            "- The file exists and is accessible\n"
            "- The file format is supported (CSV, Excel, JSON)\n"
            "- The file is not corrupted or empty\n"
            "- You have permission to read the file"
        )
    
    def _format_query_execution_error(
        self,
        error: Exception,
        user_query: Optional[str]
    ) -> str:
        """Format query execution error message."""
        message = "I encountered an error while retrieving your data."
        
        if user_query:
            message += f"\n\nYour question: \"{user_query}\""
        
        message += "\n\nThis might be because:"
        message += "\n- The data doesn't contain the information you're looking for"
        message += "\n- The column or field names don't match what you asked for"
        message += "\n- The query is too complex for the current dataset"
        
        message += "\n\nTry asking about the available data first, or rephrase your question."
        
        return message
    
    def _format_validation_error(self, error: Exception) -> str:
        """Format validation error message."""
        return (
            "The data I retrieved doesn't look quite right.\n\n"
            "This could indicate:\n"
            "- Data quality issues in your dataset\n"
            "- Unexpected data formats or values\n"
            "- A mismatch between your question and the available data\n\n"
            "Please verify your data or try a different question."
        )
    
    def _format_rate_limit_error(self, error: Exception) -> str:
        """
        Format API rate limit error message.
        
        **Validates: Requirement 12.5**
        """
        return (
            "I've reached the API rate limit for now.\n\n"
            "Please wait a moment and try again in about 30-60 seconds.\n\n"
            "If this happens frequently, consider:\n"
            "- Spacing out your questions\n"
            "- Upgrading your API plan\n"
            "- Using a different API provider"
        )
    
    def _format_api_error(self, error: Exception) -> str:
        """Format general API error message."""
        return (
            "I'm having trouble connecting to the AI service.\n\n"
            "This might be temporary. Please try again in a moment.\n\n"
            "If the problem persists, check:\n"
            "- Your internet connection\n"
            "- Your API key configuration\n"
            "- The AI service status"
        )
    
    def _format_timeout_error(
        self,
        error: Exception,
        user_query: Optional[str]
    ) -> str:
        """Format timeout error message."""
        message = "Your query took too long to process and timed out."
        
        if user_query:
            message += f"\n\nYour question: \"{user_query}\""
        
        message += "\n\nTry:"
        message += "\n- Breaking your question into smaller parts"
        message += "\n- Asking about a smaller subset of data"
        message += "\n- Simplifying your question"
        
        return message
    
    def _format_ambiguous_query_error(
        self,
        error: Exception,
        user_query: Optional[str]
    ) -> str:
        """
        Format ambiguous query error message.
        
        **Validates: Requirement 12.6**
        """
        message = "Your question is a bit ambiguous and I need clarification."
        
        if user_query:
            message += f"\n\nYour question: \"{user_query}\""
        
        message += "\n\nPlease be more specific about:"
        message += "\n- Which time period you're interested in"
        message += "\n- Which products or categories you want to analyze"
        message += "\n- What specific metrics you want to see"
        
        return message
    
    def _format_unknown_error(self, error: Exception) -> str:
        """Format unknown error message."""
        return (
            "I encountered an unexpected error.\n\n"
            "Please try again, and if the problem persists, "
            "contact support with details about what you were trying to do."
        )
    
    def suggest_alternatives(
        self,
        failed_query: str,
        error_type: ErrorType
    ) -> List[str]:
        """
        Suggest alternative approaches for failed queries.
        
        Args:
            failed_query: The query that failed
            error_type: Type of error that occurred
            
        Returns:
            List of suggested alternative queries or approaches
            
        **Validates: Requirement 12.2**
        """
        suggestions = []
        
        if error_type == ErrorType.QUERY_PARSING_ERROR:
            suggestions.extend([
                "Try: 'What were the total sales?'",
                "Try: 'Show me sales by category'",
                "Try: 'What are the top 10 products?'"
            ])
        
        elif error_type == ErrorType.QUERY_EXECUTION_ERROR:
            suggestions.extend([
                "First ask: 'What data is available?'",
                "Try: 'Show me a sample of the data'",
                "Simplify your question to focus on one metric"
            ])
        
        elif error_type == ErrorType.AMBIGUOUS_QUERY:
            suggestions.extend([
                "Add a specific time period (e.g., 'last month', 'Q4 2023')",
                "Specify which category or product you're interested in",
                "Be more specific about what you want to calculate"
            ])
        
        elif error_type == ErrorType.TIMEOUT_ERROR:
            suggestions.extend([
                "Try asking about a specific time period instead of all data",
                "Focus on one category or product at a time",
                "Ask for summary statistics instead of detailed data"
            ])
        
        return suggestions
    
    def create_data_quality_warning(
        self,
        issue_type: str,
        description: str,
        impact: str,
        affected_count: int = 0
    ) -> str:
        """
        Create a data quality warning message.
        
        Args:
            issue_type: Type of data quality issue
            description: Description of the issue
            impact: Potential impact on results
            affected_count: Number of affected rows/records
            
        Returns:
            Formatted warning message
            
        **Validates: Requirement 12.3**
        """
        warning = f"⚠️ Data Quality Warning: {issue_type}\n\n"
        warning += f"Issue: {description}\n"
        
        if affected_count > 0:
            warning += f"Affected records: {affected_count}\n"
        
        warning += f"\nPotential impact: {impact}\n"
        warning += "\nYou may want to clean your data before proceeding."
        
        # Log the warning
        logger.warning(
            f"Data quality issue: {issue_type} - {description} "
            f"(affected: {affected_count})"
        )
        
        return warning
    
    def _log_error(
        self,
        error: Exception,
        error_type: ErrorType,
        user_query: Optional[str] = None
    ) -> None:
        """
        Log error with full details for debugging.
        
        Args:
            error: The exception
            error_type: Type of error
            user_query: User's query (optional)
            
        **Validates: Requirement 12.4**
        """
        # Get stack trace
        stack_trace = traceback.format_exc()
        
        # Create error log entry
        error_entry = {
            "timestamp": time.time(),
            "error_type": error_type.value,
            "error_message": str(error),
            "error_class": error.__class__.__name__,
            "user_query": user_query,
            "stack_trace": stack_trace
        }
        
        # Add to error log
        self.error_log.append(error_entry)
        
        # Log to logger
        logger.error(
            f"Error occurred: {error_type.value}\n"
            f"Message: {str(error)}\n"
            f"Query: {user_query}\n"
            f"Stack trace:\n{stack_trace}"
        )
    
    def get_error_log(self) -> List[Dict[str, Any]]:
        """
        Get the error log.
        
        Returns:
            List of error log entries
        """
        return self.error_log
    
    def clear_error_log(self) -> None:
        """Clear the error log."""
        self.error_log.clear()
        logger.info("Error log cleared")


class ProgressFeedback:
    """
    Progress feedback for long-running operations.
    
    **Validates: Requirement 12.7**
    """
    
    def __init__(self, operation_name: str, threshold_seconds: float = 5.0):
        """
        Initialize progress feedback.
        
        Args:
            operation_name: Name of the operation
            threshold_seconds: Show progress after this many seconds
        """
        self.operation_name = operation_name
        self.threshold_seconds = threshold_seconds
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.feedback_shown = False
    
    def check_and_show_progress(self, message: Optional[str] = None) -> Optional[str]:
        """
        Check if progress feedback should be shown.
        
        Args:
            message: Custom progress message (optional)
            
        Returns:
            Progress message if threshold exceeded, None otherwise
        """
        elapsed = time.time() - self.start_time
        
        if elapsed >= self.threshold_seconds and not self.feedback_shown:
            self.feedback_shown = True
            
            if message:
                return message
            else:
                return f"Processing {self.operation_name}... This may take a moment."
        
        return None
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def is_long_running(self) -> bool:
        """Check if operation is long-running."""
        return self.get_elapsed_time() >= self.threshold_seconds
