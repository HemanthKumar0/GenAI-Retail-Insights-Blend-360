"""
Performance monitoring module.

This module provides performance monitoring capabilities including query latency
logging, LLM usage tracking, error logging, and metrics output.

**Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5**
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    """Logging levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    timestamp: datetime
    query_text: str
    operation_type: str
    latency_seconds: float
    success: bool
    error_message: Optional[str] = None
    agent_latencies: Dict[str, float] = field(default_factory=dict)


@dataclass
class LLMUsageMetrics:
    """Metrics for LLM API usage."""
    timestamp: datetime
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    cost_estimate: Optional[float] = None


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    total_queries: int
    successful_queries: int
    failed_queries: int
    average_latency: float
    min_latency: float
    max_latency: float
    total_llm_tokens: int
    total_llm_calls: int
    error_count: int


class PerformanceMonitor:
    """
    Performance monitor for tracking system metrics.
    
    This class provides:
    - Query latency logging for each agent operation
    - LLM usage tracking including token counts
    - Error logging with sufficient detail
    - Performance metrics output
    - Configurable logging levels
    
    **Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5**
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize the performance monitor.
        
        Args:
            log_level: Logging level (INFO, WARNING, ERROR, DEBUG)
            
        **Validates: Requirement 13.5**
        """
        self.log_level = LogLevel[log_level.upper()]
        
        # Metrics storage
        self.query_metrics: List[QueryMetrics] = []
        self.llm_usage_metrics: List[LLMUsageMetrics] = []
        self.error_log: List[Dict[str, Any]] = []
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self._configure_logging()
        
        self.logger.info(f"PerformanceMonitor initialized with log level: {log_level}")
    
    def _configure_logging(self) -> None:
        """Configure logging based on log level."""
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR
        }
        
        self.logger.setLevel(level_map[self.log_level])
    
    def log_query_latency(
        self,
        query_id: str,
        query_text: str,
        operation_type: str,
        latency_seconds: float,
        success: bool,
        error_message: Optional[str] = None,
        agent_latencies: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Log query latency for an operation.
        
        Args:
            query_id: Unique identifier for the query
            query_text: The query text
            operation_type: Type of operation (e.g., "parse", "execute", "validate")
            latency_seconds: Time taken in seconds
            success: Whether the operation succeeded
            error_message: Error message if failed
            agent_latencies: Breakdown of latencies by agent
            
        **Validates: Requirement 13.1**
        """
        metrics = QueryMetrics(
            query_id=query_id,
            timestamp=datetime.now(),
            query_text=query_text,
            operation_type=operation_type,
            latency_seconds=latency_seconds,
            success=success,
            error_message=error_message,
            agent_latencies=agent_latencies or {}
        )
        
        self.query_metrics.append(metrics)
        
        # Log based on level
        if self.log_level in [LogLevel.INFO, LogLevel.DEBUG]:
            self.logger.info(
                f"Query {query_id} [{operation_type}]: "
                f"{'SUCCESS' if success else 'FAILED'} in {latency_seconds:.3f}s"
            )
        
        if self.log_level == LogLevel.DEBUG and agent_latencies:
            for agent, latency in agent_latencies.items():
                self.logger.debug(f"  {agent}: {latency:.3f}s")
    
    def log_llm_usage(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_seconds: float,
        cost_estimate: Optional[float] = None
    ) -> None:
        """
        Log LLM API usage including token counts.
        
        Args:
            provider: LLM provider name (e.g., "openai", "gemini")
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            latency_seconds: API call latency
            cost_estimate: Estimated cost (optional)
            
        **Validates: Requirement 13.2**
        """
        total_tokens = prompt_tokens + completion_tokens
        
        metrics = LLMUsageMetrics(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=latency_seconds,
            cost_estimate=cost_estimate
        )
        
        self.llm_usage_metrics.append(metrics)
        
        # Log based on level
        if self.log_level in [LogLevel.INFO, LogLevel.DEBUG]:
            cost_str = f", cost: ${cost_estimate:.4f}" if cost_estimate else ""
            self.logger.info(
                f"LLM call [{provider}/{model}]: "
                f"{total_tokens} tokens ({prompt_tokens}+{completion_tokens}) "
                f"in {latency_seconds:.3f}s{cost_str}"
            )
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
        stack_trace: Optional[str] = None
    ) -> None:
        """
        Log error with sufficient detail for debugging.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Additional context information
            stack_trace: Stack trace (optional)
            
        **Validates: Requirement 13.3**
        """
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
            "stack_trace": stack_trace
        }
        
        self.error_log.append(error_entry)
        
        # Always log errors
        self.logger.error(
            f"Error [{error_type}]: {error_message}"
        )
        
        if self.log_level == LogLevel.DEBUG and stack_trace:
            self.logger.debug(f"Stack trace:\n{stack_trace}")
    
    def get_performance_stats(self) -> PerformanceStats:
        """
        Get aggregated performance statistics.
        
        Returns:
            PerformanceStats object with aggregated metrics
            
        **Validates: Requirement 13.4**
        """
        if not self.query_metrics:
            return PerformanceStats(
                total_queries=0,
                successful_queries=0,
                failed_queries=0,
                average_latency=0.0,
                min_latency=0.0,
                max_latency=0.0,
                total_llm_tokens=0,
                total_llm_calls=0,
                error_count=0
            )
        
        # Calculate query stats
        total_queries = len(self.query_metrics)
        successful_queries = sum(1 for m in self.query_metrics if m.success)
        failed_queries = total_queries - successful_queries
        
        latencies = [m.latency_seconds for m in self.query_metrics]
        average_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        # Calculate LLM stats
        total_llm_tokens = sum(m.total_tokens for m in self.llm_usage_metrics)
        total_llm_calls = len(self.llm_usage_metrics)
        
        # Error count
        error_count = len(self.error_log)
        
        return PerformanceStats(
            total_queries=total_queries,
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            average_latency=average_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            total_llm_tokens=total_llm_tokens,
            total_llm_calls=total_llm_calls,
            error_count=error_count
        )
    
    def print_performance_metrics(self) -> None:
        """
        Print performance metrics to console.
        
        **Validates: Requirement 13.4**
        """
        stats = self.get_performance_stats()
        
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        
        print("\nQuery Statistics:")
        print(f"  Total queries: {stats.total_queries}")
        print(f"  Successful: {stats.successful_queries}")
        print(f"  Failed: {stats.failed_queries}")
        
        if stats.total_queries > 0:
            success_rate = (stats.successful_queries / stats.total_queries) * 100
            print(f"  Success rate: {success_rate:.1f}%")
        
        print("\nLatency Statistics:")
        print(f"  Average: {stats.average_latency:.3f}s")
        print(f"  Min: {stats.min_latency:.3f}s")
        print(f"  Max: {stats.max_latency:.3f}s")
        
        print("\nLLM Usage:")
        print(f"  Total API calls: {stats.total_llm_calls}")
        print(f"  Total tokens: {stats.total_llm_tokens:,}")
        
        if stats.total_llm_calls > 0:
            avg_tokens = stats.total_llm_tokens / stats.total_llm_calls
            print(f"  Average tokens per call: {avg_tokens:.0f}")
        
        print("\nErrors:")
        print(f"  Total errors: {stats.error_count}")
        
        print("="*60 + "\n")
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """
        Get detailed metrics as a dictionary.
        
        Returns:
            Dictionary with all metrics
        """
        stats = self.get_performance_stats()
        
        return {
            "summary": {
                "total_queries": stats.total_queries,
                "successful_queries": stats.successful_queries,
                "failed_queries": stats.failed_queries,
                "average_latency": stats.average_latency,
                "min_latency": stats.min_latency,
                "max_latency": stats.max_latency,
                "total_llm_tokens": stats.total_llm_tokens,
                "total_llm_calls": stats.total_llm_calls,
                "error_count": stats.error_count
            },
            "query_metrics": [
                {
                    "query_id": m.query_id,
                    "timestamp": m.timestamp.isoformat(),
                    "operation_type": m.operation_type,
                    "latency_seconds": m.latency_seconds,
                    "success": m.success
                }
                for m in self.query_metrics
            ],
            "llm_usage": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "provider": m.provider,
                    "model": m.model,
                    "total_tokens": m.total_tokens,
                    "latency_seconds": m.latency_seconds
                }
                for m in self.llm_usage_metrics
            ],
            "errors": self.error_log
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.query_metrics.clear()
        self.llm_usage_metrics.clear()
        self.error_log.clear()
        self.logger.info("Performance metrics reset")
    
    def set_log_level(self, log_level: str) -> None:
        """
        Change the logging level.
        
        Args:
            log_level: New logging level (INFO, WARNING, ERROR, DEBUG)
            
        **Validates: Requirement 13.5**
        """
        self.log_level = LogLevel[log_level.upper()]
        self._configure_logging()
        self.logger.info(f"Log level changed to: {log_level}")


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """
    Get the global performance monitor instance.
    
    Returns:
        PerformanceMonitor instance
    """
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    
    return _performance_monitor


def initialize_performance_monitor(log_level: str = "INFO") -> PerformanceMonitor:
    """
    Initialize the global performance monitor.
    
    Args:
        log_level: Logging level
        
    Returns:
        PerformanceMonitor instance
    """
    global _performance_monitor
    _performance_monitor = PerformanceMonitor(log_level=log_level)
    return _performance_monitor
