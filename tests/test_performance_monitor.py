"""
Unit tests for PerformanceMonitor.

**Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5**
"""

import pytest
from datetime import datetime

from src.performance_monitor import (
    PerformanceMonitor, LogLevel, QueryMetrics, LLMUsageMetrics,
    PerformanceStats, get_performance_monitor, initialize_performance_monitor
)


class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor class."""
    
    @pytest.fixture
    def monitor(self):
        """Create a PerformanceMonitor instance."""
        return PerformanceMonitor(log_level="INFO")
    
    def test_initialization(self, monitor):
        """Test PerformanceMonitor initialization."""
        assert monitor.log_level == LogLevel.INFO
        assert len(monitor.query_metrics) == 0
        assert len(monitor.llm_usage_metrics) == 0
        assert len(monitor.error_log) == 0
    
    def test_log_query_latency(self, monitor):
        """
        Test logging query latency.
        
        **Validates: Requirement 13.1**
        """
        monitor.log_query_latency(
            query_id="q1",
            query_text="What were sales?",
            operation_type="execute",
            latency_seconds=1.5,
            success=True
        )
        
        assert len(monitor.query_metrics) == 1
        metric = monitor.query_metrics[0]
        assert metric.query_id == "q1"
        assert metric.query_text == "What were sales?"
        assert metric.operation_type == "execute"
        assert metric.latency_seconds == 1.5
        assert metric.success is True
    
    def test_log_query_latency_with_agent_breakdown(self, monitor):
        """Test logging query latency with agent breakdown."""
        agent_latencies = {
            "QueryAgent": 0.5,
            "ExtractionAgent": 0.8,
            "ValidationAgent": 0.2
        }
        
        monitor.log_query_latency(
            query_id="q2",
            query_text="Show data",
            operation_type="full_pipeline",
            latency_seconds=1.5,
            success=True,
            agent_latencies=agent_latencies
        )
        
        metric = monitor.query_metrics[0]
        assert metric.agent_latencies == agent_latencies
    
    def test_log_llm_usage(self, monitor):
        """
        Test logging LLM usage.
        
        **Validates: Requirement 13.2**
        """
        monitor.log_llm_usage(
            provider="openai",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            latency_seconds=2.0,
            cost_estimate=0.015
        )
        
        assert len(monitor.llm_usage_metrics) == 1
        metric = monitor.llm_usage_metrics[0]
        assert metric.provider == "openai"
        assert metric.model == "gpt-4"
        assert metric.prompt_tokens == 100
        assert metric.completion_tokens == 50
        assert metric.total_tokens == 150
        assert metric.latency_seconds == 2.0
        assert metric.cost_estimate == 0.015
    
    def test_log_error(self, monitor):
        """
        Test logging errors.
        
        **Validates: Requirement 13.3**
        """
        monitor.log_error(
            error_type="QueryExecutionError",
            error_message="Failed to execute query",
            context={"query": "test query"},
            stack_trace="Traceback..."
        )
        
        assert len(monitor.error_log) == 1
        error = monitor.error_log[0]
        assert error["error_type"] == "QueryExecutionError"
        assert error["error_message"] == "Failed to execute query"
        assert error["context"]["query"] == "test query"
        assert error["stack_trace"] == "Traceback..."
    
    def test_get_performance_stats_empty(self, monitor):
        """Test getting stats with no metrics."""
        stats = monitor.get_performance_stats()
        
        assert stats.total_queries == 0
        assert stats.successful_queries == 0
        assert stats.failed_queries == 0
        assert stats.average_latency == 0.0
    
    def test_get_performance_stats(self, monitor):
        """
        Test getting performance statistics.
        
        **Validates: Requirement 13.4**
        """
        # Add some query metrics
        monitor.log_query_latency("q1", "query1", "execute", 1.0, True)
        monitor.log_query_latency("q2", "query2", "execute", 2.0, True)
        monitor.log_query_latency("q3", "query3", "execute", 1.5, False)
        
        # Add LLM metrics
        monitor.log_llm_usage("openai", "gpt-4", 100, 50, 1.0)
        monitor.log_llm_usage("openai", "gpt-4", 200, 100, 1.5)
        
        # Add errors
        monitor.log_error("TestError", "Test error message")
        
        stats = monitor.get_performance_stats()
        
        assert stats.total_queries == 3
        assert stats.successful_queries == 2
        assert stats.failed_queries == 1
        assert stats.average_latency == 1.5
        assert stats.min_latency == 1.0
        assert stats.max_latency == 2.0
        assert stats.total_llm_tokens == 450  # (100+50) + (200+100)
        assert stats.total_llm_calls == 2
        assert stats.error_count == 1
    
    def test_print_performance_metrics(self, monitor, capsys):
        """Test printing performance metrics to console."""
        # Add some metrics
        monitor.log_query_latency("q1", "query1", "execute", 1.0, True)
        monitor.log_llm_usage("openai", "gpt-4", 100, 50, 1.0)
        
        # Print metrics
        monitor.print_performance_metrics()
        
        # Capture output
        captured = capsys.readouterr()
        
        assert "PERFORMANCE METRICS" in captured.out
        assert "Total queries: 1" in captured.out
        assert "Total API calls: 1" in captured.out
        assert "Total tokens: 150" in captured.out
    
    def test_get_detailed_metrics(self, monitor):
        """Test getting detailed metrics as dictionary."""
        monitor.log_query_latency("q1", "query1", "execute", 1.0, True)
        monitor.log_llm_usage("openai", "gpt-4", 100, 50, 1.0)
        
        metrics = monitor.get_detailed_metrics()
        
        assert "summary" in metrics
        assert "query_metrics" in metrics
        assert "llm_usage" in metrics
        assert "errors" in metrics
        
        assert metrics["summary"]["total_queries"] == 1
        assert len(metrics["query_metrics"]) == 1
        assert len(metrics["llm_usage"]) == 1
    
    def test_reset_metrics(self, monitor):
        """Test resetting metrics."""
        # Add some metrics
        monitor.log_query_latency("q1", "query1", "execute", 1.0, True)
        monitor.log_llm_usage("openai", "gpt-4", 100, 50, 1.0)
        monitor.log_error("TestError", "Test")
        
        assert len(monitor.query_metrics) > 0
        assert len(monitor.llm_usage_metrics) > 0
        assert len(monitor.error_log) > 0
        
        # Reset
        monitor.reset_metrics()
        
        assert len(monitor.query_metrics) == 0
        assert len(monitor.llm_usage_metrics) == 0
        assert len(monitor.error_log) == 0
    
    def test_set_log_level(self, monitor):
        """
        Test changing log level.
        
        **Validates: Requirement 13.5**
        """
        assert monitor.log_level == LogLevel.INFO
        
        monitor.set_log_level("DEBUG")
        assert monitor.log_level == LogLevel.DEBUG
        
        monitor.set_log_level("ERROR")
        assert monitor.log_level == LogLevel.ERROR
    
    def test_different_log_levels(self):
        """Test initialization with different log levels."""
        for level in ["INFO", "WARNING", "ERROR", "DEBUG"]:
            monitor = PerformanceMonitor(log_level=level)
            assert monitor.log_level == LogLevel[level]
    
    def test_global_monitor_instance(self):
        """Test global monitor instance functions."""
        # Get global instance
        monitor1 = get_performance_monitor()
        assert monitor1 is not None
        
        # Should return same instance
        monitor2 = get_performance_monitor()
        assert monitor1 is monitor2
        
        # Initialize new instance
        monitor3 = initialize_performance_monitor(log_level="DEBUG")
        assert monitor3.log_level == LogLevel.DEBUG
        
        # Get should return new instance
        monitor4 = get_performance_monitor()
        assert monitor3 is monitor4


class TestQueryMetrics:
    """Test suite for QueryMetrics dataclass."""
    
    def test_query_metrics_creation(self):
        """Test creating QueryMetrics."""
        metrics = QueryMetrics(
            query_id="q1",
            timestamp=datetime.now(),
            query_text="test query",
            operation_type="execute",
            latency_seconds=1.5,
            success=True,
            error_message=None,
            agent_latencies={"agent1": 0.5}
        )
        
        assert metrics.query_id == "q1"
        assert metrics.query_text == "test query"
        assert metrics.operation_type == "execute"
        assert metrics.latency_seconds == 1.5
        assert metrics.success is True
        assert metrics.agent_latencies == {"agent1": 0.5}


class TestLLMUsageMetrics:
    """Test suite for LLMUsageMetrics dataclass."""
    
    def test_llm_usage_metrics_creation(self):
        """Test creating LLMUsageMetrics."""
        metrics = LLMUsageMetrics(
            timestamp=datetime.now(),
            provider="openai",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_seconds=2.0,
            cost_estimate=0.015
        )
        
        assert metrics.provider == "openai"
        assert metrics.model == "gpt-4"
        assert metrics.prompt_tokens == 100
        assert metrics.completion_tokens == 50
        assert metrics.total_tokens == 150
        assert metrics.latency_seconds == 2.0
        assert metrics.cost_estimate == 0.015


class TestPerformanceStats:
    """Test suite for PerformanceStats dataclass."""
    
    def test_performance_stats_creation(self):
        """Test creating PerformanceStats."""
        stats = PerformanceStats(
            total_queries=10,
            successful_queries=8,
            failed_queries=2,
            average_latency=1.5,
            min_latency=0.5,
            max_latency=3.0,
            total_llm_tokens=1000,
            total_llm_calls=5,
            error_count=2
        )
        
        assert stats.total_queries == 10
        assert stats.successful_queries == 8
        assert stats.failed_queries == 2
        assert stats.average_latency == 1.5
        assert stats.total_llm_tokens == 1000
