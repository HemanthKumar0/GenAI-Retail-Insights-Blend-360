"""
Unit tests for Orchestrator module.

Tests the orchestrator's ability to coordinate agents, handle retries,
log communications, and format responses.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import pandas as pd

from src.orchestrator import Orchestrator
from src.models import (
    StructuredQuery, QueryResult, ValidationResult, Anomaly,
    Response, DataSchema, TableSchema, ColumnInfo, Message
)
from src.llm_provider import LLMResponse


@pytest.fixture
def mock_query_agent():
    """Create mock QueryAgent."""
    agent = Mock()
    agent.parse_query = Mock(return_value=StructuredQuery(
        operation_type="sql",
        operation="SELECT * FROM sales",
        explanation="Get all sales data"
    ))
    return agent


@pytest.fixture
def mock_extraction_agent():
    """Create mock ExtractionAgent."""
    agent = Mock()
    
    # Mock data store
    agent.data_store = Mock()
    agent.data_store.list_tables = Mock(return_value=["sales"])
    agent.data_store.get_table_schema = Mock(return_value={
        "columns": {
            "product": {"dtype": "object", "nullable": False, "unique_count": 10, "sample_values": ["A", "B"]},
            "sales": {"dtype": "float64", "nullable": False, "unique_count": 100, "sample_values": [100.0, 200.0]}
        },
        "row_count": 100,
        "sample_data": pd.DataFrame({"product": ["A", "B"], "sales": [100.0, 200.0]})
    })
    
    # Mock execute_query
    agent.execute_query = Mock(return_value=QueryResult(
        data=pd.DataFrame({"product": ["A", "B"], "sales": [100.0, 200.0]}),
        row_count=2,
        execution_time=0.5,
        query=StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM sales",
            explanation="Get all sales data"
        ),
        cached=False
    ))
    
    return agent


@pytest.fixture
def mock_validation_agent():
    """Create mock ValidationAgent."""
    agent = Mock()
    agent.validate_results = Mock(return_value=ValidationResult(
        passed=True,
        issues=[],
        anomalies=[],
        confidence=1.0
    ))
    return agent


@pytest.fixture
def mock_llm_provider():
    """Create mock LLM provider."""
    provider = Mock()
    provider.generate_with_cache = Mock(return_value=LLMResponse(
        content="The total sales are $300.00 across 2 products.",
        tokens_used=50,
        model="test-model"
    ))
    provider.count_tokens = Mock(return_value=10)
    return provider


@pytest.fixture
def orchestrator(mock_query_agent, mock_extraction_agent, mock_validation_agent, mock_llm_provider):
    """Create Orchestrator instance with mocked agents."""
    return Orchestrator(
        query_agent=mock_query_agent,
        extraction_agent=mock_extraction_agent,
        validation_agent=mock_validation_agent,
        llm_provider=mock_llm_provider,
        max_retries=3
    )


class TestOrchestratorInitialization:
    """Test Orchestrator initialization."""
    
    def test_initialization(self, orchestrator):
        """Test that orchestrator initializes correctly."""
        assert orchestrator.max_retries == 3
        assert orchestrator.query_agent is not None
        assert orchestrator.extraction_agent is not None
        assert orchestrator.validation_agent is not None
        assert orchestrator.llm_provider is not None
        assert orchestrator.context_manager is not None
        assert orchestrator.communication_log == []


class TestQueryProcessing:
    """Test query processing flow."""
    
    def test_process_query_success(self, orchestrator):
        """
        Test successful query processing through all agents.
        
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
        """
        response = orchestrator.process_query("What are the total sales?", mode="qa")
        
        # Verify response
        assert isinstance(response, Response)
        assert response.answer is not None
        assert response.data is not None
        assert "execution_time" in response.metadata
        
        # Verify agents were called
        orchestrator.query_agent.parse_query.assert_called_once()
        orchestrator.extraction_agent.execute_query.assert_called_once()
        orchestrator.validation_agent.validate_results.assert_called_once()
        orchestrator.llm_provider.generate_with_cache.assert_called()
    
    def test_process_query_invalid_mode(self, orchestrator):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            orchestrator.process_query("test query", mode="invalid")
    
    def test_query_routing_to_agents(self, orchestrator):
        """
        Test that queries are routed correctly to agents.
        
        **Validates: Requirement 3.1**
        """
        orchestrator.process_query("Show me sales data", mode="qa")
        
        # Verify QueryAgent was called with correct parameters
        call_args = orchestrator.query_agent.parse_query.call_args
        assert call_args[1]["query"] == "Show me sales data"
        assert "schema" in call_args[1]
        assert "context" in call_args[1]


class TestRetryLogic:
    """Test retry logic for validation failures."""
    
    def test_retry_on_validation_failure(self, orchestrator, mock_validation_agent):
        """
        Test that orchestrator retries on validation failure.
        
        **Validates: Requirements 3.5, 3.6**
        """
        # First two attempts fail, third succeeds
        mock_validation_agent.validate_results.side_effect = [
            ValidationResult(passed=False, issues=["Issue 1"], confidence=0.5),
            ValidationResult(passed=False, issues=["Issue 2"], confidence=0.5),
            ValidationResult(passed=True, issues=[], confidence=1.0)
        ]
        
        response = orchestrator.process_query("test query", mode="qa")
        
        # Verify retries occurred
        assert mock_validation_agent.validate_results.call_count == 3
        assert isinstance(response, Response)
    
    def test_max_retries_exceeded(self, orchestrator, mock_validation_agent):
        """
        Test that orchestrator returns error after max retries.
        
        **Validates: Requirement 3.6**
        """
        # All attempts fail
        mock_validation_agent.validate_results.return_value = ValidationResult(
            passed=False,
            issues=["Persistent issue"],
            confidence=0.3
        )
        
        response = orchestrator.process_query("test query", mode="qa")
        
        # Verify max retries were attempted
        assert mock_validation_agent.validate_results.call_count == 3
        
        # Verify error response
        assert isinstance(response, Response)
        assert "validation_failed" in response.metadata
        assert response.metadata["validation_failed"] is True
    
    def test_reformulation_on_failure(self, orchestrator, mock_validation_agent, mock_llm_provider):
        """
        Test that query is reformulated on validation failure.
        
        **Validates: Requirement 3.5**
        """
        # First attempt fails, second succeeds
        mock_validation_agent.validate_results.side_effect = [
            ValidationResult(passed=False, issues=["Data type mismatch"], confidence=0.5),
            ValidationResult(passed=True, issues=[], confidence=1.0)
        ]
        
        # Mock reformulation
        mock_llm_provider.generate_with_cache.side_effect = [
            LLMResponse(content="reformulated query", tokens_used=20, model="test"),
            LLMResponse(content="Final answer", tokens_used=50, model="test")
        ]
        
        response = orchestrator.process_query("test query", mode="qa")
        
        # Verify reformulation occurred
        assert mock_llm_provider.generate_with_cache.call_count >= 2
        assert isinstance(response, Response)


class TestCommunicationLogging:
    """Test inter-agent communication logging."""
    
    def test_communication_logging(self, orchestrator):
        """
        Test that all inter-agent communications are logged.
        
        **Validates: Requirement 3.7**
        """
        orchestrator.process_query("test query", mode="qa")
        
        # Verify communication log has entries
        assert len(orchestrator.communication_log) > 0
        
        # Verify log structure
        for entry in orchestrator.communication_log:
            assert "timestamp" in entry
            assert "sender" in entry
            assert "receiver" in entry
            assert "message" in entry
    
    def test_log_contains_agent_interactions(self, orchestrator):
        """
        Test that log contains expected agent interactions.
        
        **Validates: Requirement 3.7**
        """
        orchestrator.process_query("test query", mode="qa")
        
        log = orchestrator.communication_log
        
        # Check for expected interactions
        senders = [entry["sender"] for entry in log]
        receivers = [entry["receiver"] for entry in log]
        
        assert "Orchestrator" in senders
        assert "QueryAgent" in receivers
        assert "ExtractionAgent" in receivers
        assert "ValidationAgent" in receivers
    
    def test_get_communication_log(self, orchestrator):
        """Test retrieving communication log."""
        orchestrator.process_query("test query", mode="qa")
        
        log = orchestrator.get_communication_log()
        
        assert isinstance(log, list)
        assert len(log) > 0
    
    def test_clear_communication_log(self, orchestrator):
        """Test clearing communication log."""
        orchestrator.process_query("test query", mode="qa")
        
        assert len(orchestrator.communication_log) > 0
        
        orchestrator.clear_communication_log()
        
        assert len(orchestrator.communication_log) == 0


class TestResponseFormatting:
    """Test response formatting."""
    
    def test_response_formatting(self, orchestrator, mock_llm_provider):
        """
        Test that responses are formatted using LLM.
        
        **Validates: Requirement 7.2**
        """
        response = orchestrator.process_query("test query", mode="qa")
        
        # Verify LLM was called for formatting
        mock_llm_provider.generate_with_cache.assert_called()
        
        # Verify response structure
        assert isinstance(response, Response)
        assert response.answer is not None
        assert "tokens_used" in response.metadata
    
    def test_response_includes_data(self, orchestrator):
        """Test that response includes query result data."""
        response = orchestrator.process_query("test query", mode="qa")
        
        assert response.data is not None
        assert isinstance(response.data, pd.DataFrame)
    
    def test_response_includes_metadata(self, orchestrator):
        """Test that response includes execution metadata."""
        response = orchestrator.process_query("test query", mode="qa")
        
        metadata = response.metadata
        
        assert "execution_time" in metadata
        assert "tokens_used" in metadata
        assert "agents_involved" in metadata
        assert "row_count" in metadata
        assert "validation_confidence" in metadata
    
    def test_response_includes_warnings(self, orchestrator, mock_validation_agent):
        """Test that response includes validation warnings."""
        # Add anomalies to validation result
        mock_validation_agent.validate_results.return_value = ValidationResult(
            passed=True,
            issues=[],
            anomalies=[
                Anomaly(
                    type="negative_value",
                    description="Found negative sales",
                    severity="warning",
                    affected_rows=[1, 2]
                )
            ],
            confidence=0.9
        )
        
        response = orchestrator.process_query("test query", mode="qa")
        
        assert "warnings" in response.metadata
        assert len(response.metadata["warnings"]) > 0


class TestConversationContext:
    """Test conversation context management."""
    
    def test_context_initialization(self, orchestrator):
        """Test that context manager is initialized."""
        assert orchestrator.context_manager is not None
    
    def test_messages_added_to_context(self, orchestrator):
        """
        Test that messages are added to conversation context.
        
        **Validates: Requirement 11.1, 11.2**
        """
        orchestrator.process_query("first query", mode="qa")
        
        history = orchestrator.get_conversation_history()
        
        # Should have user and assistant messages
        assert len(history) >= 2
        assert any(msg.role == "user" for msg in history)
        assert any(msg.role == "assistant" for msg in history)
    
    def test_context_passed_to_query_agent(self, orchestrator, mock_query_agent):
        """Test that context is passed to QueryAgent."""
        # First query
        orchestrator.process_query("first query", mode="qa")
        
        # Second query should include context
        orchestrator.process_query("second query", mode="qa")
        
        # Verify context was passed
        call_args = mock_query_agent.parse_query.call_args
        assert "context" in call_args[1]
        assert call_args[1]["context"] != ""
    
    def test_reset_context(self, orchestrator):
        """Test resetting conversation context."""
        orchestrator.process_query("test query", mode="qa")
        
        assert len(orchestrator.get_conversation_history()) > 0
        
        orchestrator.reset_context()
        
        assert len(orchestrator.get_conversation_history()) == 0
    
    def test_get_conversation_history(self, orchestrator):
        """Test retrieving conversation history."""
        orchestrator.process_query("test query", mode="qa")
        
        history = orchestrator.get_conversation_history()
        
        assert isinstance(history, list)
        assert all(isinstance(msg, Message) for msg in history)


class TestErrorHandling:
    """Test error handling in orchestrator."""
    
    def test_query_agent_error(self, orchestrator, mock_query_agent):
        """Test handling of QueryAgent errors."""
        mock_query_agent.parse_query.side_effect = Exception("Parse error")
        
        response = orchestrator.process_query("test query", mode="qa")
        
        # Should return error response
        assert isinstance(response, Response)
        assert "error" in response.metadata
    
    def test_extraction_agent_error(self, orchestrator, mock_extraction_agent):
        """Test handling of ExtractionAgent errors."""
        mock_extraction_agent.execute_query.side_effect = Exception("Execution error")
        
        response = orchestrator.process_query("test query", mode="qa")
        
        # Should return error response
        assert isinstance(response, Response)
        assert "error" in response.metadata
    
    def test_validation_agent_error(self, orchestrator, mock_validation_agent):
        """Test handling of ValidationAgent errors."""
        mock_validation_agent.validate_results.side_effect = Exception("Validation error")
        
        response = orchestrator.process_query("test query", mode="qa")
        
        # Should return error response
        assert isinstance(response, Response)
        assert "error" in response.metadata
    
    def test_llm_formatting_error_fallback(self, orchestrator, mock_llm_provider):
        """Test fallback when LLM formatting fails."""
        mock_llm_provider.generate_with_cache.side_effect = Exception("LLM error")
        
        response = orchestrator.process_query("test query", mode="qa")
        
        # Should still return a response (fallback)
        assert isinstance(response, Response)
        assert response.answer is not None


class TestDataSchemaRetrieval:
    """Test data schema retrieval."""
    
    def test_get_data_schema(self, orchestrator):
        """Test retrieving data schema from extraction agent."""
        schema = orchestrator._get_data_schema()
        
        assert isinstance(schema, DataSchema)
        assert "sales" in schema.tables
        assert isinstance(schema.tables["sales"], TableSchema)
    
    def test_schema_includes_columns(self, orchestrator):
        """Test that schema includes column information."""
        schema = orchestrator._get_data_schema()
        
        table_schema = schema.tables["sales"]
        
        assert len(table_schema.columns) > 0
        assert all(isinstance(col, ColumnInfo) for col in table_schema.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
