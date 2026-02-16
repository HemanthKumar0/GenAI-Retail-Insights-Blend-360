"""
Unit tests for QAMode.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6**
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.qa_mode import QAMode
from src.models import Response, Message


class TestQAMode:
    """Test suite for QAMode class."""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        orchestrator = Mock()
        orchestrator.context_manager = Mock()
        orchestrator.llm_provider = Mock()
        return orchestrator
    
    @pytest.fixture
    def qa_mode(self, mock_orchestrator):
        """Create a QAMode instance with mock orchestrator."""
        return QAMode(orchestrator=mock_orchestrator)
    
    def test_initialization(self, mock_orchestrator):
        """Test QAMode initialization."""
        qa_mode = QAMode(orchestrator=mock_orchestrator)
        assert qa_mode.orchestrator == mock_orchestrator
    
    def test_answer_question_basic(self, qa_mode, mock_orchestrator):
        """
        Test answering a basic question.
        
        **Validates: Requirement 7.1**
        """
        # Setup
        question = "What were the total sales last month?"
        expected_response = Response(
            answer="Total sales last month were $150,000.",
            metadata={"execution_time": 1.5}
        )
        mock_orchestrator.process_query.return_value = expected_response
        
        # Execute
        response = qa_mode.answer_question(question)
        
        # Verify
        assert response == expected_response
        mock_orchestrator.process_query.assert_called_once_with(
            user_query=question,
            mode="qa"
        )
    
    def test_answer_question_empty(self, qa_mode):
        """Test that empty questions raise ValueError."""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            qa_mode.answer_question("")
        
        with pytest.raises(ValueError, match="Question cannot be empty"):
            qa_mode.answer_question("   ")
    
    def test_answer_question_with_data(self, qa_mode, mock_orchestrator):
        """
        Test answering question with data-grounded response.
        
        **Validates: Requirement 7.2**
        """
        import pandas as pd
        
        # Setup
        question = "Show me top 5 products by sales"
        data = pd.DataFrame({
            "product": ["A", "B", "C", "D", "E"],
            "sales": [1000, 900, 800, 700, 600]
        })
        expected_response = Response(
            answer="Here are the top 5 products by sales: Product A ($1000), Product B ($900)...",
            data=data,
            metadata={"execution_time": 2.0, "row_count": 5}
        )
        mock_orchestrator.process_query.return_value = expected_response
        
        # Execute
        response = qa_mode.answer_question(question)
        
        # Verify
        assert response.answer == expected_response.answer
        assert response.data is not None
        assert len(response.data) == 5
    
    def test_follow_up_question_handling(self, qa_mode, mock_orchestrator):
        """
        Test handling follow-up questions with context.
        
        **Validates: Requirements 7.3, 7.5**
        """
        # Setup - first question
        first_question = "What were sales in Q4?"
        first_response = Response(
            answer="Q4 sales were $500,000.",
            metadata={"execution_time": 1.0}
        )
        mock_orchestrator.process_query.return_value = first_response
        
        # Execute first question
        response1 = qa_mode.answer_question(first_question)
        assert response1 == first_response
        
        # Setup - follow-up question with pronoun
        follow_up = "How about last year?"
        follow_up_response = Response(
            answer="Last year's Q4 sales were $450,000.",
            metadata={"execution_time": 1.2}
        )
        mock_orchestrator.process_query.return_value = follow_up_response
        
        # Execute follow-up
        response2 = qa_mode.answer_question(follow_up)
        
        # Verify both questions were processed
        assert mock_orchestrator.process_query.call_count == 2
        assert response2 == follow_up_response
    
    def test_clarification_detection(self, qa_mode):
        """
        Test detection of clarification requests.
        
        **Validates: Requirement 7.6**
        """
        clarification_questions = [
            "Can you explain that?",
            "What do you mean by that?",
            "Tell me more about this",
            "Can you clarify?",
            "Explain in more detail",
            "How did you calculate that?",
            "Why did you say that?"
        ]
        
        for question in clarification_questions:
            assert qa_mode._is_clarification_request(question)
        
        # Non-clarification questions
        regular_questions = [
            "What were the sales?",
            "Show me the data",
            "Calculate the total"
        ]
        
        for question in regular_questions:
            assert not qa_mode._is_clarification_request(question)
    
    def test_handle_clarification_with_history(self, qa_mode, mock_orchestrator):
        """
        Test handling clarification with conversation history.
        
        **Validates: Requirement 7.6**
        """
        # Setup conversation history
        history = [
            Message(
                role="user",
                content="What were total sales?",
                timestamp=datetime.now()
            ),
            Message(
                role="assistant",
                content="Total sales were $1M.",
                timestamp=datetime.now()
            )
        ]
        mock_orchestrator.get_conversation_history.return_value = history
        
        # Setup LLM response for clarification
        mock_llm_response = Mock()
        mock_llm_response.content = "Total sales of $1M includes all products across all regions for the entire year."
        mock_llm_response.tokens_used = 50
        mock_orchestrator.llm_provider.generate_with_cache.return_value = mock_llm_response
        
        # Execute
        response = qa_mode._handle_clarification("Can you explain that?")
        
        # Verify
        assert "Total sales of $1M" in response.answer
        assert response.metadata.get("clarification") is True
        mock_orchestrator.context_manager.add_message.assert_called_once()
    
    def test_handle_clarification_no_history(self, qa_mode, mock_orchestrator):
        """Test clarification request with no conversation history."""
        # Setup empty history
        mock_orchestrator.get_conversation_history.return_value = []
        
        # Execute
        response = qa_mode._handle_clarification("Can you explain that?")
        
        # Verify
        assert "don't have any previous responses" in response.answer
        assert response.metadata.get("clarification_request") is True
    
    def test_handle_clarification_no_assistant_message(self, qa_mode, mock_orchestrator):
        """Test clarification when no assistant message exists."""
        # Setup history with only user messages
        history = [
            Message(
                role="user",
                content="What were sales?",
                timestamp=datetime.now()
            )
        ]
        mock_orchestrator.get_conversation_history.return_value = history
        
        # Execute
        response = qa_mode._handle_clarification("Explain that")
        
        # Verify
        assert "don't have" in response.answer.lower() and "previous" in response.answer.lower()
    
    def test_get_conversation_history(self, qa_mode, mock_orchestrator):
        """
        Test retrieving conversation history.
        
        **Validates: Requirement 7.7**
        """
        # Setup
        history = [
            Message(role="user", content="Question 1", timestamp=datetime.now()),
            Message(role="assistant", content="Answer 1", timestamp=datetime.now()),
            Message(role="user", content="Question 2", timestamp=datetime.now()),
            Message(role="assistant", content="Answer 2", timestamp=datetime.now())
        ]
        mock_orchestrator.get_conversation_history.return_value = history
        
        # Execute
        result = qa_mode.get_conversation_history()
        
        # Verify
        assert result == history
        assert len(result) == 4
        mock_orchestrator.get_conversation_history.assert_called_once()
    
    def test_reset_conversation(self, qa_mode, mock_orchestrator):
        """Test resetting conversation history."""
        # Execute
        qa_mode.reset_conversation()
        
        # Verify
        mock_orchestrator.reset_context.assert_called_once()
    
    def test_answer_question_error_handling(self, qa_mode, mock_orchestrator):
        """Test error handling when query processing fails."""
        # Setup
        question = "What were sales?"
        mock_orchestrator.process_query.side_effect = Exception("Database error")
        
        # Execute
        response = qa_mode.answer_question(question)
        
        # Verify
        assert "error" in response.answer.lower()
        assert response.metadata.get("error") == "Database error"
    
    def test_clarification_error_handling(self, qa_mode, mock_orchestrator):
        """Test error handling in clarification generation."""
        # Setup
        history = [
            Message(role="user", content="Question", timestamp=datetime.now()),
            Message(role="assistant", content="Answer", timestamp=datetime.now())
        ]
        mock_orchestrator.get_conversation_history.return_value = history
        mock_orchestrator.llm_provider.generate_with_cache.side_effect = Exception("LLM error")
        
        # Execute
        response = qa_mode._handle_clarification("Explain that")
        
        # Verify
        assert "error" in response.answer.lower()
        assert response.metadata.get("error") == "LLM error"
    
    def test_multiple_operations_chaining(self, qa_mode, mock_orchestrator):
        """
        Test that complex questions requiring multiple operations are handled.
        
        **Validates: Requirement 7.4**
        """
        # Setup
        complex_question = "What were sales last month and how does that compare to the previous month?"
        response_data = Response(
            answer="Last month sales were $100K, which is 10% higher than the previous month's $90K.",
            metadata={
                "execution_time": 3.5,
                "operations_chained": 2
            }
        )
        mock_orchestrator.process_query.return_value = response_data
        
        # Execute
        response = qa_mode.answer_question(complex_question)
        
        # Verify
        assert response == response_data
        mock_orchestrator.process_query.assert_called_once_with(
            user_query=complex_question,
            mode="qa"
        )
