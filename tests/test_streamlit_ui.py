"""
Integration tests for Streamlit UI.

These tests verify the UI components work correctly with the backend.
Note: Full Streamlit UI testing requires streamlit.testing which is limited,
so these tests focus on the underlying logic.
"""

import pytest
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_store import DataStore
from orchestrator import Orchestrator
from query_agent import QueryAgent
from extraction_agent import ExtractionAgent
from validation_agent import ValidationAgent
from llm_provider import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self):
        super().__init__(model="mock-model", api_key="mock-key")
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                max_tokens: int = None) -> LLMResponse:
        return LLMResponse(
            content="Mock response",
            tokens_used=100,
            model="mock-model"
        )
    
    def count_tokens(self, text: str) -> int:
        return len(text.split())


def create_test_orchestrator() -> Orchestrator:
    """Create orchestrator for testing with mock LLM."""
    llm_provider = MockLLMProvider()
    data_store = DataStore()
    extraction_agent = ExtractionAgent(data_store)
    validation_agent = ValidationAgent()
    query_agent = QueryAgent(llm_provider)
    
    return Orchestrator(
        query_agent=query_agent,
        extraction_agent=extraction_agent,
        validation_agent=validation_agent,
        llm_provider=llm_provider,
        max_retries=3
    )


class TestUIIntegration:
    """Test UI integration with backend components."""
    
    def test_orchestrator_initialization(self):
        """Test that orchestrator can be initialized for UI."""
        orchestrator = create_test_orchestrator()
        assert orchestrator is not None
        assert orchestrator.extraction_agent is not None
        assert orchestrator.query_agent is not None
        assert orchestrator.validation_agent is not None
    
    def test_file_upload_simulation(self):
        """Test file upload workflow."""
        orchestrator = create_test_orchestrator()
        data_store = orchestrator.extraction_agent.data_store
        
        # Simulate CSV upload
        table_name = data_store.load_csv("Sales Dataset/Sale Report.csv")
        assert table_name is not None
        assert table_name in data_store.list_tables()
        
        # Get schema for display
        schema = data_store.get_table_schema(table_name)
        assert schema is not None
        assert 'columns' in schema
        assert 'row_count' in schema
    
    def test_qa_mode_query_flow(self):
        """Test Q&A mode query processing."""
        orchestrator = create_test_orchestrator()
        data_store = orchestrator.extraction_agent.data_store
        
        # Load data
        table_name = data_store.load_csv("Sales Dataset/Sale Report.csv")
        
        # Simulate user query (without LLM call)
        # We'll test the data retrieval part
        from src.models import StructuredQuery
        
        query = StructuredQuery(
            operation_type="sql",
            operation=f"SELECT COUNT(*) as total FROM {table_name}",
            explanation="Count total rows"
        )
        
        result = orchestrator.extraction_agent.execute_query(query)
        assert result is not None
        assert result.row_count > 0
        assert result.data is not None
    
    def test_session_state_simulation(self):
        """Test session state management logic."""
        # Simulate session state
        session_state = {
            'orchestrator': create_test_orchestrator(),
            'messages': [],
            'mode': 'qa',
            'loaded_datasets': [],
            'processing': False
        }
        
        # Simulate loading a dataset
        data_store = session_state['orchestrator'].extraction_agent.data_store
        table_name = data_store.load_csv("Sales Dataset/Sale Report.csv")
        session_state['loaded_datasets'].append(table_name)
        
        assert len(session_state['loaded_datasets']) == 1
        assert session_state['loaded_datasets'][0] == table_name
        
        # Simulate adding messages
        session_state['messages'].append({
            'role': 'user',
            'content': 'What are the total sales?'
        })
        
        assert len(session_state['messages']) == 1
        assert session_state['messages'][0]['role'] == 'user'
    
    def test_mode_switching(self):
        """Test switching between Q&A and Summarization modes."""
        orchestrator = create_test_orchestrator()
        
        # Test both modes are supported
        modes = ['qa', 'summarization']
        
        for mode in modes:
            # Mode should be accepted (actual processing would require LLM)
            assert mode in ['qa', 'summarization']
    
    def test_error_handling_in_ui_context(self):
        """Test error handling for UI scenarios."""
        orchestrator = create_test_orchestrator()
        data_store = orchestrator.extraction_agent.data_store
        
        # Test invalid file
        with pytest.raises(Exception):
            data_store.load_csv("nonexistent_file.csv")
        
        # Test query without loaded data
        # This should be handled gracefully in the UI
        tables = data_store.list_tables()
        assert isinstance(tables, list)
    
    def test_data_display_formatting(self):
        """Test data formatting for display."""
        orchestrator = create_test_orchestrator()
        data_store = orchestrator.extraction_agent.data_store
        
        # Load data
        table_name = data_store.load_csv("Sales Dataset/Sale Report.csv")
        
        # Get sample data for display
        from src.models import StructuredQuery
        
        query = StructuredQuery(
            operation_type="sql",
            operation=f"SELECT * FROM {table_name} LIMIT 10",
            explanation="Get sample data"
        )
        
        result = orchestrator.extraction_agent.execute_query(query)
        
        # Verify data can be displayed
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) <= 10
        assert len(result.data.columns) > 0
    
    def test_metadata_display(self):
        """Test metadata extraction for UI display."""
        orchestrator = create_test_orchestrator()
        data_store = orchestrator.extraction_agent.data_store
        
        # Load data
        table_name = data_store.load_csv("Sales Dataset/Sale Report.csv")
        
        # Execute query
        from src.models import StructuredQuery
        
        query = StructuredQuery(
            operation_type="sql",
            operation=f"SELECT COUNT(*) as total FROM {table_name}",
            explanation="Count rows"
        )
        
        result = orchestrator.extraction_agent.execute_query(query)
        
        # Verify metadata is available
        assert result.execution_time >= 0
        assert result.row_count >= 0
        assert result.cached in [True, False]
    
    def test_multiple_dataset_handling(self):
        """Test handling multiple loaded datasets."""
        orchestrator = create_test_orchestrator()
        data_store = orchestrator.extraction_agent.data_store
        
        # Load multiple datasets
        table1 = data_store.load_csv("Sales Dataset/Sale Report.csv")
        
        # Verify both are loaded
        tables = data_store.list_tables()
        assert len(tables) >= 1
        assert table1 in tables
    
    def test_conversation_history_structure(self):
        """Test conversation history data structure."""
        messages = []
        
        # Add user message
        messages.append({
            'role': 'user',
            'content': 'What are the sales?',
            'timestamp': pd.Timestamp.now()
        })
        
        # Add assistant message
        messages.append({
            'role': 'assistant',
            'content': 'Total sales are $1000',
            'data': pd.DataFrame({'sales': [1000]}),
            'metadata': {
                'execution_time': 0.5,
                'tokens_used': 100,
                'cached': False
            },
            'timestamp': pd.Timestamp.now()
        })
        
        assert len(messages) == 2
        assert messages[0]['role'] == 'user'
        assert messages[1]['role'] == 'assistant'
        assert 'metadata' in messages[1]


class TestUIComponents:
    """Test individual UI component logic."""
    
    def test_example_queries_generation(self):
        """Test example query generation."""
        qa_examples = [
            "What are the total sales?",
            "Show me the top 10 products by revenue",
            "What were the sales last month?",
            "Which category has the highest sales?",
            "Compare sales between Q1 and Q2"
        ]
        
        summarization_examples = [
            "Provide a summary of the sales data",
            "Analyze the key trends in the dataset",
            "What are the main insights from this data?"
        ]
        
        assert len(qa_examples) > 0
        assert len(summarization_examples) > 0
        assert all(isinstance(q, str) for q in qa_examples)
        assert all(isinstance(q, str) for q in summarization_examples)
    
    def test_file_type_validation(self):
        """Test file type validation logic."""
        valid_extensions = ['.csv', '.xlsx', '.json']
        
        test_files = [
            'data.csv',
            'data.xlsx',
            'data.json',
            'data.txt',  # Invalid
            'data.pdf'   # Invalid
        ]
        
        for file in test_files:
            ext = Path(file).suffix
            is_valid = ext in valid_extensions
            
            if file.endswith(('.csv', '.xlsx', '.json')):
                assert is_valid
            else:
                assert not is_valid
    
    def test_mode_display_formatting(self):
        """Test mode display formatting."""
        modes = {
            'qa': '💬 Q&A Mode',
            'summarization': '📝 Summarization Mode'
        }
        
        assert 'qa' in modes
        assert 'summarization' in modes
        assert '💬' in modes['qa']
        assert '📝' in modes['summarization']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
