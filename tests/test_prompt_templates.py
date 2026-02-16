"""
Unit tests for prompt templates.
"""

import pytest
from src.prompt_templates import PromptTemplates


class TestPromptTemplates:
    """Test PromptTemplates class."""
    
    def test_format_query_parsing_prompt(self):
        """Test formatting query parsing prompt."""
        schema = {
            "sales": {
                "columns": [
                    {"name": "date", "dtype": "date"},
                    {"name": "amount", "dtype": "float"}
                ]
            }
        }
        query = "What were total sales last month?"
        context = "User previously asked about Q4 sales"
        
        prompt = PromptTemplates.format_query_parsing_prompt(query, schema, context)
        
        assert "What were total sales last month?" in prompt
        assert "sales" in prompt
        assert "date" in prompt
        assert "amount" in prompt
        assert "User previously asked about Q4 sales" in prompt
        assert "few-shot" in prompt.lower() or "example" in prompt.lower()
    
    def test_format_query_parsing_prompt_no_context(self):
        """Test formatting query parsing prompt without context."""
        schema = {"sales": {"columns": []}}
        query = "test query"
        
        prompt = PromptTemplates.format_query_parsing_prompt(query, schema)
        
        assert "test query" in prompt
        assert "No previous context" in prompt
    
    def test_format_response_prompt(self):
        """Test formatting response prompt."""
        query = "What were total sales?"
        results = {"total_sales": 10000}
        context = "Dataset: Q4 2023"
        
        prompt = PromptTemplates.format_response_prompt(query, results, context)
        
        assert "What were total sales?" in prompt
        assert "total_sales" in prompt
        assert "10000" in prompt
        assert "Q4 2023" in prompt
    
    def test_format_summarization_prompt(self):
        """Test formatting summarization prompt."""
        dataset_info = {
            "table_name": "sales",
            "row_count": 1000,
            "date_range": "2023-01-01 to 2023-12-31"
        }
        metrics = {
            "total_sales": 500000,
            "avg_order_value": 50.0,
            "growth_rate": 0.15
        }
        
        prompt = PromptTemplates.format_summarization_prompt(dataset_info, metrics)
        
        assert "sales" in prompt
        assert "1000" in prompt
        assert "500000" in prompt
        assert "50.0" in prompt or "50.00" in prompt
        assert "0.15" in prompt or "15" in prompt
    
    def test_format_ambiguity_prompt(self):
        """Test formatting ambiguity clarification prompt."""
        query = "Show me sales"
        schema = {"sales": {"columns": []}}
        ambiguity_reason = "Multiple interpretations possible"
        
        prompt = PromptTemplates.format_ambiguity_prompt(query, schema, ambiguity_reason)
        
        assert "Show me sales" in prompt
        assert "Multiple interpretations possible" in prompt
        assert "clarifying" in prompt.lower()
    
    def test_format_context_summary_prompt(self):
        """Test formatting context summary prompt."""
        history = [
            {"role": "user", "content": "What were Q4 sales?"},
            {"role": "assistant", "content": "Q4 sales were $100k"},
            {"role": "user", "content": "How about Q3?"}
        ]
        
        prompt = PromptTemplates.format_context_summary_prompt(history)
        
        assert "What were Q4 sales?" in prompt
        assert "Q4 sales were $100k" in prompt
        assert "How about Q3?" in prompt
    
    def test_format_error_explanation_prompt(self):
        """Test formatting error explanation prompt."""
        error_type = "ValidationError"
        error_message = "Invalid date format"
        query = "Show sales for 13/45/2023"
        
        prompt = PromptTemplates.format_error_explanation_prompt(
            error_type, error_message, query
        )
        
        assert "ValidationError" in prompt
        assert "Invalid date format" in prompt
        assert "13/45/2023" in prompt
    
    def test_format_time_resolution_prompt(self):
        """Test formatting time resolution prompt."""
        time_reference = "Q4"
        current_date = "2023-11-15"
        available_range = "2022-01-01 to 2023-12-31"
        
        prompt = PromptTemplates.format_time_resolution_prompt(
            time_reference, current_date, available_range
        )
        
        assert "Q4" in prompt
        assert "2023-11-15" in prompt
        assert "2022-01-01 to 2023-12-31" in prompt


class TestSchemaFormatting:
    """Test schema formatting helper."""
    
    def test_format_schema_with_columns(self):
        """Test formatting schema with column information."""
        schema = {
            "sales": {
                "columns": [
                    {"name": "id", "dtype": "int"},
                    {"name": "amount", "dtype": "float"}
                ]
            }
        }
        
        formatted = PromptTemplates._format_schema(schema)
        
        assert "Table: sales" in formatted
        assert "id (int)" in formatted
        assert "amount (float)" in formatted
    
    def test_format_schema_empty(self):
        """Test formatting empty schema."""
        schema = {}
        
        formatted = PromptTemplates._format_schema(schema)
        
        assert "No schema available" in formatted
    
    def test_format_schema_simple_columns(self):
        """Test formatting schema with simple column list."""
        schema = {
            "sales": {
                "columns": ["id", "amount", "date"]
            }
        }
        
        formatted = PromptTemplates._format_schema(schema)
        
        assert "Table: sales" in formatted
        assert "id" in formatted
        assert "amount" in formatted
        assert "date" in formatted


class TestResultsFormatting:
    """Test results formatting helper."""
    
    def test_format_results_dataframe(self):
        """Test formatting DataFrame results."""
        import pandas as pd
        
        df = pd.DataFrame({
            "product": ["A", "B", "C"],
            "sales": [100, 200, 300]
        })
        
        formatted = PromptTemplates._format_results(df)
        
        assert "product" in formatted
        assert "sales" in formatted
        assert "Total rows: 3" in formatted
    
    def test_format_results_empty_dataframe(self):
        """Test formatting empty DataFrame."""
        import pandas as pd
        
        df = pd.DataFrame()
        
        formatted = PromptTemplates._format_results(df)
        
        assert "No results found" in formatted
    
    def test_format_results_dict(self):
        """Test formatting dictionary results."""
        results = {
            "total_sales": 10000,
            "avg_order": 50
        }
        
        formatted = PromptTemplates._format_results(results)
        
        assert "total_sales: 10000" in formatted
        assert "avg_order: 50" in formatted
    
    def test_format_results_other(self):
        """Test formatting other result types."""
        results = "Simple string result"
        
        formatted = PromptTemplates._format_results(results)
        
        assert formatted == "Simple string result"


class TestMetricsFormatting:
    """Test metrics formatting helper."""
    
    def test_format_metrics_with_floats(self):
        """Test formatting metrics with float values."""
        metrics = {
            "growth_rate": 0.156789,
            "avg_value": 123.456
        }
        
        formatted = PromptTemplates._format_metrics(metrics)
        
        assert "growth_rate: 0.16" in formatted
        assert "avg_value: 123.46" in formatted
    
    def test_format_metrics_with_mixed_types(self):
        """Test formatting metrics with mixed types."""
        metrics = {
            "count": 100,
            "rate": 0.15,
            "name": "test"
        }
        
        formatted = PromptTemplates._format_metrics(metrics)
        
        assert "count: 100" in formatted
        assert "rate: 0.15" in formatted
        assert "name: test" in formatted


class TestConversationHistoryFormatting:
    """Test conversation history formatting helper."""
    
    def test_format_conversation_history(self):
        """Test formatting conversation history."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]
        
        formatted = PromptTemplates._format_conversation_history(history)
        
        assert "User: Hello" in formatted
        assert "Assistant: Hi there" in formatted
        assert "User: How are you?" in formatted
    
    def test_format_conversation_history_empty(self):
        """Test formatting empty conversation history."""
        history = []
        
        formatted = PromptTemplates._format_conversation_history(history)
        
        assert formatted == ""
