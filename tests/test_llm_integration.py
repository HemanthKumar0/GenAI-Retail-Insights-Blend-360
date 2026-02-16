"""
Integration tests for LLM components.

These tests verify that all LLM integration components work together correctly.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.llm_provider import LLMProviderFactory, LLMResponse
from src.prompt_templates import PromptTemplates
from src.llm_response_validator import LLMResponseValidator, ResponseParser
from src.context_manager import ContextManager
from src.models import Message


class TestLLMIntegration:
    """Integration tests for LLM components."""
    
    def test_query_parsing_workflow(self):
        """Test complete query parsing workflow."""
        # Setup mock LLM provider
        mock_provider = Mock()
        mock_llm_response = LLMResponse(
            content='''
            {
                "operation_type": "sql",
                "operation": "SELECT SUM(sales) FROM sales_data WHERE date >= '2023-10-01' AND date <= '2023-12-31'",
                "explanation": "Calculate total sales for Q4 2023"
            }
            ''',
            tokens_used=150,
            model="test-model"
        )
        mock_provider.generate_with_cache.return_value = mock_llm_response
        
        # Create prompt
        schema = {
            "sales_data": {
                "columns": [
                    {"name": "date", "dtype": "date"},
                    {"name": "sales", "dtype": "float"},
                    {"name": "product", "dtype": "string"}
                ]
            }
        }
        query = "What were total sales in Q4 2023?"
        context = ""
        
        prompt = PromptTemplates.format_query_parsing_prompt(query, schema, context)
        
        # Generate response
        response = mock_provider.generate_with_cache(prompt)
        
        # Validate response
        validation_result = LLMResponseValidator.validate_structured_query(response.content)
        assert validation_result.is_valid
        
        # Parse response
        parsed = ResponseParser.parse_structured_query(validation_result)
        assert parsed["operation_type"] == "sql"
        assert "SELECT SUM(sales)" in parsed["operation"]
        assert "2023-10-01" in parsed["operation"]
    
    def test_response_formatting_workflow(self):
        """Test complete response formatting workflow."""
        # Setup mock LLM provider
        mock_provider = Mock()
        mock_llm_response = LLMResponse(
            content="Total sales in Q4 2023 were $1,250,000, representing a 15% increase compared to Q3 2023.",
            tokens_used=50,
            model="test-model"
        )
        mock_provider.generate_with_cache.return_value = mock_llm_response
        
        # Create prompt
        query = "What were total sales in Q4 2023?"
        results = {"total_sales": 1250000, "growth": 0.15}
        
        prompt = PromptTemplates.format_response_prompt(query, results)
        
        # Generate response
        response = mock_provider.generate_with_cache(prompt)
        
        # Validate response is not empty
        validation_result = LLMResponseValidator.validate_non_empty_response(response.content)
        assert validation_result.is_valid
        assert "1,250,000" in response.content or "1250000" in response.content
    
    def test_context_management_workflow(self):
        """Test context management with LLM provider."""
        # Setup mock LLM provider
        mock_provider = Mock()
        mock_provider.count_tokens = lambda text: len(text) // 4
        
        # Create context manager with higher max_tokens to avoid truncation
        context_manager = ContextManager(max_tokens=4000, llm_provider=mock_provider)
        
        # Add messages
        context_manager.add_message(Message(
            role="user",
            content="What were total sales in Q4?",
            timestamp=datetime.now()
        ))
        context_manager.add_message(Message(
            role="assistant",
            content="Total sales in Q4 were $1.2M",
            timestamp=datetime.now()
        ))
        context_manager.add_message(Message(
            role="user",
            content="How about Q3?",
            timestamp=datetime.now()
        ))
        
        # Get context
        context = context_manager.get_context()
        
        assert "What were total sales in Q4?" in context
        assert "Total sales in Q4 were $1.2M" in context
        assert "How about Q3?" in context
        
        # Check stats
        stats = context_manager.get_stats()
        assert stats["message_count"] == 3
        assert stats["estimated_tokens"] > 0
    
    def test_retry_logic_with_caching(self):
        """Test retry logic and caching work together."""
        # Setup mock provider that fails once then succeeds
        call_count = {"count": 0}
        
        def mock_generate(prompt, temperature=0.7, max_tokens=None):
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise Exception("API error")
            return LLMResponse(
                content='{"operation_type": "sql", "operation": "SELECT * FROM sales", "explanation": "test"}',
                tokens_used=50,
                model="test-model"
            )
        
        mock_provider = Mock()
        mock_provider.generate = mock_generate
        mock_provider.max_retries = 3
        mock_provider._response_cache = {}
        
        # Import the actual retry method
        from src.llm_provider import LLMProvider
        mock_provider.generate_with_retry = LLMProvider.generate_with_retry.__get__(mock_provider)
        mock_provider._create_cache_key = LLMProvider._create_cache_key.__get__(mock_provider)
        mock_provider.generate_with_cache = LLMProvider.generate_with_cache.__get__(mock_provider)
        mock_provider.model = "test-model"
        
        # First call should retry and succeed
        response1 = mock_provider.generate_with_retry("test prompt")
        assert response1.content is not None
        assert call_count["count"] == 2  # Failed once, succeeded on retry
        
        # Reset for cache test
        call_count["count"] = 0
        
        # Second call with same prompt should hit cache
        response2 = mock_provider.generate_with_cache("test prompt 2")
        # First call will fail once then succeed, so count should be 2
        assert call_count["count"] == 2  # Failed once, succeeded on retry
        
        response3 = mock_provider.generate_with_cache("test prompt 2")
        assert response3.cached
        assert call_count["count"] == 2  # No additional call (cached)
    
    def test_time_resolution_workflow(self):
        """Test time period resolution workflow."""
        # Setup mock LLM provider
        mock_provider = Mock()
        mock_llm_response = LLMResponse(
            content='''
            {
                "start_date": "2023-10-01",
                "end_date": "2023-12-31",
                "description": "Q4 2023 (October through December)"
            }
            ''',
            tokens_used=80,
            model="test-model"
        )
        mock_provider.generate_with_cache.return_value = mock_llm_response
        
        # Create prompt
        prompt = PromptTemplates.format_time_resolution_prompt(
            time_reference="Q4",
            current_date="2023-11-15",
            available_range="2022-01-01 to 2023-12-31"
        )
        
        # Generate response
        response = mock_provider.generate_with_cache(prompt)
        
        # Validate response
        validation_result = LLMResponseValidator.validate_time_resolution(response.content)
        assert validation_result.is_valid
        
        # Parse response
        parsed = ResponseParser.parse_time_resolution(validation_result)
        assert parsed["start_date"] == "2023-10-01"
        assert parsed["end_date"] == "2023-12-31"
        assert "Q4" in parsed["description"]
    
    def test_error_handling_workflow(self):
        """Test error handling with malformed LLM responses."""
        # Test invalid JSON
        invalid_response = "This is not JSON at all"
        validation_result = LLMResponseValidator.validate_structured_query(invalid_response)
        assert not validation_result.is_valid
        assert len(validation_result.errors) > 0
        
        # Test missing required fields
        incomplete_response = '{"operation_type": "sql"}'
        validation_result = LLMResponseValidator.validate_structured_query(incomplete_response)
        assert not validation_result.is_valid
        assert "Missing required fields" in validation_result.errors[0]
        
        # Test invalid operation type
        invalid_type_response = '''
        {
            "operation_type": "invalid",
            "operation": "test",
            "explanation": "test"
        }
        '''
        validation_result = LLMResponseValidator.validate_structured_query(invalid_type_response)
        assert not validation_result.is_valid
        assert "Invalid operation_type" in validation_result.errors[0]
    
    def test_prompt_template_with_validation(self):
        """Test that prompt templates produce validatable responses."""
        # Test query parsing prompt
        schema = {"sales": {"columns": [{"name": "amount", "dtype": "float"}]}}
        prompt = PromptTemplates.format_query_parsing_prompt("test query", schema)
        
        assert "operation_type" in prompt
        assert "sql" in prompt.lower()
        assert "pandas" in prompt.lower()
        assert "semantic" in prompt.lower()
        
        # Test time resolution prompt
        prompt = PromptTemplates.format_time_resolution_prompt("Q4", "2023-11-15", "2022-2023")
        
        assert "start_date" in prompt
        assert "end_date" in prompt
        assert "YYYY-MM-DD" in prompt


class TestProviderFactory:
    """Test LLM provider factory integration."""
    
    def test_factory_creates_providers_with_correct_config(self):
        """Test factory creates providers with correct configuration."""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                provider = LLMProviderFactory.create_provider(
                    "gemini",
                    model="gemini-pro",
                    api_key="test-key",
                    max_retries=5
                )
                
                assert provider.model == "gemini-pro"
                assert provider.max_retries == 5
        
        with patch('openai.OpenAI'):
            provider = LLMProviderFactory.create_provider(
                "openai",
                model="gpt-4",
                api_key="test-key",
                max_retries=2
            )
            
            assert provider.model == "gpt-4"
            assert provider.max_retries == 2


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    def test_complete_query_to_response_workflow(self):
        """Test complete workflow from query to formatted response."""
        # Setup mock LLM provider
        mock_provider = Mock()
        mock_provider.count_tokens = lambda text: len(text) // 4
        
        # Step 1: Parse query
        query_response = LLMResponse(
            content='''
            {
                "operation_type": "sql",
                "operation": "SELECT product, SUM(sales) as total FROM sales_data GROUP BY product ORDER BY total DESC LIMIT 5",
                "explanation": "Get top 5 products by sales"
            }
            ''',
            tokens_used=100,
            model="test-model"
        )
        
        # Step 2: Format response
        format_response = LLMResponse(
            content="The top 5 products by sales are: Product A ($500k), Product B ($450k), Product C ($400k), Product D ($350k), and Product E ($300k).",
            tokens_used=80,
            model="test-model"
        )
        
        mock_provider.generate_with_cache.side_effect = [query_response, format_response]
        
        # Create context manager
        context_manager = ContextManager(llm_provider=mock_provider)
        
        # Add user query to context
        user_query = "What are the top 5 products by sales?"
        context_manager.add_message(Message(
            role="user",
            content=user_query,
            timestamp=datetime.now()
        ))
        
        # Step 1: Parse query
        schema = {"sales_data": {"columns": [{"name": "product", "dtype": "string"}, {"name": "sales", "dtype": "float"}]}}
        context = context_manager.get_context()
        
        parse_prompt = PromptTemplates.format_query_parsing_prompt(user_query, schema, context)
        parse_response = mock_provider.generate_with_cache(parse_prompt)
        
        # Validate and parse
        validation = LLMResponseValidator.validate_structured_query(parse_response.content)
        assert validation.is_valid
        
        parsed_query = ResponseParser.parse_structured_query(validation)
        assert parsed_query["operation_type"] == "sql"
        
        # Step 2: Format response (simulating query execution results)
        results = {
            "Product A": 500000,
            "Product B": 450000,
            "Product C": 400000,
            "Product D": 350000,
            "Product E": 300000
        }
        
        format_prompt = PromptTemplates.format_response_prompt(user_query, results, context)
        final_response = mock_provider.generate_with_cache(format_prompt)
        
        # Validate final response
        validation = LLMResponseValidator.validate_non_empty_response(final_response.content)
        assert validation.is_valid
        
        # Add assistant response to context
        context_manager.add_message(Message(
            role="assistant",
            content=final_response.content,
            timestamp=datetime.now()
        ))
        
        # Verify context contains both messages
        final_context = context_manager.get_context()
        assert user_query in final_context
        assert "top 5 products" in final_context.lower()
        
        # Check stats
        stats = context_manager.get_stats()
        assert stats["message_count"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
