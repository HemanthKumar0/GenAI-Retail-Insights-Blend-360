"""
Unit tests for LLM response validation.
"""

import pytest
from src.llm_response_validator import (
    LLMResponseValidator, ValidationResult, ResponseParser
)


class TestLLMResponseValidator:
    """Test LLMResponseValidator."""
    
    def test_validate_json_response_valid(self):
        """Test validation of valid JSON response."""
        response = '{"key": "value", "number": 42}'
        result = LLMResponseValidator.validate_json_response(response)
        
        assert result.is_valid
        assert result.parsed_data == {"key": "value", "number": 42}
        assert len(result.errors) == 0
    
    def test_validate_json_response_with_markdown(self):
        """Test validation of JSON wrapped in markdown code blocks."""
        response = '```json\n{"key": "value"}\n```'
        result = LLMResponseValidator.validate_json_response(response)
        
        assert result.is_valid
        assert result.parsed_data == {"key": "value"}
    
    def test_validate_json_response_invalid_json(self):
        """Test validation of invalid JSON."""
        response = '{invalid json}'
        result = LLMResponseValidator.validate_json_response(response)
        
        assert not result.is_valid
        assert "Invalid JSON format" in result.errors[0]
    
    def test_validate_json_response_not_object(self):
        """Test validation when JSON is not an object."""
        response = '["array", "not", "object"]'
        result = LLMResponseValidator.validate_json_response(response)
        
        assert not result.is_valid
        assert "Expected JSON object" in result.errors[0]
    
    def test_validate_json_response_missing_required_fields(self):
        """Test validation with missing required fields."""
        response = '{"key1": "value1"}'
        required_fields = ["key1", "key2", "key3"]
        result = LLMResponseValidator.validate_json_response(response, required_fields)
        
        assert not result.is_valid
        assert "Missing required fields" in result.errors[0]
        assert "key2" in result.errors[0]
        assert "key3" in result.errors[0]
    
    def test_validate_json_response_all_required_fields_present(self):
        """Test validation with all required fields present."""
        response = '{"key1": "value1", "key2": "value2"}'
        required_fields = ["key1", "key2"]
        result = LLMResponseValidator.validate_json_response(response, required_fields)
        
        assert result.is_valid
        assert result.parsed_data == {"key1": "value1", "key2": "value2"}


class TestStructuredQueryValidation:
    """Test structured query validation."""
    
    def test_validate_structured_query_valid_sql(self):
        """Test validation of valid SQL query."""
        response = '''
        {
            "operation_type": "sql",
            "operation": "SELECT * FROM sales",
            "explanation": "Get all sales records"
        }
        '''
        result = LLMResponseValidator.validate_structured_query(response)
        
        assert result.is_valid
        assert result.parsed_data["operation_type"] == "sql"
        assert result.parsed_data["operation"] == "SELECT * FROM sales"
    
    def test_validate_structured_query_valid_pandas(self):
        """Test validation of valid Pandas operation."""
        response = '''
        {
            "operation_type": "pandas",
            "operation": "df.groupby('category').sum()",
            "explanation": "Sum by category"
        }
        '''
        result = LLMResponseValidator.validate_structured_query(response)
        
        assert result.is_valid
        assert result.parsed_data["operation_type"] == "pandas"
    
    def test_validate_structured_query_valid_semantic(self):
        """Test validation of valid semantic search."""
        response = '''
        {
            "operation_type": "semantic",
            "operation": "laptop computers",
            "explanation": "Find similar products"
        }
        '''
        result = LLMResponseValidator.validate_structured_query(response)
        
        assert result.is_valid
        assert result.parsed_data["operation_type"] == "semantic"
    
    def test_validate_structured_query_invalid_operation_type(self):
        """Test validation with invalid operation type."""
        response = '''
        {
            "operation_type": "invalid",
            "operation": "some operation",
            "explanation": "test"
        }
        '''
        result = LLMResponseValidator.validate_structured_query(response)
        
        assert not result.is_valid
        assert "Invalid operation_type" in result.errors[0]
    
    def test_validate_structured_query_empty_operation(self):
        """Test validation with empty operation."""
        response = '''
        {
            "operation_type": "sql",
            "operation": "",
            "explanation": "test"
        }
        '''
        result = LLMResponseValidator.validate_structured_query(response)
        
        assert not result.is_valid
        assert "Operation field cannot be empty" in result.errors[0]
    
    def test_validate_structured_query_missing_fields(self):
        """Test validation with missing required fields."""
        response = '''
        {
            "operation_type": "sql"
        }
        '''
        result = LLMResponseValidator.validate_structured_query(response)
        
        assert not result.is_valid
        assert "Missing required fields" in result.errors[0]


class TestTimeResolutionValidation:
    """Test time resolution validation."""
    
    def test_validate_time_resolution_valid(self):
        """Test validation of valid time resolution."""
        response = '''
        {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "description": "Full year 2023"
        }
        '''
        result = LLMResponseValidator.validate_time_resolution(response)
        
        assert result.is_valid
        assert result.parsed_data["start_date"] == "2023-01-01"
        assert result.parsed_data["end_date"] == "2023-12-31"
    
    def test_validate_time_resolution_invalid_date_format(self):
        """Test validation with invalid date format."""
        response = '''
        {
            "start_date": "01/01/2023",
            "end_date": "2023-12-31",
            "description": "Invalid format"
        }
        '''
        result = LLMResponseValidator.validate_time_resolution(response)
        
        assert not result.is_valid
        assert "Invalid date format" in result.errors[0]
    
    def test_validate_time_resolution_start_after_end(self):
        """Test validation when start_date is after end_date."""
        response = '''
        {
            "start_date": "2023-12-31",
            "end_date": "2023-01-01",
            "description": "Invalid range"
        }
        '''
        result = LLMResponseValidator.validate_time_resolution(response)
        
        assert not result.is_valid
        assert "start_date must be before or equal to end_date" in result.errors[0]
    
    def test_validate_time_resolution_same_date(self):
        """Test validation when start and end dates are the same."""
        response = '''
        {
            "start_date": "2023-06-15",
            "end_date": "2023-06-15",
            "description": "Single day"
        }
        '''
        result = LLMResponseValidator.validate_time_resolution(response)
        
        assert result.is_valid


class TestNonEmptyValidation:
    """Test non-empty response validation."""
    
    def test_validate_non_empty_response_valid(self):
        """Test validation of non-empty response."""
        response = "This is a valid response"
        result = LLMResponseValidator.validate_non_empty_response(response)
        
        assert result.is_valid
        assert result.parsed_data["content"] == response
    
    def test_validate_non_empty_response_empty_string(self):
        """Test validation of empty string."""
        response = ""
        result = LLMResponseValidator.validate_non_empty_response(response)
        
        assert not result.is_valid
        assert "Response is empty" in result.errors[0]
    
    def test_validate_non_empty_response_whitespace_only(self):
        """Test validation of whitespace-only response."""
        response = "   \n\t  "
        result = LLMResponseValidator.validate_non_empty_response(response)
        
        assert not result.is_valid
        assert "Response is empty" in result.errors[0]


class TestResponseParser:
    """Test ResponseParser."""
    
    def test_parse_structured_query_valid(self):
        """Test parsing valid structured query."""
        response = '''
        {
            "operation_type": "sql",
            "operation": "SELECT * FROM sales",
            "explanation": "Get all sales"
        }
        '''
        validated = LLMResponseValidator.validate_structured_query(response)
        parsed = ResponseParser.parse_structured_query(validated)
        
        assert parsed["operation_type"] == "sql"
        assert parsed["operation"] == "SELECT * FROM sales"
        assert parsed["explanation"] == "Get all sales"
        assert parsed["parameters"] == {}
    
    def test_parse_structured_query_with_parameters(self):
        """Test parsing structured query with parameters."""
        response = '''
        {
            "operation_type": "sql",
            "operation": "SELECT * FROM sales WHERE date > ?",
            "explanation": "Get recent sales",
            "parameters": {"date": "2023-01-01"}
        }
        '''
        validated = LLMResponseValidator.validate_structured_query(response)
        parsed = ResponseParser.parse_structured_query(validated)
        
        assert parsed["parameters"] == {"date": "2023-01-01"}
    
    def test_parse_structured_query_invalid(self):
        """Test parsing invalid structured query raises error."""
        response = '{"invalid": "data"}'
        validated = LLMResponseValidator.validate_structured_query(response)
        
        with pytest.raises(ValueError, match="Cannot parse invalid result"):
            ResponseParser.parse_structured_query(validated)
    
    def test_parse_time_resolution_valid(self):
        """Test parsing valid time resolution."""
        response = '''
        {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "description": "Full year"
        }
        '''
        validated = LLMResponseValidator.validate_time_resolution(response)
        parsed = ResponseParser.parse_time_resolution(validated)
        
        assert parsed["start_date"] == "2023-01-01"
        assert parsed["end_date"] == "2023-12-31"
        assert parsed["description"] == "Full year"
    
    def test_parse_time_resolution_invalid(self):
        """Test parsing invalid time resolution raises error."""
        response = '{"invalid": "data"}'
        validated = LLMResponseValidator.validate_time_resolution(response)
        
        with pytest.raises(ValueError, match="Cannot parse invalid result"):
            ResponseParser.parse_time_resolution(validated)


class TestCleanJsonResponse:
    """Test JSON response cleaning."""
    
    def test_clean_json_with_markdown_json(self):
        """Test cleaning JSON with ```json markers."""
        response = '```json\n{"key": "value"}\n```'
        cleaned = LLMResponseValidator._clean_json_response(response)
        
        assert cleaned == '{"key": "value"}'
    
    def test_clean_json_with_markdown_plain(self):
        """Test cleaning JSON with ``` markers."""
        response = '```\n{"key": "value"}\n```'
        cleaned = LLMResponseValidator._clean_json_response(response)
        
        assert cleaned == '{"key": "value"}'
    
    def test_clean_json_with_whitespace(self):
        """Test cleaning JSON with extra whitespace."""
        response = '\n\n  {"key": "value"}  \n\n'
        cleaned = LLMResponseValidator._clean_json_response(response)
        
        assert cleaned == '{"key": "value"}'
    
    def test_clean_json_no_markers(self):
        """Test cleaning JSON without markers."""
        response = '{"key": "value"}'
        cleaned = LLMResponseValidator._clean_json_response(response)
        
        assert cleaned == '{"key": "value"}'


class TestHandleMalformedResponse:
    """Test malformed response handling."""
    
    def test_handle_malformed_response(self):
        """Test error message generation for malformed responses."""
        response = "This is not JSON at all"
        expected_format = '{"key": "value"}'
        
        error_msg = LLMResponseValidator.handle_malformed_response(
            response, expected_format
        )
        
        assert "did not match expected format" in error_msg
        assert expected_format in error_msg
        assert "This is not JSON" in error_msg
    
    def test_handle_malformed_response_truncates_long_response(self):
        """Test that long responses are truncated in error message."""
        response = "x" * 300
        expected_format = '{"key": "value"}'
        
        error_msg = LLMResponseValidator.handle_malformed_response(
            response, expected_format
        )
        
        assert "..." in error_msg
        assert len(error_msg) < len(response) + 200  # Should be truncated
