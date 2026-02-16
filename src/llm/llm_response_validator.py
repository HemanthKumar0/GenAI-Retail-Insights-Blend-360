"""
LLM response validation utilities.

This module provides validation for LLM responses to ensure they follow
expected formats and contain required fields.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of response validation."""
    is_valid: bool
    parsed_data: Optional[Dict[str, Any]] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class LLMResponseValidator:
    """Validator for LLM responses."""
    
    @staticmethod
    def validate_json_response(response: str, required_fields: List[str] = None) -> ValidationResult:
        """
        Validate that response is valid JSON with required fields.
        
        Args:
            response: LLM response string
            required_fields: List of required field names
            
        Returns:
            ValidationResult with parsed data or errors
        """
        errors = []
        
        # Try to parse JSON
        try:
            # Clean response - remove markdown code blocks if present
            cleaned_response = LLMResponseValidator._clean_json_response(response)
            parsed_data = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format: {str(e)}")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check if parsed data is a dictionary
        if not isinstance(parsed_data, dict):
            errors.append(f"Expected JSON object, got {type(parsed_data).__name__}")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check required fields
        if required_fields:
            missing_fields = [field for field in required_fields if field not in parsed_data]
            if missing_fields:
                errors.append(f"Missing required fields: {', '.join(missing_fields)}")
                return ValidationResult(is_valid=False, parsed_data=parsed_data, errors=errors)
        
        return ValidationResult(is_valid=True, parsed_data=parsed_data)
    
    @staticmethod
    def validate_structured_query(response: str) -> ValidationResult:
        """
        Validate structured query response from query parsing.
        
        Expected format:
        {
            "operation_type": "sql|pandas|semantic",
            "operation": "<query or code>",
            "explanation": "<description>"
        }
        
        Args:
            response: LLM response string
            
        Returns:
            ValidationResult with parsed query or errors
        """
        required_fields = ["operation_type", "operation", "explanation"]
        result = LLMResponseValidator.validate_json_response(response, required_fields)
        
        if not result.is_valid:
            return result
        
        # Additional validation for operation_type
        valid_types = ["sql", "pandas", "semantic"]
        operation_type = result.parsed_data.get("operation_type", "").lower()
        
        if operation_type not in valid_types:
            result.errors.append(
                f"Invalid operation_type: '{operation_type}'. "
                f"Must be one of: {', '.join(valid_types)}"
            )
            result.is_valid = False
            return result
        
        # Validate operation is not empty
        operation = result.parsed_data.get("operation", "").strip()
        if not operation:
            result.errors.append("Operation field cannot be empty")
            result.is_valid = False
            return result
        
        return result
    
    @staticmethod
    def validate_time_resolution(response: str) -> ValidationResult:
        """
        Validate time resolution response.
        
        Expected format:
        {
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD",
            "description": "Human-readable description"
        }
        
        Args:
            response: LLM response string
            
        Returns:
            ValidationResult with parsed dates or errors
        """
        required_fields = ["start_date", "end_date", "description"]
        result = LLMResponseValidator.validate_json_response(response, required_fields)
        
        if not result.is_valid:
            return result
        
        # Validate date format
        from datetime import datetime
        
        for date_field in ["start_date", "end_date"]:
            date_str = result.parsed_data.get(date_field, "")
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                result.errors.append(
                    f"Invalid date format for {date_field}: '{date_str}'. "
                    f"Expected YYYY-MM-DD"
                )
                result.is_valid = False
        
        # Validate start_date <= end_date
        if result.is_valid:
            try:
                start = datetime.strptime(result.parsed_data["start_date"], "%Y-%m-%d")
                end = datetime.strptime(result.parsed_data["end_date"], "%Y-%m-%d")
                if start > end:
                    result.errors.append("start_date must be before or equal to end_date")
                    result.is_valid = False
            except Exception as e:
                result.errors.append(f"Error validating date range: {str(e)}")
                result.is_valid = False
        
        return result
    
    @staticmethod
    def validate_non_empty_response(response: str) -> ValidationResult:
        """
        Validate that response is not empty.
        
        Args:
            response: LLM response string
            
        Returns:
            ValidationResult
        """
        if not response or not response.strip():
            return ValidationResult(
                is_valid=False,
                errors=["Response is empty"]
            )
        
        return ValidationResult(is_valid=True, parsed_data={"content": response})
    
    @staticmethod
    def _clean_json_response(response: str) -> str:
        """
        Clean JSON response by removing markdown code blocks and extra whitespace.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks
        if "```json" in response:
            # Extract content between ```json and ```
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                response = response[start:end]
        elif "```" in response:
            # Extract content between ``` and ```
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                response = response[start:end]
        
        # Strip whitespace
        response = response.strip()
        
        return response
    
    @staticmethod
    def handle_malformed_response(response: str, expected_format: str) -> str:
        """
        Generate helpful error message for malformed responses.
        
        Args:
            response: Malformed response
            expected_format: Description of expected format
            
        Returns:
            Error message string
        """
        return (
            f"LLM response did not match expected format.\n\n"
            f"Expected format:\n{expected_format}\n\n"
            f"Received:\n{response[:200]}{'...' if len(response) > 200 else ''}\n\n"
            f"Please try rephrasing your query or contact support if the issue persists."
        )


class ResponseParser:
    """Helper class for parsing validated responses."""
    
    @staticmethod
    def parse_structured_query(validated_result: ValidationResult) -> Dict[str, Any]:
        """
        Parse validated structured query result.
        
        Args:
            validated_result: ValidationResult from validate_structured_query
            
        Returns:
            Dictionary with operation details
            
        Raises:
            ValueError: If result is not valid
        """
        if not validated_result.is_valid:
            raise ValueError(f"Cannot parse invalid result: {validated_result.errors}")
        
        return {
            "operation_type": validated_result.parsed_data["operation_type"].lower(),
            "operation": validated_result.parsed_data["operation"],
            "explanation": validated_result.parsed_data["explanation"],
            "parameters": validated_result.parsed_data.get("parameters", {})
        }
    
    @staticmethod
    def parse_time_resolution(validated_result: ValidationResult) -> Dict[str, str]:
        """
        Parse validated time resolution result.
        
        Args:
            validated_result: ValidationResult from validate_time_resolution
            
        Returns:
            Dictionary with start_date, end_date, description
            
        Raises:
            ValueError: If result is not valid
        """
        if not validated_result.is_valid:
            raise ValueError(f"Cannot parse invalid result: {validated_result.errors}")
        
        return {
            "start_date": validated_result.parsed_data["start_date"],
            "end_date": validated_result.parsed_data["end_date"],
            "description": validated_result.parsed_data["description"]
        }
