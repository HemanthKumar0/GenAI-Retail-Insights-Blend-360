"""
Query Agent for natural language query parsing.

This module implements the QueryAgent that converts natural language queries
into structured data operations (SQL, Pandas, or semantic search).
"""

import logging
import json
import re
from typing import Dict, Any, Optional

from src.core.models import StructuredQuery, DataSchema
from src.llm.llm_provider import LLMProvider
from src.llm.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class QueryAgent:
    """
    Agent responsible for interpreting natural language queries.
    
    Converts user queries into structured data operations using LLM,
    handles time period resolution, and manages ambiguity.
    """
    
    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize QueryAgent.
        
        Args:
            llm_provider: LLM provider for query parsing
        """
        self.llm_provider = llm_provider
        logger.info("QueryAgent initialized")
    
    def parse_query(self, query: str, schema: DataSchema, 
                   context: str = "") -> StructuredQuery:
        """
        Parse natural language query into structured operation.
        
        Args:
            query: User's natural language query
            schema: Available data schema
            context: Conversation context
            
        Returns:
            StructuredQuery object with operation details
            
        Raises:
            ValueError: If query cannot be parsed
        """
        logger.info(f"Parsing query: {query}")
        
        try:
            # Convert schema to dict format for prompt
            schema_dict = self._schema_to_dict(schema)
            
            # Format prompt
            prompt = PromptTemplates.format_query_parsing_prompt(
                query=query,
                schema=schema_dict,
                context=context
            )
            
            # Generate structured query using LLM
            response = self.llm_provider.generate_with_cache(
                prompt=prompt,
                temperature=0.3  # Lower temperature for more consistent parsing
            )
            
            # Parse JSON response
            structured_query = self._parse_llm_response(response.content)
            
            logger.info(f"Successfully parsed query: {structured_query.operation_type}")
            return structured_query
            
        except Exception as e:
            logger.error(f"Failed to parse query: {str(e)}")
            raise ValueError(f"Could not parse query: {str(e)}")
    
    def _parse_llm_response(self, response_text: str) -> StructuredQuery:
        """
        Parse LLM response into StructuredQuery.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            StructuredQuery object
            
        Raises:
            ValueError: If response is invalid
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_text = self._extract_json(response_text)
            
            # Parse JSON
            data = json.loads(json_text)
            
            # Validate required fields
            required_fields = ["operation_type", "operation", "explanation"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Create StructuredQuery
            return StructuredQuery(
                operation_type=data["operation_type"],
                operation=data["operation"],
                explanation=data["explanation"],
                parameters=data.get("parameters", {})
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in LLM response: {str(e)}")
            raise ValueError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            raise ValueError(f"Failed to parse response: {str(e)}")
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from text (handles markdown code blocks).
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Extracted JSON string
        """
        # Try to find JSON in markdown code block
        code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(code_block_pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Try to find JSON object directly
        json_pattern = r"\{.*\}"
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            return match.group(0)
        
        # Return as-is if no pattern found
        return text.strip()
    
    def _schema_to_dict(self, schema: DataSchema) -> Dict[str, Any]:
        """
        Convert DataSchema to dictionary format for prompts.
        
        Args:
            schema: DataSchema object
            
        Returns:
            Dictionary representation of schema
        """
        schema_dict = {}
        
        for table_name, table_schema in schema.tables.items():
            schema_dict[table_name] = {
                "columns": [
                    {
                        "name": col.name,
                        "dtype": col.dtype,
                        "nullable": col.nullable,
                        "sample_values": col.sample_values[:3]  # First 3 samples
                    }
                    for col in table_schema.columns
                ],
                "row_count": table_schema.row_count
            }
        
        return schema_dict
    

