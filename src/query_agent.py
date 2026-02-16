"""
Query Agent for natural language query parsing.

This module implements the QueryAgent that converts natural language queries
into structured data operations (SQL, Pandas, or semantic search).

**Validates: Requirements 2.1, 2.4**
"""

import logging
import json
import re
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from src.models import StructuredQuery, DataSchema, Message
from src.llm_provider import LLMProvider
from src.prompt_templates import PromptTemplates

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
    
    def resolve_time_period(self, time_reference: str, 
                           current_date: Optional[datetime] = None,
                           available_range: Optional[str] = None) -> Dict[str, str]:
        """
        Resolve time period references to specific date ranges.
        
        Handles references like "Q4", "last month", "YoY", etc.
        
        Args:
            time_reference: Time reference string (e.g., "Q4", "last month")
            current_date: Reference date (defaults to today)
            available_range: Available date range in dataset (optional)
            
        Returns:
            Dictionary with start_date, end_date, and description
            
        **Validates: Requirements 2.2**
        """
        if current_date is None:
            current_date = datetime.now()
        
        logger.info(f"Resolving time reference: {time_reference}")
        
        time_ref_lower = time_reference.lower().strip()
        
        # Handle quarter references (Q1, Q2, Q3, Q4)
        if re.match(r"q[1-4]", time_ref_lower):
            return self._resolve_quarter(time_ref_lower, current_date)
        
        # Handle "last month"
        if "last month" in time_ref_lower:
            return self._resolve_last_month(current_date)
        
        # Handle "this month"
        if "this month" in time_ref_lower:
            return self._resolve_this_month(current_date)
        
        # Handle "last quarter"
        if "last quarter" in time_ref_lower:
            return self._resolve_last_quarter(current_date)
        
        # Handle "this quarter"
        if "this quarter" in time_ref_lower:
            return self._resolve_this_quarter(current_date)
        
        # Handle "last year"
        if "last year" in time_ref_lower:
            return self._resolve_last_year(current_date)
        
        # Handle "this year"
        if "this year" in time_ref_lower or "ytd" in time_ref_lower:
            return self._resolve_this_year(current_date)
        
        # Handle "yoy" or "year over year"
        if "yoy" in time_ref_lower or "year over year" in time_ref_lower:
            return self._resolve_yoy(current_date)
        
        # If no pattern matches, use LLM for complex references
        return self._resolve_with_llm(time_reference, current_date, available_range)
    
    def _resolve_quarter(self, quarter_ref: str, current_date: datetime) -> Dict[str, str]:
        """Resolve quarter reference (Q1-Q4)."""
        quarter_num = int(quarter_ref[1])
        year = current_date.year
        
        # Quarter start months: Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
        start_month = (quarter_num - 1) * 3 + 1
        end_month = start_month + 2
        
        start_date = datetime(year, start_month, 1)
        
        # Last day of end month
        if end_month == 12:
            end_date = datetime(year, 12, 31)
        else:
            end_date = datetime(year, end_month + 1, 1) - timedelta(days=1)
        
        return {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "description": f"Q{quarter_num} {year} (Quarter {quarter_num})"
        }
    
    def _resolve_last_month(self, current_date: datetime) -> Dict[str, str]:
        """Resolve 'last month' reference."""
        # First day of current month
        first_of_month = current_date.replace(day=1)
        # Last day of previous month
        end_date = first_of_month - timedelta(days=1)
        # First day of previous month
        start_date = end_date.replace(day=1)
        
        return {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "description": f"Last month ({start_date.strftime('%B %Y')})"
        }
    
    def _resolve_this_month(self, current_date: datetime) -> Dict[str, str]:
        """Resolve 'this month' reference."""
        start_date = current_date.replace(day=1)
        
        # Last day of current month
        if current_date.month == 12:
            end_date = current_date.replace(day=31)
        else:
            next_month = current_date.replace(month=current_date.month + 1, day=1)
            end_date = next_month - timedelta(days=1)
        
        return {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "description": f"This month ({start_date.strftime('%B %Y')})"
        }
    
    def _resolve_last_quarter(self, current_date: datetime) -> Dict[str, str]:
        """Resolve 'last quarter' reference."""
        # Determine current quarter
        current_quarter = (current_date.month - 1) // 3 + 1
        
        # Previous quarter
        if current_quarter == 1:
            prev_quarter = 4
            year = current_date.year - 1
        else:
            prev_quarter = current_quarter - 1
            year = current_date.year
        
        return self._resolve_quarter(f"q{prev_quarter}", datetime(year, 1, 1))
    
    def _resolve_this_quarter(self, current_date: datetime) -> Dict[str, str]:
        """Resolve 'this quarter' reference."""
        current_quarter = (current_date.month - 1) // 3 + 1
        return self._resolve_quarter(f"q{current_quarter}", current_date)
    
    def _resolve_last_year(self, current_date: datetime) -> Dict[str, str]:
        """Resolve 'last year' reference."""
        year = current_date.year - 1
        
        return {
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "description": f"Last year ({year})"
        }
    
    def _resolve_this_year(self, current_date: datetime) -> Dict[str, str]:
        """Resolve 'this year' or 'YTD' reference."""
        year = current_date.year
        
        return {
            "start_date": f"{year}-01-01",
            "end_date": current_date.strftime("%Y-%m-%d"),
            "description": f"This year / YTD ({year})"
        }
    
    def _resolve_yoy(self, current_date: datetime) -> Dict[str, str]:
        """Resolve 'year over year' reference."""
        # YoY typically means same period last year
        last_year = current_date.year - 1
        
        return {
            "start_date": f"{last_year}-01-01",
            "end_date": f"{last_year}-12-31",
            "description": f"Year over year comparison (comparing to {last_year})"
        }
    
    def _resolve_with_llm(self, time_reference: str, current_date: datetime,
                         available_range: Optional[str]) -> Dict[str, str]:
        """Use LLM to resolve complex time references."""
        try:
            prompt = PromptTemplates.format_time_resolution_prompt(
                time_reference=time_reference,
                current_date=current_date.strftime("%Y-%m-%d"),
                available_range=available_range or "Unknown"
            )
            
            response = self.llm_provider.generate_with_cache(
                prompt=prompt,
                temperature=0.3
            )
            
            # Parse JSON response
            json_text = self._extract_json(response.content)
            data = json.loads(json_text)
            
            return {
                "start_date": data["start_date"],
                "end_date": data["end_date"],
                "description": data["description"]
            }
            
        except Exception as e:
            logger.error(f"Failed to resolve time reference with LLM: {str(e)}")
            # Fallback: return current year
            return self._resolve_this_year(current_date)
    
    def detect_ambiguity(self, query: str, schema: DataSchema) -> Optional[str]:
        """
        Detect if a query is ambiguous and needs clarification.
        
        Args:
            query: User's natural language query
            schema: Available data schema
            
        Returns:
            Ambiguity reason if detected, None otherwise
            
        **Validates: Requirements 2.3**
        """
        logger.info(f"Checking query for ambiguity: {query}")
        
        query_lower = query.lower()
        
        # Check for ambiguous column references
        ambiguous_terms = ["sales", "revenue", "total", "amount", "value"]
        for term in ambiguous_terms:
            if term in query_lower:
                # Check if multiple columns match this term
                matching_columns = self._find_matching_columns(term, schema)
                if len(matching_columns) > 1:
                    return f"Multiple columns match '{term}': {', '.join(matching_columns)}"
        
        # Check for ambiguous time references
        if self._has_ambiguous_time_reference(query_lower):
            return "Time period is ambiguous or unclear"
        
        # Check for missing required context
        if self._missing_required_context(query_lower, schema):
            return "Query requires additional context (e.g., which table, time period, or category)"
        
        return None
    
    def resolve_ambiguity(self, query: str, schema: DataSchema, 
                         ambiguity_reason: str) -> str:
        """
        Generate clarifying questions for ambiguous queries.
        
        Args:
            query: User's ambiguous query
            schema: Available data schema
            ambiguity_reason: Reason for ambiguity
            
        Returns:
            Clarifying questions as a string
            
        **Validates: Requirements 2.3**
        """
        logger.info(f"Generating clarification for: {ambiguity_reason}")
        
        try:
            # Convert schema to dict
            schema_dict = self._schema_to_dict(schema)
            
            # Format prompt
            prompt = PromptTemplates.format_ambiguity_prompt(
                query=query,
                schema=schema_dict,
                ambiguity_reason=ambiguity_reason
            )
            
            # Generate clarifying questions
            response = self.llm_provider.generate_with_cache(
                prompt=prompt,
                temperature=0.7
            )
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate clarification: {str(e)}")
            # Fallback to generic clarification
            return f"Your query is ambiguous: {ambiguity_reason}. Could you please provide more details?"
    
    def _find_matching_columns(self, term: str, schema: DataSchema) -> list:
        """Find columns that match a given term."""
        matching = []
        
        for table_name, table_schema in schema.tables.items():
            for col in table_schema.columns:
                if term in col.name.lower():
                    matching.append(f"{table_name}.{col.name}")
        
        return matching
    
    def _has_ambiguous_time_reference(self, query: str) -> bool:
        """Check if query has ambiguous time reference."""
        # Ambiguous time words without specific context
        ambiguous_time_words = ["recently", "soon", "earlier", "later", "before", "after"]
        
        for word in ambiguous_time_words:
            if word in query and not any(specific in query for specific in 
                                        ["q1", "q2", "q3", "q4", "month", "year", "quarter"]):
                return True
        
        return False
    
    def _missing_required_context(self, query: str, schema: DataSchema) -> bool:
        """Check if query is missing required context."""
        # Very short queries often lack context
        if len(query.split()) < 3:
            return True
        
        # Queries with "it", "that", "this" without prior context
        pronouns = ["it", "that", "this", "them", "those"]
        for pronoun in pronouns:
            if query.startswith(pronoun):
                return True
        
        return False
    
    def parse_query_with_context(self, query: str, schema: DataSchema,
                                 conversation_history: list) -> StructuredQuery:
        """
        Parse query with conversation context for pronoun resolution.
        
        Args:
            query: User's natural language query
            schema: Available data schema
            conversation_history: List of previous messages
            
        Returns:
            StructuredQuery object
            
        **Validates: Requirements 2.6**
        """
        logger.info(f"Parsing query with conversation context")
        
        # Build context string from conversation history
        context = self._build_context_string(conversation_history)
        
        # Check if query uses pronouns or references
        if self._uses_context_references(query):
            logger.info("Query uses context references, including conversation history")
            return self.parse_query(query, schema, context)
        else:
            # Query is self-contained, parse without full context
            return self.parse_query(query, schema, "")
    
    def _build_context_string(self, conversation_history: list) -> str:
        """
        Build context string from conversation history.
        
        Args:
            conversation_history: List of Message objects or dicts
            
        Returns:
            Formatted context string
        """
        if not conversation_history:
            return ""
        
        context_parts = []
        
        # Include last 5 exchanges for context
        recent_history = conversation_history[-10:]  # Last 10 messages = ~5 exchanges
        
        for msg in recent_history:
            if isinstance(msg, Message):
                role = msg.role
                content = msg.content
            elif isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                continue
            
            context_parts.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(context_parts)
    
    def _uses_context_references(self, query: str) -> bool:
        """
        Check if query uses pronouns or references to previous context.
        
        Args:
            query: User's query
            
        Returns:
            True if query uses context references
        """
        query_lower = query.lower()
        
        # Pronouns and references
        context_words = [
            "it", "that", "this", "these", "those", "them",
            "same", "also", "too", "as well",
            "previous", "last", "earlier", "above",
            "what about", "how about"
        ]
        
        for word in context_words:
            if word in query_lower:
                return True
        
        return False
    
    def resolve_pronouns(self, query: str, conversation_history: list) -> str:
        """
        Resolve pronouns in query using conversation context.
        
        Args:
            query: User's query with pronouns
            conversation_history: List of previous messages
            
        Returns:
            Query with pronouns resolved
            
        **Validates: Requirements 2.6**
        """
        if not self._uses_context_references(query):
            return query
        
        try:
            # Build context
            context = self._build_context_string(conversation_history)
            
            # Use LLM to resolve pronouns
            prompt = f"""Given the conversation context, rewrite the user's query to be self-contained by replacing pronouns and references with their actual referents.

Conversation Context:
{context}

User Query: {query}

Rewritten Query (self-contained):"""
            
            response = self.llm_provider.generate_with_cache(
                prompt=prompt,
                temperature=0.3
            )
            
            resolved_query = response.content.strip()
            logger.info(f"Resolved pronouns: '{query}' -> '{resolved_query}'")
            
            return resolved_query
            
        except Exception as e:
            logger.error(f"Failed to resolve pronouns: {str(e)}")
            # Return original query if resolution fails
            return query
    
    def generate_error_message(self, error: Exception, query: str, 
                              schema: DataSchema) -> str:
        """
        Generate helpful error message for failed query parsing.
        
        Args:
            error: The exception that occurred
            query: User's original query
            schema: Available data schema
            
        Returns:
            User-friendly error message with suggestions
            
        **Validates: Requirements 2.5**
        """
        logger.info(f"Generating error message for: {type(error).__name__}")
        
        error_type = type(error).__name__
        error_message = str(error)
        
        try:
            # Use LLM to generate helpful error message
            prompt = PromptTemplates.format_error_explanation_prompt(
                error_type=error_type,
                error_message=error_message,
                query=query
            )
            
            response = self.llm_provider.generate_with_cache(
                prompt=prompt,
                temperature=0.7
            )
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate error message: {str(e)}")
            # Fallback to basic error message
            return self._generate_fallback_error_message(query, schema)
    
    def _generate_fallback_error_message(self, query: str, 
                                        schema: DataSchema) -> str:
        """Generate basic fallback error message."""
        message = f"I couldn't understand your query: '{query}'\n\n"
        message += "Here are some suggestions:\n"
        message += "1. Try rephrasing your question more specifically\n"
        message += "2. Include the time period you're interested in (e.g., 'in Q4', 'last month')\n"
        message += "3. Specify which metric you want (e.g., 'total sales', 'revenue', 'count')\n\n"
        
        # Add available tables info
        if schema.tables:
            message += "Available data:\n"
            for table_name in schema.tables.keys():
                message += f"  - {table_name}\n"
        
        message += "\nExample queries:\n"
        message += "  - What were total sales in Q4?\n"
        message += "  - Show me top 5 products by revenue\n"
        message += "  - Calculate year-over-year growth\n"
        
        return message
    
    def suggest_alternative_queries(self, failed_query: str, 
                                   schema: DataSchema) -> list:
        """
        Suggest alternative query phrasings.
        
        Args:
            failed_query: Query that failed to parse
            schema: Available data schema
            
        Returns:
            List of suggested alternative queries
            
        **Validates: Requirements 2.5**
        """
        suggestions = []
        
        query_lower = failed_query.lower()
        
        # Suggest adding time period if missing
        time_words = ["q1", "q2", "q3", "q4", "month", "year", "quarter", "last", "this"]
        if not any(word in query_lower for word in time_words):
            suggestions.append(f"{failed_query} in Q4 2023")
            suggestions.append(f"{failed_query} last month")
        
        # Suggest being more specific about metrics
        vague_words = ["show", "get", "find", "see"]
        if any(word in query_lower for word in vague_words):
            suggestions.append(f"What were the total sales {failed_query.replace('show', '').replace('get', '').strip()}?")
        
        # Suggest table-specific queries if schema available
        if schema.tables:
            table_name = list(schema.tables.keys())[0]
            suggestions.append(f"Show me summary statistics for {table_name}")
        
        return suggestions[:3]  # Return top 3 suggestions
