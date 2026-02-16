"""
Orchestrator module for coordinating multi-agent interactions.

This module implements the Orchestrator that coordinates communication between
QueryAgent, ExtractionAgent, and ValidationAgent to process user queries.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**
"""

import logging
import time
from typing import Optional, List
from datetime import datetime

from src.core.models import (
    Message, Response, StructuredQuery, QueryResult, 
    ValidationResult, DataSchema
)
from src.agents.query_agent import QueryAgent
from src.agents.extraction_agent import ExtractionAgent
from src.agents.validation_agent import ValidationAgent
from src.data.context_manager import ContextManager
from src.llm.llm_provider import LLMProvider
from src.llm.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrator coordinates agent communication and manages conversation flow.
    
    This class:
    - Routes queries to QueryAgent for interpretation
    - Forwards structured operations to ExtractionAgent
    - Sends results to ValidationAgent for verification
    - Retries up to 3 times on validation failures
    - Logs all inter-agent communications
    - Formats final responses using LLM
    - Maintains conversation context
    """
    
    def __init__(
        self,
        query_agent: QueryAgent,
        extraction_agent: ExtractionAgent,
        validation_agent: ValidationAgent,
        llm_provider: LLMProvider,
        max_retries: int = 3
    ):
        """
        Initialize the Orchestrator.
        
        Args:
            query_agent: QueryAgent for NL parsing
            extraction_agent: ExtractionAgent for query execution
            validation_agent: ValidationAgent for result verification
            llm_provider: LLM provider for response formatting
            max_retries: Maximum retry attempts for validation failures
        """
        self.query_agent = query_agent
        self.extraction_agent = extraction_agent
        self.validation_agent = validation_agent
        self.llm_provider = llm_provider
        self.max_retries = max_retries
        
        # Initialize context manager
        self.context_manager = ContextManager(
            max_tokens=4000,
            llm_provider=llm_provider
        )
        
        # Communication log for debugging and monitoring
        self.communication_log: List[dict] = []
        
        logger.info(
            f"Orchestrator initialized with max_retries={max_retries}"
        )
    
    def process_query(self, user_query: str, mode: str = "qa") -> Response:
        """
        Process a user query through the agent pipeline.
        
        This method orchestrates the complete query processing flow:
        1. Route query to QueryAgent for interpretation
        2. Forward structured query to ExtractionAgent
        3. Send results to ValidationAgent for verification
        4. Retry up to max_retries times on validation failures
        5. Format final response using LLM
        
        Args:
            user_query: Natural language query from user
            mode: Operating mode ("summarization" or "qa")
            
        Returns:
            Response object containing answer and metadata
            
        Raises:
            ValueError: If mode is invalid or query processing fails
            
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
        """
        if mode not in ["summarization", "qa"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'summarization' or 'qa'")
        
        logger.info(f"Processing query in {mode} mode: {user_query}")
        start_time = time.time()
        
        # Add user message to context
        user_message = Message(
            role="user",
            content=user_query,
            timestamp=datetime.now()
        )
        self.context_manager.add_message(user_message)
        
        # Get data schema from extraction agent
        schema = self._get_data_schema()
        
        # Get conversation context
        conversation_context = self.context_manager.get_context(prioritize_recent=True)
        
        # Attempt query processing with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Query processing attempt {attempt + 1}/{self.max_retries}")
                
                # Step 1: Route to QueryAgent for interpretation
                self._log_communication(
                    "Orchestrator", "QueryAgent",
                    f"Parse query: {user_query}"
                )
                
                structured_query = self.query_agent.parse_query(
                    query=user_query,
                    schema=schema,
                    context=conversation_context
                )
                
                self._log_communication(
                    "QueryAgent", "Orchestrator",
                    f"Structured query: {structured_query.operation_type} - {structured_query.explanation}"
                )
                
                # Step 2: Forward to ExtractionAgent
                self._log_communication(
                    "Orchestrator", "ExtractionAgent",
                    f"Execute query: {structured_query.explanation}"
                )
                
                query_result = self.extraction_agent.execute_query(structured_query)
                
                self._log_communication(
                    "ExtractionAgent", "Orchestrator",
                    f"Query result: {query_result.row_count} rows in {query_result.execution_time:.2f}s"
                )
                
                # Step 3: Send to ValidationAgent
                self._log_communication(
                    "Orchestrator", "ValidationAgent",
                    f"Validate results: {query_result.row_count} rows"
                )
                
                validation_result = self.validation_agent.validate_results(
                    results=query_result,
                    query=structured_query
                )
                
                self._log_communication(
                    "ValidationAgent", "Orchestrator",
                    f"Validation: {'PASSED' if validation_result.passed else 'FAILED'} "
                    f"(confidence: {validation_result.confidence:.2f})"
                )
                
                # Step 4: Check validation result
                if validation_result.passed:
                    # Validation passed, format and return response
                    logger.info(f"Validation passed on attempt {attempt + 1}")
                    
                    response = self._format_response(
                        user_query=user_query,
                        query_result=query_result,
                        validation_result=validation_result,
                        execution_time=time.time() - start_time
                    )
                    
                    # Add assistant response to context
                    assistant_message = Message(
                        role="assistant",
                        content=response.answer,
                        timestamp=datetime.now(),
                        metadata=response.metadata
                    )
                    self.context_manager.add_message(assistant_message)
                    
                    return response
                
                else:
                    # Validation failed
                    logger.warning(
                        f"Validation failed on attempt {attempt + 1}: "
                        f"{len(validation_result.issues)} issues found"
                    )
                    
                    if attempt < self.max_retries - 1:
                        # Request QueryAgent to reformulate
                        logger.info("Requesting query reformulation")
                        user_query = self._reformulate_query(
                            original_query=user_query,
                            validation_result=validation_result,
                            structured_query=structured_query
                        )
                        
                        self._log_communication(
                            "Orchestrator", "QueryAgent",
                            f"Reformulate query due to validation failure: {validation_result.issues[0]}"
                        )
                    else:
                        # Max retries reached, return error
                        logger.error(f"Max retries ({self.max_retries}) reached, returning error")
                        return self._create_error_response(
                            user_query=user_query,
                            validation_result=validation_result,
                            execution_time=time.time() - start_time
                        )
            
            except Exception as e:
                logger.error(f"Error during query processing (attempt {attempt + 1}): {str(e)}")
                
                if attempt < self.max_retries - 1:
                    logger.info("Retrying query processing")
                    continue
                else:
                    # Max retries reached, return error
                    return Response(
                        answer=f"I encountered an error processing your query: {str(e)}",
                        metadata={
                            "error": str(e),
                            "execution_time": time.time() - start_time,
                            "attempts": attempt + 1
                        }
                    )
        
        # Should not reach here, but just in case
        return Response(
            answer="I was unable to process your query after multiple attempts.",
            metadata={
                "execution_time": time.time() - start_time,
                "attempts": self.max_retries
            }
        )

    def _get_data_schema(self) -> DataSchema:
        """
        Get data schema from extraction agent's data store.
        
        Returns:
            DataSchema object with table information
        """
        from src.core.models import TableSchema, ColumnInfo
        
        schema = DataSchema()
        
        # Get tables from data store
        tables = self.extraction_agent.data_store.list_tables()
        
        for table_name in tables:
            table_schema_dict = self.extraction_agent.data_store.get_table_schema(table_name)
            
            # Convert to TableSchema object
            columns = []
            columns_data = table_schema_dict["columns"]
            
            # Handle both dict and list formats
            if isinstance(columns_data, dict):
                # Dict format: {col_name: {dtype, nullable, ...}}
                for col_name, col_info in columns_data.items():
                    columns.append(ColumnInfo(
                        name=col_name,
                        dtype=col_info.get("dtype", col_info.get("type", "unknown")),
                        nullable=col_info.get("nullable", False),
                        unique_count=col_info.get("unique_count", 0),
                        sample_values=col_info.get("sample_values", [])
                    ))
            else:
                # List format: [{name, type, nullable}, ...]
                for col_info in columns_data:
                    columns.append(ColumnInfo(
                        name=col_info["name"],
                        dtype=col_info.get("type", col_info.get("dtype", "unknown")),
                        nullable=col_info.get("nullable", False),
                        unique_count=col_info.get("unique_count", 0),
                        sample_values=col_info.get("sample_values", [])
                    ))
            
            table_schema = TableSchema(
                name=table_name,
                columns=columns,
                row_count=table_schema_dict["row_count"],
                sample_data=table_schema_dict.get("sample_data")
            )
            
            schema.tables[table_name] = table_schema
        
        return schema
    
    def _log_communication(self, sender: str, receiver: str, message: str) -> None:
        """
        Log inter-agent communication.
        
        Args:
            sender: Agent sending the message
            receiver: Agent receiving the message
            message: Communication message
            
        **Validates: Requirement 3.7**
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "sender": sender,
            "receiver": receiver,
            "message": message
        }
        
        self.communication_log.append(log_entry)
        
        logger.info(f"[{sender} -> {receiver}] {message}")
    
    def _reformulate_query(
        self,
        original_query: str,
        validation_result: ValidationResult,
        structured_query: StructuredQuery
    ) -> str:
        """
        Request QueryAgent to reformulate query based on validation failures.
        
        Args:
            original_query: Original user query
            validation_result: Validation result with issues
            structured_query: Previous structured query
            
        Returns:
            Reformulated query string
            
        **Validates: Requirement 3.5**
        """
        # Create a prompt for reformulation
        issues_summary = "\n".join(validation_result.issues[:3])  # Top 3 issues
        
        reformulation_prompt = f"""The following query failed validation. Please reformulate it to address the issues.

Original Query: {original_query}

Previous Structured Query:
Type: {structured_query.operation_type}
Operation: {structured_query.operation}

Validation Issues:
{issues_summary}

Please provide a reformulated query that addresses these issues."""
        
        try:
            # Use LLM to reformulate
            response = self.llm_provider.generate_with_cache(
                prompt=reformulation_prompt,
                temperature=0.5
            )
            
            reformulated = response.content.strip()
            logger.info(f"Reformulated query: {reformulated}")
            
            return reformulated
            
        except Exception as e:
            logger.error(f"Failed to reformulate query: {str(e)}")
            # Return original query if reformulation fails
            return original_query
    
    def _format_response(
        self,
        user_query: str,
        query_result: QueryResult,
        validation_result: ValidationResult,
        execution_time: float
    ) -> Response:
        """
        Format final response using LLM.
        
        Converts query results into natural language response with data evidence.
        
        Args:
            user_query: Original user query
            query_result: Query execution results
            validation_result: Validation results
            execution_time: Total execution time
            
        Returns:
            Response object with formatted answer
            
        **Validates: Requirement 7.2**
        """
        logger.info("Formatting response using LLM")
        
        try:
            # Prepare data summary for LLM
            data_summary = self._create_data_summary(query_result)
            
            # Include validation warnings if any
            warnings = []
            if validation_result.anomalies:
                warnings = [
                    f"{a.type}: {a.description}"
                    for a in validation_result.anomalies[:3]  # Top 3 anomalies
                ]
            
            # Format prompt
            prompt = PromptTemplates.format_response_formatting_prompt(
                user_query=user_query,
                data_summary=data_summary,
                warnings=warnings
            )
            
            # Generate natural language response
            llm_response = self.llm_provider.generate_with_cache(
                prompt=prompt,
                temperature=0.7
            )
            
            # Create Response object
            response = Response(
                answer=llm_response.content,
                data=query_result.data if query_result.row_count <= 100 else query_result.data.head(100),
                metadata={
                    "execution_time": execution_time,
                    "tokens_used": llm_response.tokens_used,
                    "agents_involved": ["QueryAgent", "ExtractionAgent", "ValidationAgent"],
                    "row_count": query_result.row_count,
                    "cached": query_result.cached,
                    "validation_confidence": validation_result.confidence,
                    "warnings": warnings
                }
            )
            
            logger.info(f"Response formatted successfully in {execution_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to format response: {str(e)}")
            # Fallback to basic response
            return Response(
                answer=f"Query executed successfully. Found {query_result.row_count} rows.",
                data=query_result.data,
                metadata={
                    "execution_time": execution_time,
                    "error": f"Response formatting failed: {str(e)}"
                }
            )
    
    def _create_data_summary(self, query_result: QueryResult) -> str:
        """
        Create a summary of query results for LLM prompt.
        
        Args:
            query_result: Query execution results
            
        Returns:
            Data summary string
        """
        df = query_result.data
        
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"Row count: {query_result.row_count}")
        summary_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Sample data (first 5 rows)
        if not df.empty:
            summary_parts.append("\nSample data (first 5 rows):")
            summary_parts.append(df.head(5).to_string())
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary_parts.append("\nNumeric column statistics:")
            for col in numeric_cols[:5]:  # Limit to 5 columns
                summary_parts.append(
                    f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, "
                    f"mean={df[col].mean():.2f}"
                )
        
        return "\n".join(summary_parts)
    
    def _create_error_response(
        self,
        user_query: str,
        validation_result: ValidationResult,
        execution_time: float
    ) -> Response:
        """
        Create error response when validation fails after max retries.
        
        Args:
            user_query: Original user query
            validation_result: Final validation result
            execution_time: Total execution time
            
        Returns:
            Response object with error message
        """
        # Summarize validation issues
        issues_summary = "\n".join([
            f"- {issue}"
            for issue in validation_result.issues[:5]  # Top 5 issues
        ])
        
        error_message = f"""I was unable to process your query successfully after multiple attempts.

Your query: {user_query}

Validation issues encountered:
{issues_summary}

Please try rephrasing your query or check if the data contains the information you're looking for."""
        
        return Response(
            answer=error_message,
            metadata={
                "execution_time": execution_time,
                "validation_failed": True,
                "issues": validation_result.issues,
                "anomalies": [
                    {"type": a.type, "description": a.description, "severity": a.severity}
                    for a in validation_result.anomalies
                ]
            }
        )
    
    def reset_context(self) -> None:
        """
        Clear conversation context.
        
        This method resets the conversation history and context manager.
        """
        self.context_manager.clear()
        logger.info("Conversation context reset")
    
    def get_conversation_history(self) -> List[Message]:
        """
        Retrieve conversation history.
        
        Returns:
            List of Message objects from conversation
        """
        return self.context_manager.context_window.messages
    
    def get_communication_log(self) -> List[dict]:
        """
        Get inter-agent communication log.
        
        Returns:
            List of communication log entries
            
        **Validates: Requirement 3.7**
        """
        return self.communication_log
    
    def clear_communication_log(self) -> None:
        """Clear the communication log."""
        self.communication_log.clear()
        logger.info("Communication log cleared")
