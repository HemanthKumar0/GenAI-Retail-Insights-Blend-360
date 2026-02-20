"""
Orchestrator module using LangGraph for multi-agent coordination.

This module implements the Orchestrator using LangGraph's StateGraph to coordinate
communication between QueryAgent, ExtractionAgent, and ValidationAgent.
"""

import logging
import time
from typing import Optional, List, TypedDict, Annotated
from datetime import datetime

from langgraph.graph import StateGraph, END

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


# ---------------------------------------------------------------------------
# LangGraph state definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """Shared state flowing through the LangGraph pipeline."""
    user_query: str
    current_query: str          # may be reformulated across retries
    mode: str
    schema: DataSchema
    conversation_context: str
    structured_query: StructuredQuery
    query_result: QueryResult
    validation_result: ValidationResult
    attempt: int
    max_retries: int
    start_time: float
    response: Response
    error: str


class Orchestrator:
    """
    Orchestrator coordinates agent communication via a LangGraph StateGraph.

    The graph has the following nodes:
        parse_query  →  execute_query  →  validate_results  →  (check_validation)
                                                                   ├─ PASSED → format_response → END
                                                                   └─ FAILED → reformulate_query → parse_query  (loop)
                                                                   └─ MAX_RETRIES → error_response → END

    This class:
    - Routes queries to QueryAgent for interpretation
    - Forwards structured operations to ExtractionAgent
    - Sends results to ValidationAgent for verification
    - Retries up to max_retries times on validation failures
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

        # Build the LangGraph workflow once
        self._graph = self._build_graph()

        logger.info(
            f"Orchestrator initialized with LangGraph, max_retries={max_retries}"
        )

    # ------------------------------------------------------------------
    # LangGraph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph StateGraph."""

        graph = StateGraph(AgentState)

        # Register nodes
        graph.add_node("parse_query", self._node_parse_query)
        graph.add_node("execute_query", self._node_execute_query)
        graph.add_node("validate_results", self._node_validate_results)
        graph.add_node("format_response", self._node_format_response)
        graph.add_node("reformulate_query", self._node_reformulate_query)
        graph.add_node("error_response", self._node_error_response)

        # Edges
        graph.set_entry_point("parse_query")
        graph.add_edge("parse_query", "execute_query")
        graph.add_edge("execute_query", "validate_results")

        # Conditional edge after validation
        graph.add_conditional_edges(
            "validate_results",
            self._route_after_validation,
            {
                "format_response": "format_response",
                "reformulate_query": "reformulate_query",
                "error_response": "error_response",
            },
        )

        graph.add_edge("reformulate_query", "parse_query")
        graph.add_edge("format_response", END)
        graph.add_edge("error_response", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    @staticmethod
    def _route_after_validation(state: AgentState) -> str:
        """Decide next node after validation."""
        validation = state.get("validation_result")
        attempt = state.get("attempt", 1)
        max_retries = state.get("max_retries", 3)

        if state.get("error"):
            if attempt < max_retries:
                return "reformulate_query"
            return "error_response"

        if validation and validation.passed:
            return "format_response"

        if attempt < max_retries:
            return "reformulate_query"

        return "error_response"

    # ------------------------------------------------------------------
    # Graph node implementations
    # ------------------------------------------------------------------

    def _node_parse_query(self, state: AgentState) -> dict:
        """Node: route query to QueryAgent for interpretation."""
        current_query = state.get("current_query", state["user_query"])
        schema = state["schema"]
        context = state.get("conversation_context", "")

        self._log_communication(
            "Orchestrator", "QueryAgent",
            f"Parse query: {current_query}"
        )

        try:
            structured_query = self.query_agent.parse_query(
                query=current_query,
                schema=schema,
                context=context,
            )
            self._log_communication(
                "QueryAgent", "Orchestrator",
                f"Structured query: {structured_query.operation_type} - {structured_query.explanation}"
            )
            return {"structured_query": structured_query, "error": ""}
        except Exception as e:
            logger.error(f"QueryAgent failed: {e}")
            return {"error": str(e)}

    def _node_execute_query(self, state: AgentState) -> dict:
        """Node: forward structured query to ExtractionAgent."""
        if state.get("error"):
            return {}

        structured_query = state["structured_query"]

        self._log_communication(
            "Orchestrator", "ExtractionAgent",
            f"Execute query: {structured_query.explanation}"
        )

        try:
            query_result = self.extraction_agent.execute_query(structured_query)
            self._log_communication(
                "ExtractionAgent", "Orchestrator",
                f"Query result: {query_result.row_count} rows in {query_result.execution_time:.2f}s"
            )
            return {"query_result": query_result, "error": ""}
        except Exception as e:
            logger.error(f"ExtractionAgent failed: {e}")
            return {"error": str(e)}

    def _node_validate_results(self, state: AgentState) -> dict:
        """Node: send results to ValidationAgent for verification."""
        if state.get("error"):
            return {"attempt": state.get("attempt", 0) + 1}

        query_result = state["query_result"]
        structured_query = state["structured_query"]

        self._log_communication(
            "Orchestrator", "ValidationAgent",
            f"Validate results: {query_result.row_count} rows"
        )

        try:
            validation_result = self.validation_agent.validate_results(
                results=query_result,
                query=structured_query,
            )
            self._log_communication(
                "ValidationAgent", "Orchestrator",
                f"Validation: {'PASSED' if validation_result.passed else 'FAILED'} "
                f"(confidence: {validation_result.confidence:.2f})"
            )
            return {
                "validation_result": validation_result,
                "attempt": state.get("attempt", 0) + 1,
            }
        except Exception as e:
            logger.error(f"ValidationAgent failed: {e}")
            return {"error": str(e), "attempt": state.get("attempt", 0) + 1}

    def _node_format_response(self, state: AgentState) -> dict:
        """Node: format final response using LLM."""
        execution_time = time.time() - state["start_time"]
        response = self._format_response(
            user_query=state["user_query"],
            query_result=state["query_result"],
            validation_result=state["validation_result"],
            execution_time=execution_time,
        )
        return {"response": response}

    def _node_reformulate_query(self, state: AgentState) -> dict:
        """Node: ask QueryAgent to reformulate after validation failure."""
        validation_result = state.get("validation_result")
        structured_query = state.get("structured_query")
        current_query = state.get("current_query", state["user_query"])

        if validation_result and structured_query:
            self._log_communication(
                "Orchestrator", "QueryAgent",
                f"Reformulate query due to validation failure: "
                f"{validation_result.issues[0] if validation_result.issues else 'unknown issue'}"
            )
            reformulated = self._reformulate_query(
                original_query=current_query,
                validation_result=validation_result,
                structured_query=structured_query,
            )
        else:
            reformulated = current_query

        logger.info(f"Reformulated query (attempt {state.get('attempt', 1)}): {reformulated}")
        return {"current_query": reformulated, "error": ""}

    def _node_error_response(self, state: AgentState) -> dict:
        """Node: create error response when max retries exhausted."""
        execution_time = time.time() - state["start_time"]
        error_msg = state.get("error", "")
        validation_result = state.get("validation_result")

        if validation_result:
            response = self._create_error_response(
                user_query=state["user_query"],
                validation_result=validation_result,
                execution_time=execution_time,
            )
        else:
            response = Response(
                answer=f"I encountered an error processing your query: {error_msg}",
                metadata={
                    "error": error_msg,
                    "execution_time": execution_time,
                    "attempts": state.get("attempt", 0),
                },
            )
        return {"response": response}

    # ------------------------------------------------------------------
    # Public API  (unchanged interface for app.py / modes / tests)
    # ------------------------------------------------------------------

    def process_query(self, user_query: str, mode: str = "qa") -> Response:
        """
        Process a user query through the LangGraph agent pipeline.

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
            ValueError: If mode is invalid
        """
        if mode not in ["summarization", "qa"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'summarization' or 'qa'")

        logger.info(f"Processing query in {mode} mode: {user_query}")

        # Add user message to context
        user_message = Message(
            role="user",
            content=user_query,
            timestamp=datetime.now(),
        )
        self.context_manager.add_message(user_message)

        # Prepare initial state for the graph
        schema = self._get_data_schema()
        conversation_context = self.context_manager.get_context(prioritize_recent=True)

        initial_state: AgentState = {
            "user_query": user_query,
            "current_query": user_query,
            "mode": mode,
            "schema": schema,
            "conversation_context": conversation_context,
            "attempt": 0,
            "max_retries": self.max_retries,
            "start_time": time.time(),
            "error": "",
        }

        # Run the LangGraph workflow
        final_state = self._graph.invoke(initial_state)

        response = final_state.get("response")
        if response is None:
            response = Response(
                answer="I was unable to process your query after multiple attempts.",
                metadata={
                    "execution_time": time.time() - initial_state["start_time"],
                    "attempts": self.max_retries,
                },
            )

        # Add assistant response to context
        assistant_message = Message(
            role="assistant",
            content=response.answer,
            timestamp=datetime.now(),
            metadata=response.metadata,
        )
        self.context_manager.add_message(assistant_message)

        return response

    # ------------------------------------------------------------------
    # Helper methods (kept from original implementation)
    # ------------------------------------------------------------------

    def _get_data_schema(self) -> DataSchema:
        """
        Get data schema from extraction agent's data store.

        Returns:
            DataSchema object with table information
        """
        from src.core.models import TableSchema, ColumnInfo

        schema = DataSchema()
        tables = self.extraction_agent.data_store.list_tables()

        for table_name in tables:
            table_schema_dict = self.extraction_agent.data_store.get_table_schema(table_name)
            columns = []
            columns_data = table_schema_dict["columns"]

            if isinstance(columns_data, dict):
                for col_name, col_info in columns_data.items():
                    columns.append(ColumnInfo(
                        name=col_name,
                        dtype=col_info.get("dtype", col_info.get("type", "unknown")),
                        nullable=col_info.get("nullable", False),
                        unique_count=col_info.get("unique_count", 0),
                        sample_values=col_info.get("sample_values", []),
                    ))
            else:
                for col_info in columns_data:
                    columns.append(ColumnInfo(
                        name=col_info["name"],
                        dtype=col_info.get("type", col_info.get("dtype", "unknown")),
                        nullable=col_info.get("nullable", False),
                        unique_count=col_info.get("unique_count", 0),
                        sample_values=col_info.get("sample_values", []),
                    ))

            table_schema = TableSchema(
                name=table_name,
                columns=columns,
                row_count=table_schema_dict["row_count"],
                sample_data=table_schema_dict.get("sample_data"),
            )
            schema.tables[table_name] = table_schema

        return schema

    def _log_communication(self, sender: str, receiver: str, message: str) -> None:
        """
        Log inter-agent communication.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "sender": sender,
            "receiver": receiver,
            "message": message,
        }
        self.communication_log.append(log_entry)
        logger.info(f"[{sender} -> {receiver}] {message}")

    def _reformulate_query(
        self,
        original_query: str,
        validation_result: ValidationResult,
        structured_query: StructuredQuery,
    ) -> str:
        """
        Request QueryAgent to reformulate query based on validation failures.
        """
        issues_summary = "\n".join(validation_result.issues[:3])

        reformulation_prompt = f"""The following query failed validation. Please reformulate it to address the issues.

Original Query: {original_query}

Previous Structured Query:
Type: {structured_query.operation_type}
Operation: {structured_query.operation}

Validation Issues:
{issues_summary}

Please provide a reformulated query that addresses these issues."""

        try:
            response = self.llm_provider.generate_with_cache(
                prompt=reformulation_prompt,
                temperature=0.5,
            )
            reformulated = response.content.strip()
            logger.info(f"Reformulated query: {reformulated}")
            return reformulated
        except Exception as e:
            logger.error(f"Failed to reformulate query: {e}")
            return original_query

    def _format_response(
        self,
        user_query: str,
        query_result: QueryResult,
        validation_result: ValidationResult,
        execution_time: float,
    ) -> Response:
        """
        Format final response using LLM.
        """
        logger.info("Formatting response using LLM")

        try:
            data_summary = self._create_data_summary(query_result)

            warnings = []
            if validation_result.anomalies:
                warnings = [
                    f"{a.type}: {a.description}"
                    for a in validation_result.anomalies[:3]
                ]

            prompt = PromptTemplates.format_response_formatting_prompt(
                user_query=user_query,
                data_summary=data_summary,
                warnings=warnings,
            )

            llm_response = self.llm_provider.generate_with_cache(
                prompt=prompt,
                temperature=0.7,
            )

            return Response(
                answer=llm_response.content,
                data=query_result.data if query_result.row_count <= 100 else query_result.data.head(100),
                metadata={
                    "execution_time": execution_time,
                    "tokens_used": llm_response.tokens_used,
                    "agents_involved": ["QueryAgent", "ExtractionAgent", "ValidationAgent"],
                    "row_count": query_result.row_count,
                    "cached": query_result.cached,
                    "validation_confidence": validation_result.confidence,
                    "warnings": warnings,
                    "orchestration": "langgraph",
                },
            )
        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return Response(
                answer=f"Query executed successfully. Found {query_result.row_count} rows.",
                data=query_result.data,
                metadata={
                    "execution_time": execution_time,
                    "error": f"Response formatting failed: {e}",
                },
            )

    def _create_data_summary(self, query_result: QueryResult) -> str:
        """Create a summary of query results for LLM prompt."""
        df = query_result.data
        summary_parts = []

        summary_parts.append(f"Row count: {query_result.row_count}")
        summary_parts.append(f"Columns: {', '.join(df.columns.tolist())}")

        if not df.empty:
            summary_parts.append("\nSample data (first 5 rows):")
            summary_parts.append(df.head(5).to_string())

        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            summary_parts.append("\nNumeric column statistics:")
            for col in numeric_cols[:5]:
                summary_parts.append(
                    f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, "
                    f"mean={df[col].mean():.2f}"
                )

        return "\n".join(summary_parts)

    def _create_error_response(
        self,
        user_query: str,
        validation_result: ValidationResult,
        execution_time: float,
    ) -> Response:
        """Create error response when validation fails after max retries."""
        issues_summary = "\n".join(
            [f"- {issue}" for issue in validation_result.issues[:5]]
        )

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
                ],
            },
        )

    def reset_context(self) -> None:
        """Clear conversation context."""
        self.context_manager.clear()
        logger.info("Conversation context reset")

    def get_conversation_history(self) -> List[Message]:
        """Retrieve conversation history."""
        return self.context_manager.context_window.messages

    def get_communication_log(self) -> List[dict]:
        """
        Get inter-agent communication log.
        """
        return self.communication_log

    def clear_communication_log(self) -> None:
        """Clear the communication log."""
        self.communication_log.clear()
        logger.info("Communication log cleared")
