"""Data models for Retail Insights Assistant.

This module defines the core data structures used throughout the application
for agent communication, query processing, and conversation management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class Message:
    """
    Represents a single message in a conversation.
    
    Attributes:
        role: The role of the message sender ("user", "assistant", "system")
        content: The text content of the message
        timestamp: When the message was created
        metadata: Additional metadata about the message (e.g., tokens used, agent info)
    """
    role: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredQuery:
    """
    Represents a parsed natural language query as a structured data operation.
    
    Attributes:
        operation_type: Type of operation ("sql", "pandas", "semantic")
        operation: The actual query/operation string (SQL query, Pandas code, or search query)
        explanation: Human-readable explanation of what the operation does
        parameters: Additional parameters for the operation
    """
    operation_type: str
    operation: str
    explanation: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """
    Represents the result of executing a structured query.
    
    Attributes:
        data: The result data as a Pandas DataFrame
        row_count: Number of rows in the result
        execution_time: Time taken to execute the query in seconds
        query: The original structured query that was executed
        cached: Whether the result was retrieved from cache
    """
    data: pd.DataFrame
    row_count: int
    execution_time: float
    query: StructuredQuery
    cached: bool = False


@dataclass
class Anomaly:
    """
    Represents a data anomaly detected during validation.
    
    Attributes:
        type: Type of anomaly ("negative_value", "missing_data", "outlier", etc.)
        description: Human-readable description of the anomaly
        severity: Severity level ("warning", "error")
        affected_rows: List of row indices affected by the anomaly
    """
    type: str
    description: str
    severity: str
    affected_rows: List[int] = field(default_factory=list)


@dataclass
class ValidationResult:
    """
    Represents the result of validating query results.
    
    Attributes:
        passed: Whether validation passed
        issues: List of validation issues found
        anomalies: List of data anomalies detected
        confidence: Confidence score for the validation (0.0 to 1.0)
    """
    passed: bool
    issues: List[str] = field(default_factory=list)
    anomalies: List[Anomaly] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class Response:
    """
    Represents the final response to a user query.
    
    Attributes:
        answer: Natural language answer to the user's query
        data: Optional DataFrame containing result data
        visualizations: List of visualization specifications
        metadata: Additional metadata (execution_time, tokens_used, agents_involved, etc.)
    """
    answer: str
    data: Optional[pd.DataFrame] = None
    visualizations: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ColumnInfo:
    """
    Represents information about a single column in a table.
    
    Attributes:
        name: Column name
        dtype: Data type of the column
        nullable: Whether the column contains null values
        unique_count: Number of unique values in the column
        sample_values: Sample values from the column
    """
    name: str
    dtype: str
    nullable: bool
    unique_count: int
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class TableSchema:
    """
    Represents the schema of a single table.
    
    Attributes:
        name: Table name
        columns: List of column information
        row_count: Number of rows in the table
        sample_data: Sample rows from the table
    """
    name: str
    columns: List[ColumnInfo]
    row_count: int
    sample_data: pd.DataFrame


@dataclass
class DataSchema:
    """
    Represents the complete schema of all loaded datasets.
    
    Attributes:
        tables: Dictionary mapping table names to their schemas
    """
    tables: Dict[str, TableSchema] = field(default_factory=dict)


