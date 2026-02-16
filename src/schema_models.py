"""
Pydantic models for automatic schema detection and validation.
Handles edge cases and provides robust type conversion for any uploaded files.
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum
import pandas as pd
import re
from datetime import datetime


class ColumnDataType(str, Enum):
    """Enumeration of supported column data types."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    UNKNOWN = "unknown"


class ColumnMetadata(BaseModel):
    """Metadata for a single column in a dataset."""
    
    name: str = Field(..., description="Original column name")
    sanitized_name: str = Field(..., description="Sanitized column name for SQL")
    detected_type: ColumnDataType = Field(..., description="Automatically detected data type")
    original_type: str = Field(..., description="Original pandas dtype")
    is_nullable: bool = Field(default=True, description="Whether column contains null values")
    unique_count: int = Field(..., description="Number of unique values")
    sample_values: List[Any] = Field(default_factory=list, description="Sample values from column")
    
    # Type-specific metadata
    is_numeric: bool = Field(default=False, description="Whether column is numeric")
    is_text_stored_numeric: bool = Field(default=False, description="Whether numeric values are stored as text")
    is_categorical: bool = Field(default=False, description="Whether column is categorical")
    is_temporal: bool = Field(default=False, description="Whether column is date/time")
    
    # Statistical metadata for numeric columns
    min_value: Optional[float] = Field(default=None, description="Minimum value for numeric columns")
    max_value: Optional[float] = Field(default=None, description="Maximum value for numeric columns")
    mean_value: Optional[float] = Field(default=None, description="Mean value for numeric columns")
    
    # Conversion metadata
    requires_casting: bool = Field(default=False, description="Whether column needs type casting in SQL")
    cast_expression: Optional[str] = Field(default=None, description="SQL expression for type casting")
    
    class Config:
        use_enum_values = True
    
    @validator('sanitized_name', pre=True, always=True)
    def sanitize_column_name(cls, v, values):
        """Sanitize column name for SQL compatibility."""
        if 'name' in values:
            name = values['name']
            # Replace special characters with underscores
            sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
            # Remove leading/trailing underscores
            sanitized = sanitized.strip('_')
            # Ensure it doesn't start with a number
            if sanitized and sanitized[0].isdigit():
                sanitized = f"col_{sanitized}"
            return sanitized or "unnamed_column"
        return v
    
    def get_sql_column_reference(self) -> str:
        """Get the proper SQL column reference with quotes if needed."""
        # If original name has spaces or special chars, use double quotes
        if ' ' in self.name or any(c in self.name for c in ['-', '.', '/', '(', ')', '#']):
            return f'"{self.name}"'
        return self.name
    
    def get_sql_cast_expression(self, alias: Optional[str] = None) -> str:
        """Get SQL expression with casting if needed."""
        col_ref = self.get_sql_column_reference()
        
        if self.requires_casting:
            if self.cast_expression:
                expr = self.cast_expression.replace("{column}", col_ref)
            else:
                # Default casting based on detected type
                if self.detected_type in [ColumnDataType.INTEGER, ColumnDataType.FLOAT, 
                                         ColumnDataType.CURRENCY, ColumnDataType.PERCENTAGE]:
                    expr = f"TRY_CAST({col_ref} AS DOUBLE)"
                else:
                    expr = col_ref
        else:
            expr = col_ref
        
        if alias:
            expr = f"{expr} as {alias}"
        
        return expr


class TableSchema(BaseModel):
    """Schema for a complete table/dataset."""
    
    table_name: str = Field(..., description="Name of the table")
    original_filename: str = Field(..., description="Original filename")
    row_count: int = Field(..., description="Number of rows")
    column_count: int = Field(..., description="Number of columns")
    columns: List[ColumnMetadata] = Field(default_factory=list, description="Column metadata")
    
    # Table-level metadata
    has_text_stored_numerics: bool = Field(default=False, description="Whether table has text-stored numeric columns")
    has_date_columns: bool = Field(default=False, description="Whether table has date columns")
    has_categorical_columns: bool = Field(default=False, description="Whether table has categorical columns")
    
    # Suggested operations
    suggested_group_by_columns: List[str] = Field(default_factory=list, description="Columns suitable for GROUP BY")
    suggested_aggregate_columns: List[str] = Field(default_factory=list, description="Columns suitable for aggregation")
    suggested_filter_columns: List[str] = Field(default_factory=list, description="Columns suitable for filtering")
    
    # Data quality indicators
    data_quality_score: float = Field(default=1.0, description="Overall data quality score (0-1)")
    data_quality_issues: List[str] = Field(default_factory=list, description="List of data quality issues")
    
    @root_validator
    def compute_table_metadata(cls, values):
        """Compute table-level metadata from columns."""
        columns = values.get('columns', [])
        
        # Check for text-stored numerics
        values['has_text_stored_numerics'] = any(col.is_text_stored_numeric for col in columns)
        
        # Check for date columns
        values['has_date_columns'] = any(col.is_temporal for col in columns)
        
        # Check for categorical columns
        values['has_categorical_columns'] = any(col.is_categorical for col in columns)
        
        # Suggest columns for different operations
        values['suggested_group_by_columns'] = [
            col.name for col in columns 
            if col.is_categorical or (col.unique_count < values.get('row_count', 0) * 0.1)
        ]
        
        values['suggested_aggregate_columns'] = [
            col.name for col in columns 
            if col.is_numeric or col.is_text_stored_numeric
        ]
        
        values['suggested_filter_columns'] = [
            col.name for col in columns 
            if col.is_categorical or col.is_temporal
        ]
        
        # Compute data quality score
        issues = []
        quality_score = 1.0
        
        if values['has_text_stored_numerics']:
            issues.append("Some numeric columns are stored as text")
            quality_score -= 0.1
        
        # Check for high null percentage
        high_null_cols = [col.name for col in columns if col.is_nullable and col.unique_count == 0]
        if high_null_cols:
            issues.append(f"Columns with all null values: {', '.join(high_null_cols[:3])}")
            quality_score -= 0.2
        
        values['data_quality_issues'] = issues
        values['data_quality_score'] = max(0.0, quality_score)
        
        return values
    
    def get_column_by_name(self, name: str) -> Optional[ColumnMetadata]:
        """Get column metadata by name."""
        for col in self.columns:
            if col.name == name or col.sanitized_name == name:
                return col
        return None
    
    def get_numeric_columns(self) -> List[ColumnMetadata]:
        """Get all numeric columns (including text-stored)."""
        return [col for col in self.columns if col.is_numeric or col.is_text_stored_numeric]
    
    def get_categorical_columns(self) -> List[ColumnMetadata]:
        """Get all categorical columns."""
        return [col for col in self.columns if col.is_categorical]
    
    def get_temporal_columns(self) -> List[ColumnMetadata]:
        """Get all temporal columns."""
        return [col for col in self.columns if col.is_temporal]
    
    def to_llm_schema_description(self) -> str:
        """Generate a human-readable schema description for LLM."""
        lines = [f"Table: {self.table_name}"]
        lines.append(f"Rows: {self.row_count:,}")
        lines.append(f"Columns: {self.column_count}")
        lines.append("")
        lines.append("Column Details:")
        
        for col in self.columns:
            col_desc = f"  - {col.name} ({col.detected_type.value})"
            
            if col.is_text_stored_numeric:
                col_desc += " [stored as text, requires casting]"
            
            if col.is_categorical:
                col_desc += f" [categorical, {col.unique_count} unique values]"
            
            if col.is_temporal:
                col_desc += " [date/time]"
            
            lines.append(col_desc)
        
        if self.suggested_group_by_columns:
            lines.append("")
            lines.append(f"Suggested GROUP BY columns: {', '.join(self.suggested_group_by_columns[:5])}")
        
        if self.suggested_aggregate_columns:
            lines.append(f"Suggested aggregate columns: {', '.join(self.suggested_aggregate_columns[:5])}")
        
        if self.data_quality_issues:
            lines.append("")
            lines.append("Data Quality Notes:")
            for issue in self.data_quality_issues:
                lines.append(f"  ⚠ {issue}")
        
        return "\n".join(lines)


class DatasetCollection(BaseModel):
    """Collection of all loaded datasets."""
    
    tables: Dict[str, TableSchema] = Field(default_factory=dict, description="Map of table name to schema")
    total_tables: int = Field(default=0, description="Total number of tables")
    total_rows: int = Field(default=0, description="Total rows across all tables")
    
    @root_validator
    def compute_totals(cls, values):
        """Compute total statistics."""
        tables = values.get('tables', {})
        values['total_tables'] = len(tables)
        values['total_rows'] = sum(schema.row_count for schema in tables.values())
        return values
    
    def add_table(self, schema: TableSchema):
        """Add a table schema to the collection."""
        self.tables[schema.table_name] = schema
        self.total_tables = len(self.tables)
        self.total_rows = sum(s.row_count for s in self.tables.values())
    
    def get_table(self, table_name: str) -> Optional[TableSchema]:
        """Get table schema by name."""
        return self.tables.get(table_name)
    
    def to_llm_schema_description(self) -> str:
        """Generate complete schema description for LLM."""
        lines = [f"Available Datasets: {self.total_tables}"]
        lines.append(f"Total Rows: {self.total_rows:,}")
        lines.append("")
        
        for table_name, schema in self.tables.items():
            lines.append(schema.to_llm_schema_description())
            lines.append("")
        
        return "\n".join(lines)


class QueryContext(BaseModel):
    """Context for query execution with schema information."""
    
    available_tables: List[str] = Field(default_factory=list, description="List of available table names")
    schemas: Dict[str, TableSchema] = Field(default_factory=dict, description="Table schemas")
    
    def get_relevant_tables(self, keywords: List[str]) -> List[str]:
        """Get tables relevant to given keywords."""
        relevant = []
        
        for table_name, schema in self.schemas.items():
            # Check if any keyword matches table name
            if any(kw.lower() in table_name.lower() for kw in keywords):
                relevant.append(table_name)
                continue
            
            # Check if any keyword matches column names
            for col in schema.columns:
                if any(kw.lower() in col.name.lower() for kw in keywords):
                    relevant.append(table_name)
                    break
        
        return relevant or self.available_tables  # Return all if no match
    
    def get_schema_for_llm(self, table_names: Optional[List[str]] = None) -> str:
        """Get schema description for LLM, optionally filtered by table names."""
        if table_names:
            schemas = [self.schemas[name] for name in table_names if name in self.schemas]
        else:
            schemas = list(self.schemas.values())
        
        lines = []
        for schema in schemas:
            lines.append(schema.to_llm_schema_description())
            lines.append("")
        
        return "\n".join(lines)
