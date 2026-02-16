"""Unit tests for data model classes.

**Validates: Requirements 11.1, 11.2**
"""

import pytest
from datetime import datetime
import pandas as pd
from src.models import (
    Message,
    StructuredQuery,
    QueryResult,
    ValidationResult,
    Anomaly,
    Response,
    DataSchema,
    TableSchema,
    ColumnInfo,
    ConversationState
)


class TestMessage:
    """Tests for Message dataclass."""
    
    def test_message_creation_with_required_fields(self):
        """Test creating a message with required fields."""
        timestamp = datetime.now()
        msg = Message(
            role="user",
            content="What were total sales?",
            timestamp=timestamp
        )
        
        assert msg.role == "user"
        assert msg.content == "What were total sales?"
        assert msg.timestamp == timestamp
        assert msg.metadata == {}
    
    def test_message_with_metadata(self):
        """Test creating a message with metadata."""
        timestamp = datetime.now()
        metadata = {"tokens": 10, "agent": "query_agent"}
        msg = Message(
            role="assistant",
            content="Total sales were $1M",
            timestamp=timestamp,
            metadata=metadata
        )
        
        assert msg.metadata == metadata
        assert msg.metadata["tokens"] == 10
    
    def test_message_roles(self):
        """Test different message roles."""
        timestamp = datetime.now()
        
        user_msg = Message("user", "Hello", timestamp)
        assert user_msg.role == "user"
        
        assistant_msg = Message("assistant", "Hi there", timestamp)
        assert assistant_msg.role == "assistant"
        
        system_msg = Message("system", "System initialized", timestamp)
        assert system_msg.role == "system"


class TestStructuredQuery:
    """Tests for StructuredQuery dataclass."""
    
    def test_sql_query_creation(self):
        """Test creating a SQL structured query."""
        query = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM sales WHERE year = 2023",
            explanation="Get all sales from 2023"
        )
        
        assert query.operation_type == "sql"
        assert "SELECT" in query.operation
        assert query.explanation == "Get all sales from 2023"
        assert query.parameters == {}
    
    def test_pandas_query_creation(self):
        """Test creating a Pandas structured query."""
        query = StructuredQuery(
            operation_type="pandas",
            operation="df.groupby('category')['sales'].sum()",
            explanation="Sum sales by category",
            parameters={"column": "sales"}
        )
        
        assert query.operation_type == "pandas"
        assert query.parameters["column"] == "sales"
    
    def test_semantic_query_creation(self):
        """Test creating a semantic search query."""
        query = StructuredQuery(
            operation_type="semantic",
            operation="products with high customer satisfaction",
            explanation="Find products with positive reviews"
        )
        
        assert query.operation_type == "semantic"


class TestQueryResult:
    """Tests for QueryResult dataclass."""
    
    def test_query_result_creation(self):
        """Test creating a query result."""
        df = pd.DataFrame({"sales": [100, 200, 300]})
        query = StructuredQuery("sql", "SELECT * FROM sales", "Get sales")
        
        result = QueryResult(
            data=df,
            row_count=3,
            execution_time=0.5,
            query=query
        )
        
        assert len(result.data) == 3
        assert result.row_count == 3
        assert result.execution_time == 0.5
        assert result.cached is False
    
    def test_cached_query_result(self):
        """Test query result from cache."""
        df = pd.DataFrame({"sales": [100]})
        query = StructuredQuery("sql", "SELECT * FROM sales", "Get sales")
        
        result = QueryResult(
            data=df,
            row_count=1,
            execution_time=0.01,
            query=query,
            cached=True
        )
        
        assert result.cached is True


class TestAnomaly:
    """Tests for Anomaly dataclass."""
    
    def test_anomaly_creation(self):
        """Test creating an anomaly."""
        anomaly = Anomaly(
            type="negative_value",
            description="Sales value is negative",
            severity="error",
            affected_rows=[5, 10, 15]
        )
        
        assert anomaly.type == "negative_value"
        assert anomaly.severity == "error"
        assert len(anomaly.affected_rows) == 3
    
    def test_anomaly_without_affected_rows(self):
        """Test anomaly without specific affected rows."""
        anomaly = Anomaly(
            type="missing_data",
            description="Some columns have missing values",
            severity="warning"
        )
        
        assert anomaly.affected_rows == []


class TestValidationResult:
    """Tests for ValidationResult dataclass."""
    
    def test_validation_passed(self):
        """Test successful validation."""
        result = ValidationResult(passed=True)
        
        assert result.passed is True
        assert result.issues == []
        assert result.anomalies == []
        assert result.confidence == 1.0
    
    def test_validation_failed_with_issues(self):
        """Test failed validation with issues."""
        anomaly = Anomaly("outlier", "Value exceeds expected range", "warning")
        result = ValidationResult(
            passed=False,
            issues=["Data type mismatch", "Missing required field"],
            anomalies=[anomaly],
            confidence=0.7
        )
        
        assert result.passed is False
        assert len(result.issues) == 2
        assert len(result.anomalies) == 1
        assert result.confidence == 0.7


class TestResponse:
    """Tests for Response dataclass."""
    
    def test_response_with_text_only(self):
        """Test response with only text answer."""
        response = Response(answer="Total sales were $1,000,000")
        
        assert response.answer == "Total sales were $1,000,000"
        assert response.data is None
        assert response.visualizations == []
        assert response.metadata == {}
    
    def test_response_with_data(self):
        """Test response with data and metadata."""
        df = pd.DataFrame({"category": ["A", "B"], "sales": [100, 200]})
        response = Response(
            answer="Here are the sales by category",
            data=df,
            metadata={"execution_time": 1.5, "tokens_used": 50}
        )
        
        assert len(response.data) == 2
        assert response.metadata["execution_time"] == 1.5
    
    def test_response_with_visualizations(self):
        """Test response with visualization specs."""
        viz = [{"type": "bar", "x": "category", "y": "sales"}]
        response = Response(
            answer="Sales by category",
            visualizations=viz
        )
        
        assert len(response.visualizations) == 1
        assert response.visualizations[0]["type"] == "bar"


class TestColumnInfo:
    """Tests for ColumnInfo dataclass."""
    
    def test_column_info_creation(self):
        """Test creating column info."""
        col = ColumnInfo(
            name="sales",
            dtype="float64",
            nullable=False,
            unique_count=100,
            sample_values=[100.0, 200.0, 300.0]
        )
        
        assert col.name == "sales"
        assert col.dtype == "float64"
        assert col.nullable is False
        assert col.unique_count == 100
        assert len(col.sample_values) == 3
    
    def test_nullable_column(self):
        """Test column with null values."""
        col = ColumnInfo(
            name="description",
            dtype="object",
            nullable=True,
            unique_count=50
        )
        
        assert col.nullable is True
        assert col.sample_values == []


class TestTableSchema:
    """Tests for TableSchema dataclass."""
    
    def test_table_schema_creation(self):
        """Test creating table schema."""
        col1 = ColumnInfo("id", "int64", False, 100, [1, 2, 3])
        col2 = ColumnInfo("name", "object", False, 100, ["A", "B", "C"])
        sample_df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        
        schema = TableSchema(
            name="products",
            columns=[col1, col2],
            row_count=100,
            sample_data=sample_df
        )
        
        assert schema.name == "products"
        assert len(schema.columns) == 2
        assert schema.row_count == 100
        assert len(schema.sample_data) == 2


class TestDataSchema:
    """Tests for DataSchema dataclass."""
    
    def test_empty_data_schema(self):
        """Test creating empty data schema."""
        schema = DataSchema()
        
        assert schema.tables == {}
    
    def test_data_schema_with_tables(self):
        """Test data schema with multiple tables."""
        col = ColumnInfo("id", "int64", False, 10, [1, 2])
        sample_df = pd.DataFrame({"id": [1, 2]})
        table1 = TableSchema("sales", [col], 10, sample_df)
        table2 = TableSchema("products", [col], 20, sample_df)
        
        schema = DataSchema(tables={"sales": table1, "products": table2})
        
        assert len(schema.tables) == 2
        assert "sales" in schema.tables
        assert "products" in schema.tables
        assert schema.tables["sales"].row_count == 10


class TestConversationState:
    """Tests for ConversationState dataclass."""
    
    def test_empty_conversation_state(self):
        """Test creating empty conversation state."""
        state = ConversationState()
        
        assert state.messages == []
        assert state.current_mode == "qa"
        assert state.loaded_datasets == []
        assert state.active_table is None
        assert state.user_preferences == {}
    
    def test_conversation_state_with_messages(self):
        """Test conversation state with messages."""
        msg1 = Message("user", "Hello", datetime.now())
        msg2 = Message("assistant", "Hi", datetime.now())
        
        state = ConversationState(
            messages=[msg1, msg2],
            current_mode="summarization",
            loaded_datasets=["sales.csv"],
            active_table="sales"
        )
        
        assert len(state.messages) == 2
        assert state.current_mode == "summarization"
        assert state.loaded_datasets == ["sales.csv"]
        assert state.active_table == "sales"
    
    def test_conversation_state_with_preferences(self):
        """Test conversation state with user preferences."""
        state = ConversationState(
            user_preferences={"language": "en", "timezone": "UTC"}
        )
        
        assert state.user_preferences["language"] == "en"
        assert state.user_preferences["timezone"] == "UTC"


class TestDataModelSerialization:
    """Tests for dataclass serialization and deserialization."""
    
    def test_message_to_dict(self):
        """Test converting Message to dictionary."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        msg = Message(
            role="user",
            content="Test message",
            timestamp=timestamp,
            metadata={"key": "value"}
        )
        
        # Convert to dict using dataclass fields
        msg_dict = {
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat(),
            "metadata": msg.metadata
        }
        
        assert msg_dict["role"] == "user"
        assert msg_dict["content"] == "Test message"
        assert msg_dict["metadata"]["key"] == "value"
    
    def test_structured_query_to_dict(self):
        """Test converting StructuredQuery to dictionary."""
        query = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM sales",
            explanation="Get all sales",
            parameters={"limit": 100}
        )
        
        query_dict = {
            "operation_type": query.operation_type,
            "operation": query.operation,
            "explanation": query.explanation,
            "parameters": query.parameters
        }
        
        assert query_dict["operation_type"] == "sql"
        assert query_dict["parameters"]["limit"] == 100
    
    def test_validation_result_serialization(self):
        """Test ValidationResult can be serialized."""
        anomaly = Anomaly("outlier", "High value", "warning", [1, 2])
        result = ValidationResult(
            passed=False,
            issues=["Issue 1", "Issue 2"],
            anomalies=[anomaly],
            confidence=0.8
        )
        
        # Verify all fields are accessible for serialization
        assert result.passed is False
        assert len(result.issues) == 2
        assert len(result.anomalies) == 1
        assert result.confidence == 0.8


class TestDefaultValuesAndFactories:
    """Tests for default values and field factories."""
    
    def test_message_default_metadata(self):
        """Test Message default metadata is empty dict."""
        msg1 = Message("user", "Hello", datetime.now())
        msg2 = Message("user", "World", datetime.now())
        
        # Each instance should have its own metadata dict
        msg1.metadata["key"] = "value1"
        msg2.metadata["key"] = "value2"
        
        assert msg1.metadata["key"] == "value1"
        assert msg2.metadata["key"] == "value2"
    
    def test_structured_query_default_parameters(self):
        """Test StructuredQuery default parameters is empty dict."""
        query1 = StructuredQuery("sql", "SELECT 1", "Test")
        query2 = StructuredQuery("sql", "SELECT 2", "Test")
        
        query1.parameters["a"] = 1
        query2.parameters["b"] = 2
        
        assert "a" in query1.parameters
        assert "b" in query2.parameters
        assert "a" not in query2.parameters
    
    def test_query_result_default_cached(self):
        """Test QueryResult default cached value is False."""
        df = pd.DataFrame({"x": [1]})
        query = StructuredQuery("sql", "SELECT 1", "Test")
        result = QueryResult(df, 1, 0.1, query)
        
        assert result.cached is False
    
    def test_anomaly_default_affected_rows(self):
        """Test Anomaly default affected_rows is empty list."""
        anomaly1 = Anomaly("type1", "desc1", "warning")
        anomaly2 = Anomaly("type2", "desc2", "error")
        
        anomaly1.affected_rows.append(1)
        anomaly2.affected_rows.append(2)
        
        assert 1 in anomaly1.affected_rows
        assert 2 in anomaly2.affected_rows
        assert 1 not in anomaly2.affected_rows
    
    def test_validation_result_defaults(self):
        """Test ValidationResult default values."""
        result = ValidationResult(passed=True)
        
        assert result.issues == []
        assert result.anomalies == []
        assert result.confidence == 1.0
    
    def test_response_defaults(self):
        """Test Response default values."""
        response = Response(answer="Test answer")
        
        assert response.data is None
        assert response.visualizations == []
        assert response.metadata == {}
    
    def test_column_info_default_sample_values(self):
        """Test ColumnInfo default sample_values is empty list."""
        col = ColumnInfo("id", "int64", False, 10)
        
        assert col.sample_values == []
    
    def test_data_schema_default_tables(self):
        """Test DataSchema default tables is empty dict."""
        schema = DataSchema()
        
        assert schema.tables == {}
        assert isinstance(schema.tables, dict)
    
    def test_conversation_state_defaults(self):
        """Test ConversationState default values."""
        state = ConversationState()
        
        assert state.messages == []
        assert state.current_mode == "qa"
        assert state.loaded_datasets == []
        assert state.active_table is None
        assert state.user_preferences == {}


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_dataframe_in_query_result(self):
        """Test QueryResult with empty DataFrame."""
        df = pd.DataFrame()
        query = StructuredQuery("sql", "SELECT * FROM empty_table", "Get empty")
        result = QueryResult(df, 0, 0.1, query)
        
        assert len(result.data) == 0
        assert result.row_count == 0
    
    def test_response_with_empty_answer(self):
        """Test Response with empty answer string."""
        response = Response(answer="")
        
        assert response.answer == ""
        assert response.data is None
    
    def test_validation_result_with_zero_confidence(self):
        """Test ValidationResult with zero confidence."""
        result = ValidationResult(passed=False, confidence=0.0)
        
        assert result.confidence == 0.0
        assert result.passed is False
    
    def test_conversation_state_mode_values(self):
        """Test ConversationState with different mode values."""
        qa_state = ConversationState(current_mode="qa")
        summary_state = ConversationState(current_mode="summarization")
        
        assert qa_state.current_mode == "qa"
        assert summary_state.current_mode == "summarization"
    
    def test_large_metadata_dict(self):
        """Test Message with large metadata dictionary."""
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(100)}
        msg = Message("system", "Test", datetime.now(), metadata=large_metadata)
        
        assert len(msg.metadata) == 100
        assert msg.metadata["key_50"] == "value_50"


class TestDataModelIntegration:
    """Integration tests for data models working together."""
    
    def test_complete_query_flow(self):
        """Test complete flow from query to response."""
        # Create structured query
        query = StructuredQuery(
            operation_type="sql",
            operation="SELECT category, SUM(sales) FROM sales GROUP BY category",
            explanation="Sum sales by category"
        )
        
        # Create query result
        df = pd.DataFrame({
            "category": ["Electronics", "Clothing"],
            "sales": [5000, 3000]
        })
        result = QueryResult(
            data=df,
            row_count=2,
            execution_time=0.3,
            query=query
        )
        
        # Create validation result
        validation = ValidationResult(passed=True, confidence=1.0)
        
        # Create response
        response = Response(
            answer="Sales by category: Electronics $5000, Clothing $3000",
            data=df,
            metadata={
                "execution_time": 0.3,
                "validation_passed": validation.passed
            }
        )
        
        assert response.data is not None
        assert len(response.data) == 2
        assert response.metadata["validation_passed"] is True
    
    def test_conversation_with_schema(self):
        """Test conversation state with data schema."""
        # Create data schema
        col = ColumnInfo("sales", "float64", False, 100, [100.0, 200.0])
        sample_df = pd.DataFrame({"sales": [100.0, 200.0]})
        table = TableSchema("sales", [col], 100, sample_df)
        schema = DataSchema(tables={"sales": table})
        
        # Create conversation state
        msg = Message("user", "Show me sales data", datetime.now())
        state = ConversationState(
            messages=[msg],
            loaded_datasets=["sales"],
            active_table="sales"
        )
        
        assert state.active_table in schema.tables
        assert len(state.messages) == 1
