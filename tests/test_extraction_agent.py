"""Unit tests for ExtractionAgent class.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
"""

import pytest
import pandas as pd
import time
from src.extraction_agent import ExtractionAgent
from src.data_store import DataStore
from src.models import StructuredQuery


class TestExtractionAgentInitialization:
    """Tests for ExtractionAgent initialization."""
    
    def test_initialization_with_data_store(self):
        """Test that ExtractionAgent initializes with a DataStore."""
        store = DataStore()
        agent = ExtractionAgent(store)
        
        assert agent.data_store is store
        assert agent.query_timeout > 0
        assert agent.max_rows_display > 0
        assert agent.cache_size > 0
        assert isinstance(agent._query_cache, dict)
        assert isinstance(agent._cache_order, list)
        
        store.close()
    
    def test_initialization_uses_config_values(self):
        """Test that ExtractionAgent uses configuration values."""
        from src.config import Config
        
        store = DataStore()
        agent = ExtractionAgent(store)
        
        assert agent.query_timeout == Config.QUERY_TIMEOUT
        assert agent.max_rows_display == Config.MAX_ROWS_DISPLAY
        assert agent.cache_size == Config.CACHE_SIZE
        
        store.close()


class TestExecuteQuerySQL:
    """Tests for execute_query with SQL operations."""
    
    def test_execute_simple_sql_query(self):
        """Test executing a simple SQL query."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        store.register_dataframe("test", df)
        
        agent = ExtractionAgent(store)
        query = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM test",
            explanation="Get all rows from test table"
        )
        
        result = agent.execute_query(query)
        
        assert result.row_count == 3
        assert len(result.data) == 3
        assert list(result.data.columns) == ["a", "b"]
        assert result.execution_time > 0
        assert result.cached == False
        assert result.query == query
        
        store.close()
    
    def test_execute_sql_with_where_clause(self):
        """Test executing SQL with WHERE clause."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        store.register_dataframe("test", df)
        
        agent = ExtractionAgent(store)
        query = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM test WHERE a > 2",
            explanation="Filter rows where a > 2"
        )
        
        result = agent.execute_query(query)
        
        assert result.row_count == 3
        assert result.data["a"].tolist() == [3, 4, 5]
        
        store.close()
    
    def test_execute_sql_aggregation(self):
        """Test executing SQL aggregation query."""
        store = DataStore()
        df = pd.DataFrame({
            "category": ["A", "A", "B", "B"],
            "value": [10, 20, 30, 40]
        })
        store.register_dataframe("sales", df)
        
        agent = ExtractionAgent(store)
        query = StructuredQuery(
            operation_type="sql",
            operation="SELECT category, SUM(value) as total FROM sales GROUP BY category",
            explanation="Sum values by category"
        )
        
        result = agent.execute_query(query)
        
        assert result.row_count == 2
        assert "total" in result.data.columns
        assert result.data["total"].sum() == 100
        
        store.close()
    
    def test_execute_sql_join(self):
        """Test executing SQL JOIN query."""
        store = DataStore()
        df1 = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        df2 = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        store.register_dataframe("table1", df1)
        store.register_dataframe("table2", df2)
        
        agent = ExtractionAgent(store)
        query = StructuredQuery(
            operation_type="sql",
            operation="SELECT t1.name, t2.value FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id",
            explanation="Join two tables"
        )
        
        result = agent.execute_query(query)
        
        assert result.row_count == 2
        assert list(result.data.columns) == ["name", "value"]
        
        store.close()


class TestExecuteQueryPandas:
    """Tests for execute_query with Pandas operations."""
    
    def test_execute_pandas_filter(self):
        """Test executing Pandas filter operation."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        store.register_dataframe("test", df)
        
        agent = ExtractionAgent(store)
        query = StructuredQuery(
            operation_type="pandas",
            operation="df[df['a'] > 2]",
            explanation="Filter rows where a > 2",
            parameters={"table_name": "test"}
        )
        
        result = agent.execute_query(query)
        
        assert result.row_count == 3
        assert result.data["a"].tolist() == [3, 4, 5]
        
        store.close()
    
    def test_execute_pandas_groupby(self):
        """Test executing Pandas groupby operation."""
        store = DataStore()
        df = pd.DataFrame({
            "category": ["A", "A", "B", "B"],
            "value": [10, 20, 30, 40]
        })
        store.register_dataframe("sales", df)
        
        agent = ExtractionAgent(store)
        query = StructuredQuery(
            operation_type="pandas",
            operation="df.groupby('category')['value'].sum().to_frame()",
            explanation="Sum values by category",
            parameters={"table_name": "sales"}
        )
        
        result = agent.execute_query(query)
        
        assert result.row_count == 2
        assert result.data["value"].sum() == 100
        
        store.close()
    
    def test_execute_pandas_column_selection(self):
        """Test executing Pandas column selection."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        store.register_dataframe("test", df)
        
        agent = ExtractionAgent(store)
        query = StructuredQuery(
            operation_type="pandas",
            operation="df[['a', 'c']]",
            explanation="Select columns a and c",
            parameters={"table_name": "test"}
        )
        
        result = agent.execute_query(query)
        
        assert result.row_count == 3
        assert list(result.data.columns) == ["a", "c"]
        
        store.close()
    
    def test_execute_pandas_series_converts_to_dataframe(self):
        """Test that Pandas operations returning Series are converted to DataFrame."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        store.register_dataframe("test", df)
        
        agent = ExtractionAgent(store)
        query = StructuredQuery(
            operation_type="pandas",
            operation="df['a']",
            explanation="Select column a",
            parameters={"table_name": "test"}
        )
        
        result = agent.execute_query(query)
        
        assert isinstance(result.data, pd.DataFrame)
        assert result.row_count == 3
        
        store.close()
    
    def test_execute_pandas_missing_table_name(self):
        """Test that Pandas operation without table_name raises error."""
        store = DataStore()
        agent = ExtractionAgent(store)
        
        query = StructuredQuery(
            operation_type="pandas",
            operation="df['a']",
            explanation="Select column a",
            parameters={}
        )
        
        with pytest.raises(ValueError, match="Pandas operation requires 'table_name'"):
            agent.execute_query(query)
        
        store.close()
    
    def test_execute_pandas_nonexistent_table(self):
        """Test that Pandas operation on nonexistent table raises error."""
        store = DataStore()
        agent = ExtractionAgent(store)
        
        query = StructuredQuery(
            operation_type="pandas",
            operation="df['a']",
            explanation="Select column a",
            parameters={"table_name": "nonexistent"}
        )
        
        with pytest.raises(ValueError, match="Table 'nonexistent' not found"):
            agent.execute_query(query)
        
        store.close()
    
    def test_execute_pandas_invalid_operation(self):
        """Test that invalid Pandas operation raises error."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3]})
        store.register_dataframe("test", df)
        
        agent = ExtractionAgent(store)
        query = StructuredQuery(
            operation_type="pandas",
            operation="df.nonexistent_method()",
            explanation="Invalid operation",
            parameters={"table_name": "test"}
        )
        
        with pytest.raises(ValueError, match="Failed to execute Pandas operation"):
            agent.execute_query(query)
        
        store.close()


class TestQueryTimeout:
    """Tests for query timeout handling."""
    
    def test_timeout_not_exceeded(self):
        """Test that queries completing within timeout succeed."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3]})
        store.register_dataframe("test", df)
        
        agent = ExtractionAgent(store)
        # Set a reasonable timeout
        agent.query_timeout = 5
        
        query = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM test",
            explanation="Simple query"
        )
        
        result = agent.execute_query(query)
        
        assert result.execution_time < agent.query_timeout
        
        store.close()


class TestPagination:
    """Tests for pagination of large result sets."""
    
    def test_pagination_applied_for_large_results(self):
        """Test that pagination is applied when result exceeds max_rows_display."""
        store = DataStore()
        # Create a large DataFrame
        df = pd.DataFrame({"a": range(15000), "b": range(15000)})
        store.register_dataframe("large", df)
        
        agent = ExtractionAgent(store)
        # Set max_rows_display to a small value for testing
        agent.max_rows_display = 100
        
        query = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM large",
            explanation="Get all rows"
        )
        
        result = agent.execute_query(query)
        
        # row_count should reflect original size
        assert result.row_count == 15000
        # But data should be paginated
        assert len(result.data) == 100
        
        store.close()
    
    def test_no_pagination_for_small_results(self):
        """Test that pagination is not applied for small result sets."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3]})
        store.register_dataframe("small", df)
        
        agent = ExtractionAgent(store)
        agent.max_rows_display = 10000
        
        query = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM small",
            explanation="Get all rows"
        )
        
        result = agent.execute_query(query)
        
        assert result.row_count == 3
        assert len(result.data) == 3
        
        store.close()


class TestQueryCaching:
    """Tests for query caching functionality."""
    
    def test_cache_hit_on_repeated_query(self):
        """Test that repeated queries hit the cache."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3]})
        store.register_dataframe("test", df)
        
        agent = ExtractionAgent(store)
        query = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM test",
            explanation="Get all rows"
        )
        
        # First execution - cache miss
        result1 = agent.execute_query(query)
        assert result1.cached == False
        
        # Second execution - cache hit
        result2 = agent.execute_query(query)
        assert result2.cached == True
        
        # Results should be the same
        assert result1.row_count == result2.row_count
        pd.testing.assert_frame_equal(result1.data, result2.data)
        
        store.close()
    
    def test_cache_miss_on_different_query(self):
        """Test that different queries don't hit the cache."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3]})
        store.register_dataframe("test", df)
        
        agent = ExtractionAgent(store)
        
        query1 = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM test",
            explanation="Get all rows"
        )
        
        query2 = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM test WHERE a > 1",
            explanation="Get filtered rows"
        )
        
        result1 = agent.execute_query(query1)
        assert result1.cached == False
        
        result2 = agent.execute_query(query2)
        assert result2.cached == False
        
        store.close()
    
    def test_cache_eviction_lru(self):
        """Test that LRU eviction works when cache is full."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3]})
        store.register_dataframe("test", df)
        
        agent = ExtractionAgent(store)
        # Set small cache size for testing
        agent.cache_size = 2
        
        # Execute 3 different queries
        query1 = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM test WHERE a = 1",
            explanation="Query 1"
        )
        query2 = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM test WHERE a = 2",
            explanation="Query 2"
        )
        query3 = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM test WHERE a = 3",
            explanation="Query 3"
        )
        
        # Execute all three queries
        agent.execute_query(query1)  # Cache: [q1]
        agent.execute_query(query2)  # Cache: [q1, q2]
        agent.execute_query(query3)  # Cache: [q2, q3] (q1 evicted)
        
        # Query 1 should be evicted (cache miss)
        result1 = agent.execute_query(query1)
        assert result1.cached == False
        
        # Query 2 should still be in cache (but was evicted by q1)
        result2 = agent.execute_query(query2)
        # After q1 was added, q2 was evicted, so this should be a miss
        assert result2.cached == False
        
        store.close()
    
    def test_clear_cache(self):
        """Test that clear_cache removes all cached entries."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3]})
        store.register_dataframe("test", df)
        
        agent = ExtractionAgent(store)
        query = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM test",
            explanation="Get all rows"
        )
        
        # Execute query to populate cache
        result1 = agent.execute_query(query)
        assert result1.cached == False
        
        # Verify cache hit
        result2 = agent.execute_query(query)
        assert result2.cached == True
        
        # Clear cache
        agent.clear_cache()
        
        # Should be cache miss now
        result3 = agent.execute_query(query)
        assert result3.cached == False
        
        store.close()
    
    def test_cache_key_generation(self):
        """Test that cache keys are generated correctly."""
        store = DataStore()
        agent = ExtractionAgent(store)
        
        query1 = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM test",
            explanation="Query 1"
        )
        query2 = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM test",
            explanation="Query 2"  # Different explanation, same operation
        )
        
        key1 = agent._get_cache_key(query1)
        key2 = agent._get_cache_key(query2)
        
        # Keys should be the same (explanation doesn't affect cache key)
        assert key1 == key2
        
        store.close()


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_invalid_query_type(self):
        """Test that invalid query type raises ValueError."""
        store = DataStore()
        agent = ExtractionAgent(store)
        
        with pytest.raises(ValueError, match="query must be a StructuredQuery instance"):
            agent.execute_query("not a query")
        
        store.close()
    
    def test_unsupported_operation_type(self):
        """Test that unsupported operation_type raises ValueError."""
        store = DataStore()
        agent = ExtractionAgent(store)
        
        query = StructuredQuery(
            operation_type="unsupported",
            operation="some operation",
            explanation="Test"
        )
        
        with pytest.raises(ValueError, match="Unsupported operation_type"):
            agent.execute_query(query)
        
        store.close()
    
    def test_sql_error_propagates(self):
        """Test that SQL errors are propagated."""
        store = DataStore()
        agent = ExtractionAgent(store)
        
        query = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM nonexistent_table",
            explanation="Query nonexistent table"
        )
        
        with pytest.raises(Exception):
            agent.execute_query(query)
        
        store.close()
