"""Extraction Agent module for executing data queries.

This module implements the ExtractionAgent responsible for executing
structured queries against the data store and retrieving results efficiently.
"""

import logging
import time
import hashlib
from functools import lru_cache
from typing import Optional, Dict
import pandas as pd
from src.data.data_store import DataStore
from src.core.models import StructuredQuery, QueryResult
from src.core.config import Config

logger = logging.getLogger(__name__)


class ExtractionAgent:
    """
    Extraction Agent responsible for executing data queries efficiently.
    
    This agent:
    - Executes SQL queries using DuckDB
    - Executes Pandas operations on DataFrames
    - Implements query caching for performance
    - Handles timeouts and pagination
    - Optimizes queries using columnar processing
    """
    
    def __init__(self, data_store: DataStore):
        """
        Initialize the Extraction Agent.
        
        Args:
            data_store: DataStore instance for query execution
        """
        self.data_store = data_store
        self.query_timeout = Config.QUERY_TIMEOUT
        self.max_rows_display = Config.MAX_ROWS_DISPLAY
        self.cache_size = Config.CACHE_SIZE
        # Initialize query cache (LRU cache with configurable size)
        self._query_cache: Dict[str, QueryResult] = {}
        self._cache_order: list = []  # Track access order for LRU
        logger.info(
            f"ExtractionAgent initialized with timeout={self.query_timeout}s, "
            f"max_rows={self.max_rows_display}, cache_size={self.cache_size}"
        )
    
    def execute_query(self, query: StructuredQuery) -> QueryResult:
        """
        Execute structured query against data store.
        
        This method:
        - Routes to appropriate execution method based on operation_type
        - Implements timeout handling (30 second limit)
        - Applies pagination for large result sets (>10,000 rows)
        - Uses caching for frequently executed queries
        
        Args:
            query: Structured query from Query Agent
            
        Returns:
            QueryResult with data and metadata
            
        Raises:
            ValueError: If operation_type is unsupported
            TimeoutError: If query execution exceeds timeout limit
        """
        if not isinstance(query, StructuredQuery):
            raise ValueError("query must be a StructuredQuery instance")
        
        logger.info(f"Executing {query.operation_type} query: {query.explanation}")
        
        # Check cache first
        cache_key = self._get_cache_key(query)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            logger.info(f"Cache hit for query: {query.explanation}")
            return cached_result
        
        start_time = time.time()
        
        try:
            # Route to appropriate execution method
            if query.operation_type == "sql":
                result_df = self._execute_sql_query(query)
            elif query.operation_type == "pandas":
                result_df = self._execute_pandas_query(query)
            else:
                raise ValueError(
                    f"Unsupported operation_type: {query.operation_type}. "
                    f"Supported types: 'sql', 'pandas'"
                )
            
            execution_time = time.time() - start_time
            
            # Check timeout
            if execution_time > self.query_timeout:
                raise TimeoutError(
                    f"Query execution exceeded timeout limit of {self.query_timeout} seconds. "
                    f"Actual time: {execution_time:.2f}s"
                )
            
            # Apply pagination if result is too large
            original_row_count = len(result_df)
            if original_row_count > self.max_rows_display:
                logger.warning(
                    f"Result set has {original_row_count} rows, applying pagination "
                    f"to limit to {self.max_rows_display} rows"
                )
                result_df = result_df.head(self.max_rows_display)
            
            # Create QueryResult
            result = QueryResult(
                data=result_df,
                row_count=original_row_count,
                execution_time=execution_time,
                query=query,
                cached=False
            )
            
            # Store in cache
            self._add_to_cache(cache_key, result)
            
            logger.info(
                f"Query executed successfully in {execution_time:.2f}s, "
                f"returned {original_row_count} rows"
            )
            
            return result
            
        except TimeoutError:
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Query execution failed after {execution_time:.2f}s: {str(e)}"
            )
            raise
    
    def _execute_sql_query(self, query: StructuredQuery) -> pd.DataFrame:
        """
        Execute SQL query using DuckDB.
        
        DuckDB provides automatic columnar processing optimization for analytical queries,
        which is significantly faster than row-based processing for aggregations and scans.
        
        Args:
            query: StructuredQuery with SQL operation
            
        Returns:
            Pandas DataFrame with results
        """
        logger.debug(f"Executing SQL: {query.operation[:100]}...")
        
        # DuckDB provides columnar processing optimization automatically
        # This is especially efficient for:
        # - Aggregations (SUM, AVG, COUNT, etc.)
        # - Column scans and filters
        # - Analytical queries on large datasets
        result = self.data_store.execute_sql(query.operation)
        
        return result
    
    def _execute_pandas_query(self, query: StructuredQuery) -> pd.DataFrame:
        """
        Execute Pandas operation on DataFrames.
        
        Pandas operations use vectorized operations which provide columnar processing
        optimization for numerical computations and transformations.
        
        Args:
            query: StructuredQuery with Pandas operation
            
        Returns:
            Pandas DataFrame with results
            
        Raises:
            ValueError: If table not found or operation fails
        """
        logger.debug(f"Executing Pandas operation: {query.operation[:100]}...")
        
        # Get table name from parameters
        table_name = query.parameters.get("table_name")
        if not table_name:
            raise ValueError("Pandas operation requires 'table_name' in parameters")
        
        # Get DataFrame from data store
        if table_name not in self.data_store.tables:
            raise ValueError(
                f"Table '{table_name}' not found. "
                f"Available tables: {self.data_store.list_tables()}"
            )
        
        df = self.data_store.tables[table_name]
        
        # Execute Pandas operation
        # The operation should be a valid Pandas expression
        # For safety, we use eval with restricted namespace
        try:
            # Create namespace with df and pandas
            # Pandas uses vectorized operations for columnar processing optimization
            namespace = {"df": df, "pd": pd}
            result = eval(query.operation, {"__builtins__": {}}, namespace)
            
            # Ensure result is a DataFrame
            if not isinstance(result, pd.DataFrame):
                # If result is a Series, convert to DataFrame
                if isinstance(result, pd.Series):
                    result = result.to_frame()
                else:
                    raise ValueError(
                        f"Pandas operation must return DataFrame or Series, "
                        f"got {type(result)}"
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Pandas operation failed: {str(e)}")
            raise ValueError(f"Failed to execute Pandas operation: {str(e)}")

    
    def _get_cache_key(self, query: StructuredQuery) -> str:
        """
        Generate cache key for a query.
        
        Uses hash of operation_type, operation, and parameters.
        
        Args:
            query: StructuredQuery to generate key for
            
        Returns:
            Cache key string
        """
        # Create a string representation of the query
        query_str = f"{query.operation_type}:{query.operation}:{str(query.parameters)}"
        # Generate hash
        cache_key = hashlib.md5(query_str.encode()).hexdigest()
        return cache_key
    
    def _get_from_cache(self, cache_key: str) -> Optional[QueryResult]:
        """
        Retrieve result from cache if available.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Cached QueryResult or None if not found
        """
        if cache_key in self._query_cache:
            # Update access order (move to end for LRU)
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            
            # Mark result as cached
            cached_result = self._query_cache[cache_key]
            cached_result.cached = True
            
            return cached_result
        
        return None
    
    def _add_to_cache(self, cache_key: str, result: QueryResult) -> None:
        """
        Add query result to cache with LRU eviction.
        
        If cache is full, evicts least recently used entry.
        
        Args:
            cache_key: Cache key for the result
            result: QueryResult to cache
        """
        # If cache is full, evict least recently used
        if len(self._query_cache) >= self.cache_size and cache_key not in self._query_cache:
            if self._cache_order:
                lru_key = self._cache_order.pop(0)
                del self._query_cache[lru_key]
                logger.debug(f"Evicted LRU cache entry: {lru_key}")
        
        # Add to cache
        self._query_cache[cache_key] = result
        
        # Update access order
        if cache_key in self._cache_order:
            self._cache_order.remove(cache_key)
        self._cache_order.append(cache_key)
        
        logger.debug(f"Added query to cache. Cache size: {len(self._query_cache)}/{self.cache_size}")
    
    def clear_cache(self) -> None:
        """
        Clear the query cache.
        
        Useful for testing or when data has been updated.
        """
        self._query_cache.clear()
        self._cache_order.clear()
        logger.info("Query cache cleared")
