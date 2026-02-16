"""Data Store module for managing data loading and SQL execution."""

import logging
from typing import Dict, List, Optional
import pandas as pd
import duckdb

logger = logging.getLogger(__name__)


class DataStore:
    """
    Data Store class that manages data loading and query execution using DuckDB.
    
    This class provides an interface for:
    - Loading data from various sources into DuckDB tables
    - Executing SQL queries against loaded data
    - Retrieving schema information and table listings
    """
    
    def __init__(self):
        """Initialize DuckDB in-memory connection."""
        self.connection = duckdb.connect(database=":memory:")
        self.tables: Dict[str, pd.DataFrame] = {}
        logger.info("DataStore initialized with DuckDB in-memory connection")
    
    def register_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """
        Register a Pandas DataFrame as a DuckDB table.
        
        Args:
            table_name: Name for the table in DuckDB
            df: Pandas DataFrame to register
            
        Raises:
            ValueError: If table_name is empty or df is not a DataFrame
        """
        if not table_name or not isinstance(table_name, str):
            raise ValueError("table_name must be a non-empty string")
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        
        # Store DataFrame reference
        self.tables[table_name] = df
        
        # Register with DuckDB
        self.connection.register(table_name, df)
        
        logger.info(f"Registered table '{table_name}' with {len(df)} rows and {len(df.columns)} columns")
    
    def execute_sql(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as a Pandas DataFrame.
        
        Args:
            query: SQL query string to execute
            
        Returns:
            Pandas DataFrame containing query results
            
        Raises:
            ValueError: If query is empty
            duckdb.Error: If query execution fails
        """
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")
        
        # DuckDB uses double quotes for identifiers, not backticks
        # Automatically convert backticks to double quotes for compatibility
        query = query.replace('`', '"')
        
        try:
            logger.debug(f"Executing SQL query: {query[:100]}...")
            result = self.connection.execute(query).fetchdf()
            logger.info(f"Query executed successfully, returned {len(result)} rows")
            return result
        except duckdb.Error as e:
            logger.error(f"SQL execution failed: {str(e)}")
            raise
    
    def get_table_schema(self, table_name: str) -> Dict:
        """
        Get schema information for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary containing schema information with keys:
            - name: table name
            - columns: list of column info dicts with 'name', 'type', 'nullable'
            - row_count: number of rows in the table
            
        Raises:
            ValueError: If table_name doesn't exist
        """
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not found. Available tables: {list(self.tables.keys())}")
        
        df = self.tables[table_name]
        
        # Get column information
        columns = []
        for col_name in df.columns:
            col_info = {
                "name": col_name,
                "type": str(df[col_name].dtype),
                "nullable": df[col_name].isna().any()
            }
            columns.append(col_info)
        
        schema = {
            "name": table_name,
            "columns": columns,
            "row_count": len(df)
        }
        
        logger.debug(f"Retrieved schema for table '{table_name}'")
        return schema
    
    def list_tables(self) -> List[str]:
        """
        List all registered tables.
        
        Returns:
            List of table names
        """
        table_list = list(self.tables.keys())
        logger.debug(f"Listed {len(table_list)} tables")
        return table_list

    def load_csv(self, file_path: str, table_name: Optional[str] = None) -> str:
        """
        Load CSV file into the data store.
        For files >1GB, uses chunked processing to avoid memory overflow.

        Args:
            file_path: Path to the CSV file
            table_name: Optional name for the table. If not provided, uses filename without extension

        Returns:
            Name of the created table

        Raises:
            ValueError: If file_path is invalid or file format is unsupported
            FileNotFoundError: If file doesn't exist
            pd.errors.ParserError: If CSV parsing fails
        """
        import os

        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Validate file extension
        if not file_path.lower().endswith('.csv'):
            raise ValueError(
                f"Unsupported file format. Expected CSV file but got: {file_path}. "
                f"Please provide a file with .csv extension."
            )

        # Generate table name if not provided
        if table_name is None:
            table_name = os.path.splitext(os.path.basename(file_path))[0]
            # Sanitize table name (replace spaces and special chars with underscores)
            table_name = "".join(c if c.isalnum() else "_" for c in table_name)

        try:
            # Check file size to determine loading strategy
            file_size = os.path.getsize(file_path)
            file_size_gb = file_size / (1024 ** 3)
            
            logger.info(f"Loading CSV file: {file_path} (Size: {file_size_gb:.2f} GB)")
            
            # For files >1GB, use chunked processing
            if file_size > 1024 ** 3:  # 1GB in bytes
                logger.info(f"File size exceeds 1GB, using chunked processing")
                return self._load_csv_chunked(file_path, table_name)
            else:
                # Load CSV with pandas, preserving NaN for missing values
                df = pd.read_csv(file_path, keep_default_na=True)

                # Register the DataFrame as a table
                self.register_dataframe(table_name, df)

                logger.info(
                    f"Successfully loaded CSV '{file_path}' as table '{table_name}' "
                    f"with {len(df)} rows and {len(df.columns)} columns"
                )

                return table_name

        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {file_path}")
        except pd.errors.ParserError as e:
            raise ValueError(
                f"Failed to parse CSV file '{file_path}'. "
                f"The file may be corrupted or not in valid CSV format. Error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error loading CSV '{file_path}': {str(e)}")
            raise ValueError(
                f"Failed to load CSV file '{file_path}'. Error: {str(e)}"
            )
    
    def _load_csv_chunked(self, file_path: str, table_name: str, chunk_size: int = 100000) -> str:
        """
        Load large CSV file using chunked processing.
        Reads CSV in chunks and appends to DuckDB table to avoid memory overflow.

        Args:
            file_path: Path to the CSV file
            table_name: Name for the table in DuckDB
            chunk_size: Number of rows per chunk (default: 100,000)

        Returns:
            Name of the created table

        Raises:
            ValueError: If CSV parsing fails
        """
        logger.info(f"Loading CSV in chunks of {chunk_size} rows")
        
        total_rows = 0
        chunk_count = 0
        
        try:
            # Read CSV in chunks using pandas chunksize parameter
            for chunk in pd.read_csv(file_path, keep_default_na=True, chunksize=chunk_size):
                chunk_count += 1
                chunk_rows = len(chunk)
                total_rows += chunk_rows
                
                if chunk_count == 1:
                    # First chunk: create table using CREATE TABLE AS
                    # Register chunk temporarily to create the table
                    self.connection.register("temp_chunk", chunk)
                    self.connection.execute(
                        f"CREATE TABLE {table_name} AS SELECT * FROM temp_chunk"
                    )
                    self.connection.unregister("temp_chunk")
                    logger.info(f"Created table '{table_name}' with first chunk ({chunk_rows} rows)")
                else:
                    # Subsequent chunks: append to existing DuckDB table
                    # Register chunk temporarily and insert
                    self.connection.register("temp_chunk", chunk)
                    self.connection.execute(
                        f"INSERT INTO {table_name} SELECT * FROM temp_chunk"
                    )
                    self.connection.unregister("temp_chunk")
                    logger.debug(f"Appended chunk {chunk_count} ({chunk_rows} rows)")
                
                # Log progress every 10 chunks
                if chunk_count % 10 == 0:
                    logger.info(f"Processed {chunk_count} chunks, {total_rows} total rows")
            
            # After all chunks are loaded, update the stored DataFrame reference
            # by querying the complete table from DuckDB
            self.tables[table_name] = self.connection.execute(
                f"SELECT * FROM {table_name}"
            ).fetchdf()
            
            logger.info(
                f"Successfully loaded CSV '{file_path}' as table '{table_name}' "
                f"with {total_rows} rows in {chunk_count} chunks"
            )
            
            return table_name
            
        except pd.errors.ParserError as e:
            raise ValueError(
                f"Failed to parse CSV file '{file_path}'. "
                f"The file may be corrupted or not in valid CSV format. Error: {str(e)}"
            )
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {file_path}")
        except Exception as e:
            logger.error(f"Error during chunked CSV loading: {str(e)}")
            raise ValueError(
                f"Failed to load CSV file '{file_path}' in chunks. Error: {str(e)}"
            )

    
    def load_excel(self, file_path: str, table_prefix: Optional[str] = None) -> List[str]:
        """
        Load Excel file with multi-sheet support into the data store.
        Each sheet is converted to a separate table.

        Args:
            file_path: Path to the Excel file
            table_prefix: Optional prefix for table names. If not provided, uses filename without extension

        Returns:
            List of table names created (one per sheet)

        Raises:
            ValueError: If file_path is invalid or file format is unsupported
            FileNotFoundError: If file doesn't exist
            Exception: If Excel parsing fails
        """
        import os

        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Validate file extension
        if not file_path.lower().endswith(('.xlsx', '.xls')):
            raise ValueError(
                f"Unsupported file format. Expected Excel file but got: {file_path}. "
                f"Please provide a file with .xlsx or .xls extension."
            )

        # Generate table prefix if not provided
        if table_prefix is None:
            table_prefix = os.path.splitext(os.path.basename(file_path))[0]
            # Sanitize table prefix (replace spaces and special chars with underscores)
            table_prefix = "".join(c if c.isalnum() else "_" for c in table_prefix)

        try:
            # Load Excel file and get all sheets
            logger.info(f"Loading Excel file: {file_path}")
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            if not sheet_names:
                raise ValueError(f"Excel file contains no sheets: {file_path}")

            logger.info(f"Found {len(sheet_names)} sheets in Excel file: {sheet_names}")

            # Load each sheet and register as a separate table
            table_names = []
            for sheet_name in sheet_names:
                # Create table name from prefix and sheet name
                # Sanitize sheet name
                sanitized_sheet = "".join(c if c.isalnum() else "_" for c in sheet_name)
                table_name = f"{table_prefix}_{sanitized_sheet}"

                # Load sheet data
                df = pd.read_excel(file_path, sheet_name=sheet_name, keep_default_na=True)
                
                # Register the DataFrame as a table
                self.register_dataframe(table_name, df)
                table_names.append(table_name)

                logger.info(
                    f"Loaded sheet '{sheet_name}' as table '{table_name}' "
                    f"with {len(df)} rows and {len(df.columns)} columns"
                )

            logger.info(
                f"Successfully loaded Excel file '{file_path}' with {len(table_names)} sheets"
            )

            return table_names

        except ValueError:
            # Re-raise ValueError as-is
            raise
        except FileNotFoundError:
            # Re-raise FileNotFoundError as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading Excel file '{file_path}': {str(e)}")
            raise ValueError(
                f"Failed to load Excel file '{file_path}'. Error: {str(e)}"
            )

    def load_json(self, file_path: str, table_name: Optional[str] = None) -> str:
        """
        Load JSON file into the data store with structure parsing and normalization.
        Handles nested JSON by normalizing to flat tables.

        Args:
            file_path: Path to the JSON file
            table_name: Optional name for the table. If not provided, uses filename without extension

        Returns:
            Name of the created table

        Raises:
            ValueError: If file_path is invalid, file format is unsupported, or JSON is invalid
            FileNotFoundError: If file doesn't exist
        """
        import os
        import json

        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Validate file extension
        if not file_path.lower().endswith('.json'):
            raise ValueError(
                f"Unsupported file format. Expected JSON file but got: {file_path}. "
                f"Please provide a file with .json extension."
            )

        # Generate table name if not provided
        if table_name is None:
            table_name = os.path.splitext(os.path.basename(file_path))[0]
            # Sanitize table name (replace spaces and special chars with underscores)
            table_name = "".join(c if c.isalnum() else "_" for c in table_name)

        try:
            # Load JSON file
            logger.info(f"Loading JSON file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Normalize JSON structure to flat DataFrame
            df = self._normalize_json(data)

            if df.empty:
                raise ValueError(f"JSON file resulted in empty dataset: {file_path}")

            # Register the DataFrame as a table
            self.register_dataframe(table_name, df)

            logger.info(
                f"Successfully loaded JSON '{file_path}' as table '{table_name}' "
                f"with {len(df)} rows and {len(df.columns)} columns"
            )

            return table_name

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON file '{file_path}'. "
                f"The file may be corrupted or not in valid JSON format. Error: {str(e)}"
            )
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except FileNotFoundError:
            # Re-raise FileNotFoundError as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading JSON '{file_path}': {str(e)}")
            raise ValueError(
                f"Failed to load JSON file '{file_path}'. Error: {str(e)}"
            )

    def _normalize_json(self, data) -> pd.DataFrame:
        """
        Normalize JSON data to a flat DataFrame.
        Handles nested objects and arrays.

        Args:
            data: JSON data (dict, list, or primitive)

        Returns:
            Normalized Pandas DataFrame
        """
        # Case 1: Data is already a list of records (most common case)
        if isinstance(data, list):
            if not data:
                # Empty list
                return pd.DataFrame()
            
            # Check if list contains dictionaries (records)
            if all(isinstance(item, dict) for item in data):
                # Use pandas json_normalize for nested structure handling
                df = pd.json_normalize(data)
                return df
            else:
                # List of primitives - create single column DataFrame
                df = pd.DataFrame({'value': data})
                return df
        
        # Case 2: Data is a single dictionary
        elif isinstance(data, dict):
            # Check if dictionary has list values (potential records)
            list_keys = [k for k, v in data.items() if isinstance(v, list)]
            
            if list_keys:
                # If there's a key with a list of dicts, normalize that
                for key in list_keys:
                    if data[key] and isinstance(data[key][0], dict):
                        # Normalize the list of records
                        df = pd.json_normalize(data[key])
                        # Add other top-level keys as columns if they're not lists
                        for k, v in data.items():
                            if k != key and not isinstance(v, (list, dict)):
                                df[k] = v
                        return df
                
                # If lists contain primitives, create columns for each list
                df_dict = {}
                max_len = 0
                
                # First pass: collect all values and find max length
                for k, v in data.items():
                    if isinstance(v, list):
                        df_dict[k] = v
                        max_len = max(max_len, len(v))
                    else:
                        df_dict[k] = v
                
                # Second pass: normalize lengths
                for k, v in df_dict.items():
                    if isinstance(v, list):
                        # Pad lists with None if shorter than max_len
                        if len(v) < max_len:
                            df_dict[k] = v + [None] * (max_len - len(v))
                    else:
                        # Repeat scalar values for each row
                        df_dict[k] = [v] * max_len
                
                return pd.DataFrame(df_dict)
            else:
                # Dictionary with no lists - single row DataFrame
                # Use json_normalize to handle nested dicts
                df = pd.json_normalize(data)
                return df
        
        # Case 3: Data is a primitive value
        else:
            # Create single cell DataFrame
            df = pd.DataFrame({'value': [data]})
            return df

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self.connection:
            self.connection.close()
            logger.info("DuckDB connection closed")
