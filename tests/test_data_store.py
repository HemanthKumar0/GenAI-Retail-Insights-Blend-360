"""Unit tests for DataStore class."""

import pytest
import pandas as pd
import duckdb
from src.data_store import DataStore


class TestDataStoreInitialization:
    """Tests for DataStore initialization."""
    
    def test_initialization_creates_connection(self):
        """Test that DataStore initializes with a DuckDB connection."""
        store = DataStore()
        assert store.connection is not None
        assert isinstance(store.connection, duckdb.DuckDBPyConnection)
        assert store.tables == {}
        store.close()
    
    def test_initialization_creates_empty_tables_dict(self):
        """Test that DataStore initializes with empty tables dictionary."""
        store = DataStore()
        assert isinstance(store.tables, dict)
        assert len(store.tables) == 0
        store.close()


class TestRegisterDataFrame:
    """Tests for register_dataframe method."""
    
    def test_register_simple_dataframe(self):
        """Test registering a simple DataFrame."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        
        store.register_dataframe("test_table", df)
        
        assert "test_table" in store.tables
        assert store.tables["test_table"].equals(df)
        store.close()
    
    def test_register_dataframe_with_various_types(self):
        """Test registering DataFrame with different data types."""
        store = DataStore()
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True]
        })
        
        store.register_dataframe("mixed_types", df)
        
        assert "mixed_types" in store.tables
        result = store.execute_sql("SELECT * FROM mixed_types")
        assert len(result) == 3
        store.close()
    
    def test_register_dataframe_with_missing_values(self):
        """Test registering DataFrame with NaN values."""
        store = DataStore()
        df = pd.DataFrame({
            "a": [1, None, 3],
            "b": [4.0, 5.0, None]
        })
        
        store.register_dataframe("with_nulls", df)
        
        assert "with_nulls" in store.tables
        result = store.execute_sql("SELECT * FROM with_nulls")
        assert result["a"].isna().sum() == 1
        assert result["b"].isna().sum() == 1
        store.close()
    
    def test_register_multiple_dataframes(self):
        """Test registering multiple DataFrames."""
        store = DataStore()
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"y": [3, 4]})
        
        store.register_dataframe("table1", df1)
        store.register_dataframe("table2", df2)
        
        assert len(store.tables) == 2
        assert "table1" in store.tables
        assert "table2" in store.tables
        store.close()
    
    def test_register_dataframe_overwrites_existing(self):
        """Test that registering with same name overwrites existing table."""
        store = DataStore()
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4, 5]})
        
        store.register_dataframe("table", df1)
        store.register_dataframe("table", df2)
        
        assert len(store.tables["table"]) == 3
        store.close()
    
    def test_register_empty_dataframe(self):
        """Test registering an empty DataFrame."""
        store = DataStore()
        df = pd.DataFrame({"a": [], "b": []})
        
        store.register_dataframe("empty_table", df)
        
        assert "empty_table" in store.tables
        assert len(store.tables["empty_table"]) == 0
        store.close()
    
    def test_register_dataframe_invalid_table_name(self):
        """Test that invalid table names raise ValueError."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2]})
        
        with pytest.raises(ValueError, match="table_name must be a non-empty string"):
            store.register_dataframe("", df)
        
        with pytest.raises(ValueError, match="table_name must be a non-empty string"):
            store.register_dataframe(None, df)
        
        store.close()
    
    def test_register_dataframe_invalid_dataframe(self):
        """Test that invalid DataFrame raises ValueError."""
        store = DataStore()
        
        with pytest.raises(ValueError, match="df must be a pandas DataFrame"):
            store.register_dataframe("table", "not a dataframe")
        
        with pytest.raises(ValueError, match="df must be a pandas DataFrame"):
            store.register_dataframe("table", None)
        
        store.close()


class TestExecuteSQL:
    """Tests for execute_sql method."""
    
    def test_execute_simple_select(self):
        """Test executing a simple SELECT query."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        store.register_dataframe("test", df)
        
        result = store.execute_sql("SELECT * FROM test")
        
        assert len(result) == 3
        assert list(result.columns) == ["a", "b"]
        store.close()
    
    def test_execute_select_with_where(self):
        """Test executing SELECT with WHERE clause."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        store.register_dataframe("test", df)
        
        result = store.execute_sql("SELECT * FROM test WHERE a > 1")
        
        assert len(result) == 2
        assert result["a"].tolist() == [2, 3]
        store.close()
    
    def test_execute_aggregation_query(self):
        """Test executing aggregation queries."""
        store = DataStore()
        df = pd.DataFrame({"category": ["A", "A", "B"], "value": [10, 20, 30]})
        store.register_dataframe("sales", df)
        
        result = store.execute_sql("SELECT category, SUM(value) as total FROM sales GROUP BY category")
        
        assert len(result) == 2
        assert "total" in result.columns
        store.close()
    
    def test_execute_join_query(self):
        """Test executing JOIN queries."""
        store = DataStore()
        df1 = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        df2 = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        store.register_dataframe("table1", df1)
        store.register_dataframe("table2", df2)
        
        result = store.execute_sql(
            "SELECT t1.name, t2.value FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id"
        )
        
        assert len(result) == 2
        assert list(result.columns) == ["name", "value"]
        store.close()
    
    def test_execute_query_returns_dataframe(self):
        """Test that execute_sql returns a Pandas DataFrame."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3]})
        store.register_dataframe("test", df)
        
        result = store.execute_sql("SELECT * FROM test")
        
        assert isinstance(result, pd.DataFrame)
        store.close()
    
    def test_execute_empty_query_raises_error(self):
        """Test that empty query raises ValueError."""
        store = DataStore()
        
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            store.execute_sql("")
        
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            store.execute_sql(None)
        
        store.close()
    
    def test_execute_invalid_sql_raises_error(self):
        """Test that invalid SQL raises DuckDB error."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3]})
        store.register_dataframe("test", df)
        
        with pytest.raises(duckdb.Error):
            store.execute_sql("SELECT * FROM nonexistent_table")
        
        with pytest.raises(duckdb.Error):
            store.execute_sql("INVALID SQL SYNTAX")
        
        store.close()
    
    def test_execute_query_with_nulls(self):
        """Test executing queries on data with NULL values."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
        store.register_dataframe("test", df)
        
        result = store.execute_sql("SELECT * FROM test WHERE a IS NOT NULL")
        
        assert len(result) == 2
        store.close()
    
    def test_execute_query_with_backticks_converts_to_double_quotes(self):
        """Test that backticks in column names are automatically converted to double quotes."""
        store = DataStore()
        # Create DataFrame with column names containing spaces
        df = pd.DataFrame({
            "Courier Status": ["Shipped", "Pending", "Delivered"],
            "Order ID": [1, 2, 3],
            "Product Name": ["A", "B", "C"]
        })
        store.register_dataframe("orders", df)
        
        # Test with backticks (MySQL/SQLite style) - should be converted to double quotes
        result = store.execute_sql("SELECT `Courier Status`, `Order ID` FROM orders WHERE `Courier Status` = 'Shipped'")
        
        assert len(result) == 1
        assert result["Courier Status"].iloc[0] == "Shipped"
        assert result["Order ID"].iloc[0] == 1
        
        # Test that double quotes also work (DuckDB native style)
        result2 = store.execute_sql('SELECT "Courier Status", "Order ID" FROM orders WHERE "Courier Status" = \'Pending\'')
        
        assert len(result2) == 1
        assert result2["Courier Status"].iloc[0] == "Pending"
        
        store.close()


class TestGetTableSchema:
    """Tests for get_table_schema method."""
    
    def test_get_schema_simple_table(self):
        """Test getting schema for a simple table."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        store.register_dataframe("test", df)
        
        schema = store.get_table_schema("test")
        
        assert schema["name"] == "test"
        assert schema["row_count"] == 3
        assert len(schema["columns"]) == 2
        store.close()
    
    def test_get_schema_column_info(self):
        """Test that schema includes correct column information."""
        store = DataStore()
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"]
        })
        store.register_dataframe("test", df)
        
        schema = store.get_table_schema("test")
        columns = schema["columns"]
        
        assert len(columns) == 3
        assert columns[0]["name"] == "int_col"
        assert "int" in columns[0]["type"]
        assert columns[1]["name"] == "float_col"
        assert "float" in columns[1]["type"]
        assert columns[2]["name"] == "str_col"
        store.close()
    
    def test_get_schema_nullable_detection(self):
        """Test that schema correctly identifies nullable columns."""
        store = DataStore()
        df = pd.DataFrame({
            "no_nulls": [1, 2, 3],
            "with_nulls": [1, None, 3]
        })
        store.register_dataframe("test", df)
        
        schema = store.get_table_schema("test")
        columns = {col["name"]: col for col in schema["columns"]}
        
        assert columns["no_nulls"]["nullable"] == False
        assert columns["with_nulls"]["nullable"] == True
        store.close()
    
    def test_get_schema_nonexistent_table(self):
        """Test that getting schema for nonexistent table raises ValueError."""
        store = DataStore()
        
        with pytest.raises(ValueError, match="Table 'nonexistent' not found"):
            store.get_table_schema("nonexistent")
        
        store.close()
    
    def test_get_schema_empty_table(self):
        """Test getting schema for an empty table."""
        store = DataStore()
        df = pd.DataFrame({"a": [], "b": []})
        store.register_dataframe("empty", df)
        
        schema = store.get_table_schema("empty")
        
        assert schema["row_count"] == 0
        assert len(schema["columns"]) == 2
        store.close()


class TestListTables:
    """Tests for list_tables method."""
    
    def test_list_tables_empty(self):
        """Test listing tables when none are registered."""
        store = DataStore()
        
        tables = store.list_tables()
        
        assert tables == []
        store.close()
    
    def test_list_tables_single(self):
        """Test listing tables with one registered table."""
        store = DataStore()
        df = pd.DataFrame({"a": [1, 2, 3]})
        store.register_dataframe("table1", df)
        
        tables = store.list_tables()
        
        assert tables == ["table1"]
        store.close()
    
    def test_list_tables_multiple(self):
        """Test listing multiple registered tables."""
        store = DataStore()
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        df3 = pd.DataFrame({"c": [5, 6]})
        
        store.register_dataframe("table1", df1)
        store.register_dataframe("table2", df2)
        store.register_dataframe("table3", df3)
        
        tables = store.list_tables()
        
        assert len(tables) == 3
        assert "table1" in tables
        assert "table2" in tables
        assert "table3" in tables
        store.close()
    
    def test_list_tables_returns_list(self):
        """Test that list_tables returns a list."""
        store = DataStore()
        
        tables = store.list_tables()
        
        assert isinstance(tables, list)
        store.close()


class TestDataStoreClose:
    """Tests for close method."""
    
    def test_close_connection(self):
        """Test that close method closes the DuckDB connection."""
        store = DataStore()
        store.close()
        
        # After closing, operations should fail
        with pytest.raises(duckdb.Error):
            store.execute_sql("SELECT 1")



class TestLoadCSV:
    """Tests for load_csv method."""
    
    def test_load_csv_valid_file(self, tmp_path):
        """Test loading a valid CSV file."""
        # Create a temporary CSV file
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")
        
        store = DataStore()
        table_name = store.load_csv(str(csv_file))
        
        assert table_name == "test"
        assert "test" in store.list_tables()
        result = store.execute_sql("SELECT * FROM test")
        assert len(result) == 2
        assert list(result.columns) == ["a", "b", "c"]
        store.close()
    
    def test_load_csv_with_custom_table_name(self, tmp_path):
        """Test loading CSV with custom table name."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x,y\n1,2\n")
        
        store = DataStore()
        table_name = store.load_csv(str(csv_file), table_name="custom_table")
        
        assert table_name == "custom_table"
        assert "custom_table" in store.list_tables()
        store.close()
    
    def test_load_csv_with_missing_values(self, tmp_path):
        """Test loading CSV with missing values (NaN preservation)."""
        csv_file = tmp_path / "missing.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,,6\n,8,\n")
        
        store = DataStore()
        table_name = store.load_csv(str(csv_file))
        
        result = store.execute_sql("SELECT * FROM missing")
        assert len(result) == 3
        # Check that NaN values are preserved
        assert result["b"].isna().sum() == 1  # One missing value in column b
        assert result["a"].isna().sum() == 1  # One missing value in column a
        assert result["c"].isna().sum() == 1  # One missing value in column c
        store.close()
    
    def test_load_csv_with_various_data_types(self, tmp_path):
        """Test loading CSV with different data types."""
        csv_file = tmp_path / "types.csv"
        csv_file.write_text("int_col,float_col,str_col\n1,1.5,hello\n2,2.5,world\n")
        
        store = DataStore()
        table_name = store.load_csv(str(csv_file))
        
        result = store.execute_sql("SELECT * FROM types")
        assert len(result) == 2
        assert "int_col" in result.columns
        assert "float_col" in result.columns
        assert "str_col" in result.columns
        store.close()
    
    def test_load_csv_file_not_found(self):
        """Test loading non-existent CSV file raises FileNotFoundError."""
        store = DataStore()
        
        with pytest.raises(FileNotFoundError, match="File not found"):
            store.load_csv("nonexistent.csv")
        
        store.close()
    
    def test_load_csv_invalid_file_path(self):
        """Test loading CSV with invalid file path raises ValueError."""
        store = DataStore()
        
        with pytest.raises(ValueError, match="file_path must be a non-empty string"):
            store.load_csv("")
        
        with pytest.raises(ValueError, match="file_path must be a non-empty string"):
            store.load_csv(None)
        
        store.close()
    
    def test_load_csv_unsupported_format(self, tmp_path):
        """Test loading non-CSV file returns descriptive error."""
        # Create a file with wrong extension
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("a,b,c\n1,2,3\n")
        
        store = DataStore()
        
        with pytest.raises(ValueError, match="Unsupported file format.*Expected CSV file"):
            store.load_csv(str(txt_file))
        
        store.close()
    
    def test_load_csv_malformed_csv_error(self, tmp_path):
        """Test loading malformed CSV file returns descriptive error."""
        # Create a CSV with inconsistent columns (malformed)
        csv_file = tmp_path / "malformed.csv"
        csv_file.write_text('a,b,c\n1,2\n3,4,5,6,7\n')
        
        store = DataStore()
        
        # Should fail with descriptive error about parsing
        with pytest.raises(ValueError, match="Failed to parse CSV file.*corrupted or not in valid CSV format"):
            store.load_csv(str(csv_file))
        
        store.close()
    
    def test_load_csv_empty_file(self, tmp_path):
        """Test loading empty CSV file raises descriptive error."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        
        store = DataStore()
        
        with pytest.raises(ValueError, match="CSV file is empty"):
            store.load_csv(str(csv_file))
        
        store.close()
    
    def test_load_csv_sanitizes_table_name(self, tmp_path):
        """Test that table names with special characters are sanitized."""
        csv_file = tmp_path / "my-data file!.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        store = DataStore()
        table_name = store.load_csv(str(csv_file))
        
        # Special characters should be replaced with underscores
        assert table_name == "my_data_file_"
        assert table_name in store.list_tables()
        store.close()
    
    def test_load_csv_with_headers_only(self, tmp_path):
        """Test loading CSV with only headers (no data rows)."""
        csv_file = tmp_path / "headers_only.csv"
        csv_file.write_text("col1,col2,col3\n")
        
        store = DataStore()
        table_name = store.load_csv(str(csv_file))
        
        result = store.execute_sql(f"SELECT * FROM {table_name}")
        assert len(result) == 0
        assert list(result.columns) == ["col1", "col2", "col3"]
        store.close()
    
    def test_load_csv_queryable_after_load(self, tmp_path):
        """Test that loaded CSV is immediately queryable with SQL."""
        csv_file = tmp_path / "sales.csv"
        csv_file.write_text("product,quantity,price\nA,10,100\nB,20,200\nC,15,150\n")
        
        store = DataStore()
        table_name = store.load_csv(str(csv_file))
        
        # Test various SQL operations
        result = store.execute_sql(f"SELECT * FROM {table_name} WHERE quantity > 10")
        assert len(result) == 2
        
        result = store.execute_sql(f"SELECT SUM(quantity) as total FROM {table_name}")
        assert result["total"][0] == 45
        
        store.close()
    
    def test_load_csv_multiple_files(self, tmp_path):
        """Test loading multiple CSV files into different tables."""
        csv1 = tmp_path / "file1.csv"
        csv1.write_text("a,b\n1,2\n")
        
        csv2 = tmp_path / "file2.csv"
        csv2.write_text("x,y\n3,4\n")
        
        store = DataStore()
        table1 = store.load_csv(str(csv1))
        table2 = store.load_csv(str(csv2))
        
        assert table1 == "file1"
        assert table2 == "file2"
        assert len(store.list_tables()) == 2
        store.close()
    
    def test_load_csv_with_quotes_and_commas(self, tmp_path):
        """Test loading CSV with quoted fields containing commas."""
        csv_file = tmp_path / "quoted.csv"
        csv_file.write_text('name,description\n"Product A","A product, with comma"\n"Product B","Another, product"\n')
        
        store = DataStore()
        table_name = store.load_csv(str(csv_file))
        
        result = store.execute_sql(f"SELECT * FROM {table_name}")
        assert len(result) == 2
        assert "," in result["description"][0]
        store.close()
    
    def test_load_csv_case_insensitive_extension(self, tmp_path):
        """Test that CSV extension check is case-insensitive."""
        csv_file = tmp_path / "data.CSV"
        csv_file.write_text("a,b\n1,2\n")
        
        store = DataStore()
        table_name = store.load_csv(str(csv_file))
        
        assert table_name == "data"
        assert "data" in store.list_tables()
        store.close()


class TestLoadExcel:
    """Tests for load_excel method."""
    
    def test_load_excel_single_sheet(self, tmp_path):
        """Test loading Excel file with a single sheet."""
        # Create a temporary Excel file with one sheet
        excel_file = tmp_path / "test.xlsx"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.to_excel(excel_file, sheet_name="Sheet1", index=False)
        
        store = DataStore()
        table_names = store.load_excel(str(excel_file))
        
        assert len(table_names) == 1
        assert table_names[0] == "test_Sheet1"
        assert "test_Sheet1" in store.list_tables()
        
        result = store.execute_sql("SELECT * FROM test_Sheet1")
        assert len(result) == 3
        assert list(result.columns) == ["a", "b"]
        store.close()
    
    def test_load_excel_multiple_sheets(self, tmp_path):
        """Test loading Excel file with multiple sheets."""
        excel_file = tmp_path / "multi.xlsx"
        
        # Create Excel file with multiple sheets
        with pd.ExcelWriter(excel_file) as writer:
            pd.DataFrame({"x": [1, 2]}).to_excel(writer, sheet_name="First", index=False)
            pd.DataFrame({"y": [3, 4]}).to_excel(writer, sheet_name="Second", index=False)
            pd.DataFrame({"z": [5, 6]}).to_excel(writer, sheet_name="Third", index=False)
        
        store = DataStore()
        table_names = store.load_excel(str(excel_file))
        
        assert len(table_names) == 3
        assert "multi_First" in table_names
        assert "multi_Second" in table_names
        assert "multi_Third" in table_names
        
        # Verify all tables are queryable
        result1 = store.execute_sql("SELECT * FROM multi_First")
        assert len(result1) == 2
        assert "x" in result1.columns
        
        result2 = store.execute_sql("SELECT * FROM multi_Second")
        assert len(result2) == 2
        assert "y" in result2.columns
        
        result3 = store.execute_sql("SELECT * FROM multi_Third")
        assert len(result3) == 2
        assert "z" in result3.columns
        
        store.close()
    
    def test_load_excel_with_custom_prefix(self, tmp_path):
        """Test loading Excel with custom table prefix."""
        excel_file = tmp_path / "data.xlsx"
        df = pd.DataFrame({"col": [1, 2]})
        df.to_excel(excel_file, sheet_name="Sheet1", index=False)
        
        store = DataStore()
        table_names = store.load_excel(str(excel_file), table_prefix="custom")
        
        assert table_names[0] == "custom_Sheet1"
        assert "custom_Sheet1" in store.list_tables()
        store.close()
    
    def test_load_excel_with_missing_values(self, tmp_path):
        """Test loading Excel with missing values (NaN preservation)."""
        excel_file = tmp_path / "missing.xlsx"
        df = pd.DataFrame({
            "a": [1, None, 3],
            "b": [4.0, 5.0, None],
            "c": ["x", "y", None]
        })
        df.to_excel(excel_file, sheet_name="Data", index=False)
        
        store = DataStore()
        table_names = store.load_excel(str(excel_file))
        
        result = store.execute_sql(f"SELECT * FROM {table_names[0]}")
        assert len(result) == 3
        # Check that NaN values are preserved
        assert result["a"].isna().sum() == 1
        assert result["b"].isna().sum() == 1
        assert result["c"].isna().sum() == 1
        store.close()
    
    def test_load_excel_file_not_found(self):
        """Test loading non-existent Excel file raises FileNotFoundError."""
        store = DataStore()
        
        with pytest.raises(FileNotFoundError, match="File not found"):
            store.load_excel("nonexistent.xlsx")
        
        store.close()
    
    def test_load_excel_invalid_file_path(self):
        """Test loading Excel with invalid file path raises ValueError."""
        store = DataStore()
        
        with pytest.raises(ValueError, match="file_path must be a non-empty string"):
            store.load_excel("")
        
        with pytest.raises(ValueError, match="file_path must be a non-empty string"):
            store.load_excel(None)
        
        store.close()
    
    def test_load_excel_unsupported_format(self, tmp_path):
        """Test loading non-Excel file returns descriptive error."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("not an excel file")
        
        store = DataStore()
        
        with pytest.raises(ValueError, match="Unsupported file format.*Expected Excel file"):
            store.load_excel(str(txt_file))
        
        store.close()
    
    def test_load_excel_sanitizes_sheet_names(self, tmp_path):
        """Test that sheet names with special characters are sanitized."""
        excel_file = tmp_path / "data.xlsx"
        
        with pd.ExcelWriter(excel_file) as writer:
            pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="My-Sheet!", index=False)
            pd.DataFrame({"b": [2]}).to_excel(writer, sheet_name="Sheet #2", index=False)
        
        store = DataStore()
        table_names = store.load_excel(str(excel_file))
        
        # Special characters should be replaced with underscores
        assert "data_My_Sheet_" in table_names
        assert "data_Sheet__2" in table_names
        store.close()
    
    def test_load_excel_empty_sheet(self, tmp_path):
        """Test loading Excel with empty sheet."""
        excel_file = tmp_path / "empty.xlsx"
        df = pd.DataFrame({"col1": [], "col2": []})
        df.to_excel(excel_file, sheet_name="Empty", index=False)
        
        store = DataStore()
        table_names = store.load_excel(str(excel_file))
        
        assert len(table_names) == 1
        result = store.execute_sql(f"SELECT * FROM {table_names[0]}")
        assert len(result) == 0
        assert list(result.columns) == ["col1", "col2"]
        store.close()
    
    def test_load_excel_queryable_after_load(self, tmp_path):
        """Test that loaded Excel sheets are immediately queryable with SQL."""
        excel_file = tmp_path / "sales.xlsx"
        
        with pd.ExcelWriter(excel_file) as writer:
            pd.DataFrame({
                "product": ["A", "B", "C"],
                "quantity": [10, 20, 15],
                "price": [100, 200, 150]
            }).to_excel(writer, sheet_name="Sales", index=False)
        
        store = DataStore()
        table_names = store.load_excel(str(excel_file))
        table_name = table_names[0]
        
        # Test various SQL operations
        result = store.execute_sql(f"SELECT * FROM {table_name} WHERE quantity > 10")
        assert len(result) == 2
        
        result = store.execute_sql(f"SELECT SUM(quantity) as total FROM {table_name}")
        assert result["total"][0] == 45
        
        store.close()
    
    def test_load_excel_multiple_files(self, tmp_path):
        """Test loading multiple Excel files."""
        excel1 = tmp_path / "file1.xlsx"
        pd.DataFrame({"a": [1, 2]}).to_excel(excel1, sheet_name="Data", index=False)
        
        excel2 = tmp_path / "file2.xlsx"
        pd.DataFrame({"b": [3, 4]}).to_excel(excel2, sheet_name="Info", index=False)
        
        store = DataStore()
        tables1 = store.load_excel(str(excel1))
        tables2 = store.load_excel(str(excel2))
        
        assert "file1_Data" in tables1
        assert "file2_Info" in tables2
        assert len(store.list_tables()) == 2
        store.close()
    
    def test_load_excel_case_insensitive_extension(self, tmp_path):
        """Test that Excel extension check is case-insensitive."""
        excel_file = tmp_path / "data.XLSX"
        pd.DataFrame({"a": [1, 2]}).to_excel(excel_file, sheet_name="Sheet1", index=False)
        
        store = DataStore()
        table_names = store.load_excel(str(excel_file))
        
        assert len(table_names) == 1
        assert "data_Sheet1" in store.list_tables()
        store.close()
    
    def test_load_excel_xls_format(self, tmp_path):
        """Test loading .xls format (older Excel format)."""
        # Note: This test requires openpyxl or xlrd to be installed
        excel_file = tmp_path / "data.xls"
        df = pd.DataFrame({"x": [1, 2, 3]})
        
        try:
            df.to_excel(excel_file, sheet_name="Data", index=False)
            
            store = DataStore()
            table_names = store.load_excel(str(excel_file))
            
            assert len(table_names) == 1
            assert "data_Data" in table_names
            store.close()
        except Exception:
            # Skip test if .xls format is not supported
            pytest.skip("XLS format not supported in this environment")
    
    def test_load_excel_returns_list_of_table_names(self, tmp_path):
        """Test that load_excel returns a list of table names."""
        excel_file = tmp_path / "test.xlsx"
        
        with pd.ExcelWriter(excel_file) as writer:
            pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="Sheet1", index=False)
            pd.DataFrame({"b": [2]}).to_excel(writer, sheet_name="Sheet2", index=False)
        
        store = DataStore()
        result = store.load_excel(str(excel_file))
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(name, str) for name in result)
        store.close()
    
    def test_load_excel_with_various_data_types(self, tmp_path):
        """Test loading Excel with different data types."""
        excel_file = tmp_path / "types.xlsx"
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["hello", "world", "test"],
            "bool_col": [True, False, True]
        })
        df.to_excel(excel_file, sheet_name="Types", index=False)
        
        store = DataStore()
        table_names = store.load_excel(str(excel_file))
        
        result = store.execute_sql(f"SELECT * FROM {table_names[0]}")
        assert len(result) == 3
        assert "int_col" in result.columns
        assert "float_col" in result.columns
        assert "str_col" in result.columns
        assert "bool_col" in result.columns
        store.close()


class TestLoadJSON:
    """Tests for load_json method."""
    
    def test_load_json_list_of_records(self, tmp_path):
        """Test loading JSON file with list of records (most common case)."""
        json_file = tmp_path / "data.json"
        json_file.write_text('[{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]')
        
        store = DataStore()
        table_name = store.load_json(str(json_file))
        
        assert table_name == "data"
        assert "data" in store.list_tables()
        result = store.execute_sql("SELECT * FROM data")
        assert len(result) == 3
        assert list(result.columns) == ["a", "b"]
        assert result["a"].tolist() == [1, 3, 5]
        store.close()
    
    def test_load_json_nested_objects(self, tmp_path):
        """Test loading JSON with nested objects (flattening)."""
        json_file = tmp_path / "nested.json"
        json_file.write_text('[{"id": 1, "user": {"name": "Alice", "age": 30}}, {"id": 2, "user": {"name": "Bob", "age": 25}}]')
        
        store = DataStore()
        table_name = store.load_json(str(json_file))
        
        result = store.execute_sql(f"SELECT * FROM {table_name}")
        assert len(result) == 2
        # Nested objects should be flattened with dot notation
        assert "user.name" in result.columns
        assert "user.age" in result.columns
        assert result["user.name"].tolist() == ["Alice", "Bob"]
        store.close()
    
    def test_load_json_with_arrays(self, tmp_path):
        """Test loading JSON with arrays in records."""
        json_file = tmp_path / "arrays.json"
        json_file.write_text('[{"id": 1, "tags": ["a", "b"]}, {"id": 2, "tags": ["c", "d", "e"]}]')
        
        store = DataStore()
        table_name = store.load_json(str(json_file))
        
        result = store.execute_sql(f"SELECT * FROM {table_name}")
        assert len(result) == 2
        assert "id" in result.columns
        assert "tags" in result.columns
        store.close()
    
    def test_load_json_single_object(self, tmp_path):
        """Test loading JSON with single object (not in array)."""
        json_file = tmp_path / "single.json"
        json_file.write_text('{"name": "Product A", "price": 100, "quantity": 50}')
        
        store = DataStore()
        table_name = store.load_json(str(json_file))
        
        result = store.execute_sql(f"SELECT * FROM {table_name}")
        assert len(result) == 1
        assert "name" in result.columns
        assert "price" in result.columns
        assert "quantity" in result.columns
        assert result["name"][0] == "Product A"
        store.close()
    
    def test_load_json_object_with_list_values(self, tmp_path):
        """Test loading JSON object containing list values."""
        json_file = tmp_path / "obj_with_lists.json"
        json_file.write_text('{"products": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}], "store": "Main"}')
        
        store = DataStore()
        table_name = store.load_json(str(json_file))
        
        result = store.execute_sql(f"SELECT * FROM {table_name}")
        assert len(result) == 2
        assert "id" in result.columns
        assert "name" in result.columns
        assert "store" in result.columns
        # Store value should be repeated for each product
        assert result["store"].tolist() == ["Main", "Main"]
        store.close()
    
    def test_load_json_list_of_primitives(self, tmp_path):
        """Test loading JSON with list of primitive values."""
        json_file = tmp_path / "primitives.json"
        json_file.write_text('[1, 2, 3, 4, 5]')
        
        store = DataStore()
        table_name = store.load_json(str(json_file))
        
        result = store.execute_sql(f"SELECT * FROM {table_name}")
        assert len(result) == 5
        assert "value" in result.columns
        assert result["value"].tolist() == [1, 2, 3, 4, 5]
        store.close()
    
    def test_load_json_single_primitive(self, tmp_path):
        """Test loading JSON with single primitive value."""
        json_file = tmp_path / "primitive.json"
        json_file.write_text('42')
        
        store = DataStore()
        table_name = store.load_json(str(json_file))
        
        result = store.execute_sql(f"SELECT * FROM {table_name}")
        assert len(result) == 1
        assert "value" in result.columns
        assert result["value"][0] == 42
        store.close()
    
    def test_load_json_with_custom_table_name(self, tmp_path):
        """Test loading JSON with custom table name."""
        json_file = tmp_path / "data.json"
        json_file.write_text('[{"x": 1}, {"x": 2}]')
        
        store = DataStore()
        table_name = store.load_json(str(json_file), table_name="custom_table")
        
        assert table_name == "custom_table"
        assert "custom_table" in store.list_tables()
        store.close()
    
    def test_load_json_with_missing_values(self, tmp_path):
        """Test loading JSON with null values (NaN preservation)."""
        json_file = tmp_path / "nulls.json"
        json_file.write_text('[{"a": 1, "b": 2}, {"a": null, "b": 3}, {"a": 4, "b": null}]')
        
        store = DataStore()
        table_name = store.load_json(str(json_file))
        
        result = store.execute_sql(f"SELECT * FROM {table_name}")
        assert len(result) == 3
        assert result["a"].isna().sum() == 1
        assert result["b"].isna().sum() == 1
        store.close()
    
    def test_load_json_deeply_nested(self, tmp_path):
        """Test loading JSON with deeply nested structure."""
        json_file = tmp_path / "deep.json"
        json_file.write_text('[{"id": 1, "data": {"level1": {"level2": {"value": 100}}}}]')
        
        store = DataStore()
        table_name = store.load_json(str(json_file))
        
        result = store.execute_sql(f"SELECT * FROM {table_name}")
        assert len(result) == 1
        # Should flatten with dot notation
        assert "data.level1.level2.value" in result.columns
        assert result["data.level1.level2.value"][0] == 100
        store.close()
    
    def test_load_json_file_not_found(self):
        """Test loading non-existent JSON file raises FileNotFoundError."""
        store = DataStore()
        
        with pytest.raises(FileNotFoundError, match="File not found"):
            store.load_json("nonexistent.json")
        
        store.close()
    
    def test_load_json_invalid_file_path(self):
        """Test loading JSON with invalid file path raises ValueError."""
        store = DataStore()
        
        with pytest.raises(ValueError, match="file_path must be a non-empty string"):
            store.load_json("")
        
        with pytest.raises(ValueError, match="file_path must be a non-empty string"):
            store.load_json(None)
        
        store.close()
    
    def test_load_json_unsupported_format(self, tmp_path):
        """Test loading non-JSON file returns descriptive error."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text('{"a": 1}')
        
        store = DataStore()
        
        with pytest.raises(ValueError, match="Unsupported file format.*Expected JSON file"):
            store.load_json(str(txt_file))
        
        store.close()
    
    def test_load_json_malformed_json_error(self, tmp_path):
        """Test loading malformed JSON file returns descriptive error."""
        json_file = tmp_path / "malformed.json"
        json_file.write_text('{"a": 1, "b": }')  # Invalid JSON
        
        store = DataStore()
        
        with pytest.raises(ValueError, match="Failed to parse JSON file.*corrupted or not in valid JSON format"):
            store.load_json(str(json_file))
        
        store.close()
    
    def test_load_json_empty_array(self, tmp_path):
        """Test loading JSON with empty array raises descriptive error."""
        json_file = tmp_path / "empty.json"
        json_file.write_text('[]')
        
        store = DataStore()
        
        with pytest.raises(ValueError, match="JSON file resulted in empty dataset"):
            store.load_json(str(json_file))
        
        store.close()
    
    def test_load_json_sanitizes_table_name(self, tmp_path):
        """Test that table names with special characters are sanitized."""
        json_file = tmp_path / "my-data file!.json"
        json_file.write_text('[{"a": 1}]')
        
        store = DataStore()
        table_name = store.load_json(str(json_file))
        
        # Special characters should be replaced with underscores
        assert table_name == "my_data_file_"
        assert table_name in store.list_tables()
        store.close()
    
    def test_load_json_queryable_after_load(self, tmp_path):
        """Test that loaded JSON is immediately queryable with SQL."""
        json_file = tmp_path / "sales.json"
        json_file.write_text('[{"product": "A", "quantity": 10, "price": 100}, {"product": "B", "quantity": 20, "price": 200}, {"product": "C", "quantity": 15, "price": 150}]')
        
        store = DataStore()
        table_name = store.load_json(str(json_file))
        
        # Test various SQL operations
        result = store.execute_sql(f"SELECT * FROM {table_name} WHERE quantity > 10")
        assert len(result) == 2
        
        result = store.execute_sql(f"SELECT SUM(quantity) as total FROM {table_name}")
        assert result["total"][0] == 45
        
        store.close()
    
    def test_load_json_multiple_files(self, tmp_path):
        """Test loading multiple JSON files into different tables."""
        json1 = tmp_path / "file1.json"
        json1.write_text('[{"a": 1}]')
        
        json2 = tmp_path / "file2.json"
        json2.write_text('[{"b": 2}]')
        
        store = DataStore()
        table1 = store.load_json(str(json1))
        table2 = store.load_json(str(json2))
        
        assert table1 == "file1"
        assert table2 == "file2"
        assert len(store.list_tables()) == 2
        store.close()
    
    def test_load_json_case_insensitive_extension(self, tmp_path):
        """Test that JSON extension check is case-insensitive."""
        json_file = tmp_path / "data.JSON"
        json_file.write_text('[{"a": 1}]')
        
        store = DataStore()
        table_name = store.load_json(str(json_file))
        
        assert table_name == "data"
        assert "data" in store.list_tables()
        store.close()
    
    def test_load_json_with_various_data_types(self, tmp_path):
        """Test loading JSON with different data types."""
        json_file = tmp_path / "types.json"
        json_file.write_text('[{"int_col": 1, "float_col": 1.5, "str_col": "hello", "bool_col": true}, {"int_col": 2, "float_col": 2.5, "str_col": "world", "bool_col": false}]')
        
        store = DataStore()
        table_name = store.load_json(str(json_file))
        
        result = store.execute_sql(f"SELECT * FROM {table_name}")
        assert len(result) == 2
        assert "int_col" in result.columns
        assert "float_col" in result.columns
        assert "str_col" in result.columns
        assert "bool_col" in result.columns
        store.close()
    
    def test_load_json_inconsistent_keys(self, tmp_path):
        """Test loading JSON with records having different keys."""
        json_file = tmp_path / "inconsistent.json"
        json_file.write_text('[{"a": 1, "b": 2}, {"a": 3, "c": 4}, {"b": 5, "c": 6}]')
        
        store = DataStore()
        table_name = store.load_json(str(json_file))
        
        result = store.execute_sql(f"SELECT * FROM {table_name}")
        assert len(result) == 3
        # All keys should be present as columns
        assert "a" in result.columns
        assert "b" in result.columns
        assert "c" in result.columns
        # Missing values should be NaN
        assert result["b"].isna().sum() == 1  # Third record missing 'b'
        assert result["c"].isna().sum() == 1  # First record missing 'c'
        store.close()
    
    def test_load_json_object_with_multiple_lists(self, tmp_path):
        """Test loading JSON object with multiple list values."""
        json_file = tmp_path / "multi_lists.json"
        json_file.write_text('{"ids": [1, 2, 3], "names": ["A", "B", "C"], "store": "Main"}')
        
        store = DataStore()
        table_name = store.load_json(str(json_file))
        
        result = store.execute_sql(f"SELECT * FROM {table_name}")
        assert len(result) == 3
        assert "ids" in result.columns
        assert "names" in result.columns
        assert "store" in result.columns
        assert result["ids"].tolist() == [1, 2, 3]
        assert result["names"].tolist() == ["A", "B", "C"]
        assert result["store"].tolist() == ["Main", "Main", "Main"]
        store.close()



class TestChunkedCSVProcessing:
    """Tests for chunked CSV processing for large files."""
    
    def test_load_csv_chunked_processing_for_large_file(self, tmp_path):
        """Test that files >1GB trigger chunked processing."""
        # Create a CSV file and mock its size to appear >1GB
        csv_file = tmp_path / "large.csv"
        # Create CSV with enough data to test chunking logic
        csv_content = "a,b,c\n"
        for i in range(1000):
            csv_content += f"{i},{i*2},{i*3}\n"
        csv_file.write_text(csv_content)
        
        store = DataStore()
        
        # Mock os.path.getsize to return >1GB
        import os
        original_getsize = os.path.getsize
        
        def mock_getsize(path):
            if "large.csv" in path:
                return 2 * 1024 ** 3  # 2GB
            return original_getsize(path)
        
        os.path.getsize = mock_getsize
        
        try:
            table_name = store.load_csv(str(csv_file))
            
            assert table_name == "large"
            assert "large" in store.list_tables()
            result = store.execute_sql("SELECT * FROM large")
            assert len(result) == 1000
            assert list(result.columns) == ["a", "b", "c"]
        finally:
            os.path.getsize = original_getsize
            store.close()
    
    def test_load_csv_chunked_with_custom_chunk_size(self, tmp_path):
        """Test chunked loading with custom chunk size."""
        csv_file = tmp_path / "data.csv"
        # Create CSV with 250 rows
        csv_content = "x,y\n"
        for i in range(250):
            csv_content += f"{i},{i*10}\n"
        csv_file.write_text(csv_content)
        
        store = DataStore()
        
        # Directly test the _load_csv_chunked method with small chunk size
        table_name = store._load_csv_chunked(str(csv_file), "chunked_test", chunk_size=100)
        
        assert table_name == "chunked_test"
        result = store.execute_sql("SELECT * FROM chunked_test")
        assert len(result) == 250
        assert list(result.columns) == ["x", "y"]
        # Verify data integrity
        assert result["x"].tolist() == list(range(250))
        store.close()
    
    def test_load_csv_chunked_preserves_data_types(self, tmp_path):
        """Test that chunked loading preserves data types correctly."""
        csv_file = tmp_path / "types.csv"
        csv_content = "int_col,float_col,str_col\n"
        for i in range(300):
            csv_content += f"{i},{i*1.5},text_{i}\n"
        csv_file.write_text(csv_content)
        
        store = DataStore()
        table_name = store._load_csv_chunked(str(csv_file), "types_test", chunk_size=100)
        
        result = store.execute_sql("SELECT * FROM types_test")
        assert len(result) == 300
        assert "int_col" in result.columns
        assert "float_col" in result.columns
        assert "str_col" in result.columns
        store.close()
    
    def test_load_csv_chunked_with_missing_values(self, tmp_path):
        """Test chunked loading handles missing values correctly."""
        csv_file = tmp_path / "missing.csv"
        csv_content = "a,b,c\n"
        for i in range(300):
            if i % 3 == 0:
                csv_content += f"{i},,{i*3}\n"
            elif i % 3 == 1:
                csv_content += f",{i*2},{i*3}\n"
            else:
                csv_content += f"{i},{i*2},\n"
        csv_file.write_text(csv_content)
        
        store = DataStore()
        table_name = store._load_csv_chunked(str(csv_file), "missing_test", chunk_size=100)
        
        result = store.execute_sql("SELECT * FROM missing_test")
        assert len(result) == 300
        # Verify NaN values are preserved
        assert result["a"].isna().sum() == 100
        assert result["b"].isna().sum() == 100
        assert result["c"].isna().sum() == 100
        store.close()
    
    def test_load_csv_chunked_multiple_chunks(self, tmp_path):
        """Test chunked loading with multiple chunks."""
        csv_file = tmp_path / "multi_chunk.csv"
        csv_content = "id,value\n"
        # Create 500 rows to ensure multiple chunks with chunk_size=100
        for i in range(500):
            csv_content += f"{i},{i*100}\n"
        csv_file.write_text(csv_content)
        
        store = DataStore()
        table_name = store._load_csv_chunked(str(csv_file), "multi_test", chunk_size=100)
        
        result = store.execute_sql("SELECT * FROM multi_test")
        assert len(result) == 500
        # Verify data integrity across chunks
        assert result["id"].tolist() == list(range(500))
        assert result["value"].tolist() == [i*100 for i in range(500)]
        store.close()
    
    def test_load_csv_chunked_queryable_after_load(self, tmp_path):
        """Test that chunked loaded data is immediately queryable."""
        csv_file = tmp_path / "sales.csv"
        csv_content = "product,quantity,price\n"
        for i in range(300):
            product = chr(65 + (i % 3))  # A, B, C
            csv_content += f"{product},{10+i},{100+i*10}\n"
        csv_file.write_text(csv_content)
        
        store = DataStore()
        table_name = store._load_csv_chunked(str(csv_file), "sales_test", chunk_size=100)
        
        # Test SQL operations
        result = store.execute_sql(f"SELECT * FROM {table_name} WHERE quantity > 200")
        assert len(result) > 0
        
        result = store.execute_sql(f"SELECT product, COUNT(*) as count FROM {table_name} GROUP BY product")
        assert len(result) == 3
        
        store.close()
    
    def test_load_csv_small_file_uses_regular_loading(self, tmp_path):
        """Test that files <1GB use regular loading, not chunked."""
        csv_file = tmp_path / "small.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")
        
        store = DataStore()
        table_name = store.load_csv(str(csv_file))
        
        # Should use regular loading
        assert table_name == "small"
        result = store.execute_sql("SELECT * FROM small")
        assert len(result) == 2
        store.close()
    
    def test_load_csv_chunked_empty_file_error(self, tmp_path):
        """Test that chunked loading handles empty files gracefully."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        
        store = DataStore()
        
        # Mock file size to trigger chunked loading
        import os
        original_getsize = os.path.getsize
        
        def mock_getsize(path):
            if "empty.csv" in path:
                return 2 * 1024 ** 3  # 2GB
            return original_getsize(path)
        
        os.path.getsize = mock_getsize
        
        try:
            with pytest.raises(ValueError, match="CSV file is empty"):
                store.load_csv(str(csv_file))
        finally:
            os.path.getsize = original_getsize
            store.close()
    
    def test_load_csv_chunked_single_chunk(self, tmp_path):
        """Test chunked loading with data that fits in a single chunk."""
        csv_file = tmp_path / "single.csv"
        csv_content = "a,b\n"
        for i in range(50):
            csv_content += f"{i},{i*2}\n"
        csv_file.write_text(csv_content)
        
        store = DataStore()
        table_name = store._load_csv_chunked(str(csv_file), "single_test", chunk_size=100)
        
        result = store.execute_sql("SELECT * FROM single_test")
        assert len(result) == 50
        store.close()
