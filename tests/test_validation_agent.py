"""Unit tests for ValidationAgent.

Tests cover:
- Data type verification
- Mathematical consistency checks
- Empty result validation
- Anomaly detection (negative sales, invalid dates)
- Confidence calculation
"""

import pytest
import pandas as pd
import numpy as np
from src.validation_agent import ValidationAgent
from src.models import QueryResult, StructuredQuery, ValidationResult, Anomaly


@pytest.fixture
def validation_agent():
    """Create a ValidationAgent instance for testing."""
    return ValidationAgent()


@pytest.fixture
def sample_query():
    """Create a sample StructuredQuery for testing."""
    return StructuredQuery(
        operation_type="sql",
        operation="SELECT * FROM sales",
        explanation="Get all sales data"
    )


class TestValidateResults:
    """Tests for validate_results method."""
    
    def test_validate_results_with_valid_data(self, validation_agent, sample_query):
        """Test validation passes with valid data."""
        df = pd.DataFrame({
            'product': ['A', 'B', 'C'],
            'sales': [100.0, 200.0, 150.0],
            'quantity': [10, 20, 15]
        })
        
        result = QueryResult(
            data=df,
            row_count=3,
            execution_time=0.1,
            query=sample_query
        )
        
        validation = validation_agent.validate_results(result, sample_query)
        
        assert isinstance(validation, ValidationResult)
        assert validation.passed is True
        assert validation.confidence > 0.8
    
    def test_validate_results_with_negative_sales(self, validation_agent, sample_query):
        """Test validation detects negative sales values."""
        df = pd.DataFrame({
            'product': ['A', 'B', 'C'],
            'sales': [100.0, -50.0, 150.0],
            'quantity': [10, 20, 15]
        })
        
        result = QueryResult(
            data=df,
            row_count=3,
            execution_time=0.1,
            query=sample_query
        )
        
        validation = validation_agent.validate_results(result, sample_query)
        
        assert len(validation.anomalies) > 0
        assert any(a.type == "negative_value" for a in validation.anomalies)
        assert validation.confidence < 1.0
    
    def test_validate_results_with_empty_dataframe(self, validation_agent, sample_query):
        """Test validation handles empty results."""
        df = pd.DataFrame()
        
        result = QueryResult(
            data=df,
            row_count=0,
            execution_time=0.1,
            query=sample_query
        )
        
        validation = validation_agent.validate_results(result, sample_query)
        
        assert isinstance(validation, ValidationResult)
        assert len(validation.issues) > 0
    
    def test_validate_results_invalid_input_types(self, validation_agent, sample_query):
        """Test validation raises error for invalid input types."""
        with pytest.raises(ValueError, match="results must be a QueryResult instance"):
            validation_agent.validate_results("not a result", sample_query)
        
        df = pd.DataFrame({'col': [1, 2, 3]})
        result = QueryResult(
            data=df,
            row_count=3,
            execution_time=0.1,
            query=sample_query
        )
        
        with pytest.raises(ValueError, match="query must be a StructuredQuery instance"):
            validation_agent.validate_results(result, "not a query")


class TestDataTypeValidation:
    """Tests for _validate_data_types method."""
    
    def test_validate_data_types_with_valid_types(self, validation_agent):
        """Test data type validation with correct types."""
        df = pd.DataFrame({
            'name': ['A', 'B', 'C'],
            'value': [1, 2, 3],
            'price': [10.5, 20.3, 15.7]
        })
        
        issues = validation_agent._validate_data_types(df)
        
        assert isinstance(issues, list)
        # Should have minimal or no issues with clean data
    
    def test_validate_data_types_with_datetime_nulls(self, validation_agent):
        """Test detection of null values in datetime columns."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', None, '2023-01-03']),
            'value': [1, 2, 3]
        })
        
        issues = validation_agent._validate_data_types(df)
        
        assert len(issues) > 0
        assert any('datetime' in issue.lower() for issue in issues)
    
    def test_validate_data_types_with_empty_dataframe(self, validation_agent):
        """Test data type validation with empty DataFrame."""
        df = pd.DataFrame()
        
        issues = validation_agent._validate_data_types(df)
        
        assert issues == []


class TestMathematicalConsistency:
    """Tests for _check_mathematical_consistency method."""
    
    def test_check_mathematical_consistency_valid_data(self, validation_agent):
        """Test mathematical consistency with valid data."""
        df = pd.DataFrame({
            'product': ['A', 'B', 'C'],
            'sales': [100.0, 200.0, 150.0]
        })
        
        issues, anomalies = validation_agent._check_mathematical_consistency(df)
        
        assert isinstance(issues, list)
        assert isinstance(anomalies, list)
    
    def test_check_mathematical_consistency_negative_sales(self, validation_agent):
        """Test detection of negative sales values."""
        df = pd.DataFrame({
            'product': ['A', 'B', 'C'],
            'sales': [100.0, -50.0, 150.0]
        })
        
        issues, anomalies = validation_agent._check_mathematical_consistency(df)
        
        assert len(anomalies) > 0
        negative_anomalies = [a for a in anomalies if a.type == "negative_value"]
        assert len(negative_anomalies) > 0
        assert negative_anomalies[0].severity == "warning"
    
    def test_check_mathematical_consistency_sum_inconsistency(self, validation_agent):
        """Test detection of sum inconsistencies (total != subtotal + tax)."""
        df = pd.DataFrame({
            'subtotal': [100.0, 200.0, 150.0],
            'tax': [10.0, 20.0, 15.0],
            'total': [110.0, 220.0, 999.0]  # Last one is inconsistent
        })
        
        issues, anomalies = validation_agent._check_mathematical_consistency(df)
        
        assert len(issues) > 0
        assert any('inconsistency' in issue.lower() for issue in issues)
        assert len(anomalies) > 0
        sum_anomalies = [a for a in anomalies if a.type == "sum_inconsistency"]
        assert len(sum_anomalies) > 0
        assert sum_anomalies[0].severity == "error"
    
    def test_check_mathematical_consistency_multiplication_inconsistency(self, validation_agent):
        """Test detection of multiplication inconsistencies (amount != price * quantity)."""
        df = pd.DataFrame({
            'price': [10.0, 20.0, 15.0],
            'quantity': [5, 10, 8],
            'amount': [50.0, 200.0, 999.0]  # Last one is inconsistent
        })
        
        issues, anomalies = validation_agent._check_mathematical_consistency(df)
        
        assert len(issues) > 0
        assert any('inconsistency' in issue.lower() for issue in issues)
        mult_anomalies = [a for a in anomalies if a.type == "multiplication_inconsistency"]
        assert len(mult_anomalies) > 0
    
    def test_check_mathematical_consistency_empty_dataframe(self, validation_agent):
        """Test mathematical consistency with empty DataFrame."""
        df = pd.DataFrame()
        
        issues, anomalies = validation_agent._check_mathematical_consistency(df)
        
        assert issues == []
        assert anomalies == []


class TestEmptyResultValidation:
    """Tests for _validate_empty_results method."""
    
    def test_validate_empty_results_with_data(self, validation_agent, sample_query):
        """Test empty result validation when results contain data."""
        df = pd.DataFrame({'col': [1, 2, 3]})
        result = QueryResult(
            data=df,
            row_count=3,
            execution_time=0.1,
            query=sample_query
        )
        
        issues = validation_agent._validate_empty_results(result, sample_query)
        
        assert issues == []
    
    def test_validate_empty_results_with_where_clause(self, validation_agent):
        """Test empty result validation detects WHERE clause."""
        query = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM sales WHERE price > 1000",
            explanation="Get expensive items"
        )
        
        df = pd.DataFrame()
        result = QueryResult(
            data=df,
            row_count=0,
            execution_time=0.1,
            query=query
        )
        
        issues = validation_agent._validate_empty_results(result, query)
        
        assert len(issues) > 0
        assert any('WHERE' in issue for issue in issues)
    
    def test_validate_empty_results_with_join(self, validation_agent):
        """Test empty result validation detects JOIN."""
        query = StructuredQuery(
            operation_type="sql",
            operation="SELECT * FROM sales JOIN products ON sales.id = products.id",
            explanation="Join sales with products"
        )
        
        df = pd.DataFrame()
        result = QueryResult(
            data=df,
            row_count=0,
            execution_time=0.1,
            query=query
        )
        
        issues = validation_agent._validate_empty_results(result, query)
        
        assert len(issues) > 0
        assert any('JOIN' in issue for issue in issues)
    
    def test_validate_empty_results_with_having(self, validation_agent):
        """Test empty result validation detects HAVING clause."""
        query = StructuredQuery(
            operation_type="sql",
            operation="SELECT category, SUM(sales) FROM sales GROUP BY category HAVING SUM(sales) > 10000",
            explanation="Get high-sales categories"
        )
        
        df = pd.DataFrame()
        result = QueryResult(
            data=df,
            row_count=0,
            execution_time=0.1,
            query=query
        )
        
        issues = validation_agent._validate_empty_results(result, query)
        
        assert len(issues) > 0
        assert any('HAVING' in issue for issue in issues)


class TestConfidenceCalculation:
    """Tests for _calculate_confidence method."""
    
    def test_calculate_confidence_no_issues(self, validation_agent):
        """Test confidence calculation with no issues."""
        confidence = validation_agent._calculate_confidence([], [])
        
        assert confidence == 1.0
    
    def test_calculate_confidence_with_issues(self, validation_agent):
        """Test confidence decreases with issues."""
        issues = ["Issue 1", "Issue 2"]
        confidence = validation_agent._calculate_confidence(issues, [])
        
        assert confidence < 1.0
        assert confidence == 0.8  # 1.0 - (2 * 0.1)
    
    def test_calculate_confidence_with_warning_anomalies(self, validation_agent):
        """Test confidence decreases with warning anomalies."""
        anomalies = [
            Anomaly(type="test", description="test", severity="warning")
        ]
        confidence = validation_agent._calculate_confidence([], anomalies)
        
        assert confidence < 1.0
        assert confidence == 0.95  # 1.0 - 0.05
    
    def test_calculate_confidence_with_error_anomalies(self, validation_agent):
        """Test confidence decreases more with error anomalies."""
        anomalies = [
            Anomaly(type="test", description="test", severity="error")
        ]
        confidence = validation_agent._calculate_confidence([], anomalies)
        
        assert confidence < 1.0
        assert confidence == 0.8  # 1.0 - 0.2
    
    def test_calculate_confidence_minimum_zero(self, validation_agent):
        """Test confidence never goes below 0.0."""
        issues = ["Issue"] * 20  # Many issues
        anomalies = [
            Anomaly(type="test", description="test", severity="error")
        ] * 10  # Many errors
        
        confidence = validation_agent._calculate_confidence(issues, anomalies)
        
        assert confidence >= 0.0
        assert confidence == 0.0


class TestAnomalyDetection:
    """Tests for _detect_anomalies method."""
    
    def test_detect_anomalies_with_valid_data(self, validation_agent):
        """Test anomaly detection with valid data."""
        df = pd.DataFrame({
            'product': ['A', 'B', 'C'],
            'sales': [100.0, 200.0, 150.0],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        })
        
        issues, anomalies = validation_agent._detect_anomalies(df)
        
        assert isinstance(issues, list)
        assert isinstance(anomalies, list)
        assert len(anomalies) == 0  # No anomalies in valid data
    
    def test_detect_anomalies_negative_sales(self, validation_agent):
        """Test detection of negative sales values."""
        df = pd.DataFrame({
            'product': ['A', 'B', 'C'],
            'sales': [100.0, -50.0, 150.0]
        })
        
        issues, anomalies = validation_agent._detect_anomalies(df)
        
        assert len(anomalies) > 0
        negative_anomalies = [a for a in anomalies if a.type == "negative_sales"]
        assert len(negative_anomalies) > 0
        assert negative_anomalies[0].severity == "error"  # Sales should be error severity
        assert len(negative_anomalies[0].affected_rows) > 0
    
    def test_detect_anomalies_negative_revenue(self, validation_agent):
        """Test detection of negative revenue values."""
        df = pd.DataFrame({
            'product': ['A', 'B', 'C'],
            'revenue': [100.0, -50.0, 150.0]
        })
        
        issues, anomalies = validation_agent._detect_anomalies(df)
        
        assert len(anomalies) > 0
        negative_anomalies = [a for a in anomalies if a.type == "negative_sales"]
        assert len(negative_anomalies) > 0
        assert negative_anomalies[0].severity == "error"  # Revenue should be error severity
    
    def test_detect_anomalies_negative_price(self, validation_agent):
        """Test detection of negative price values."""
        df = pd.DataFrame({
            'product': ['A', 'B', 'C'],
            'price': [10.0, -5.0, 15.0]
        })
        
        issues, anomalies = validation_agent._detect_anomalies(df)
        
        assert len(anomalies) > 0
        negative_anomalies = [a for a in anomalies if a.type == "negative_sales"]
        assert len(negative_anomalies) > 0
        assert negative_anomalies[0].severity == "warning"  # Price should be warning severity
    
    def test_detect_anomalies_invalid_dates_nat(self, validation_agent):
        """Test detection of NaT (Not a Time) values in date columns."""
        df = pd.DataFrame({
            'product': ['A', 'B', 'C'],
            'date': pd.to_datetime(['2023-01-01', None, '2023-01-03'])
        })
        
        issues, anomalies = validation_agent._detect_anomalies(df)
        
        assert len(anomalies) > 0
        date_anomalies = [a for a in anomalies if a.type == "invalid_date"]
        assert len(date_anomalies) > 0
        assert "NaT" in date_anomalies[0].description
        assert date_anomalies[0].severity == "warning"
    
    def test_detect_anomalies_out_of_range_dates(self, validation_agent):
        """Test detection of dates outside reasonable range."""
        df = pd.DataFrame({
            'product': ['A', 'B', 'C'],
            'date': pd.to_datetime(['2023-01-01', '1850-01-01', '2150-01-01'])
        })
        
        issues, anomalies = validation_agent._detect_anomalies(df)
        
        assert len(anomalies) > 0
        date_anomalies = [a for a in anomalies if a.type == "invalid_date"]
        assert len(date_anomalies) > 0
        assert "out-of-range" in date_anomalies[0].description.lower() or "outside reasonable range" in date_anomalies[0].description.lower()
    
    def test_detect_anomalies_multiple_issues(self, validation_agent):
        """Test detection of multiple anomalies in same dataset."""
        df = pd.DataFrame({
            'product': ['A', 'B', 'C'],
            'sales': [100.0, -50.0, 150.0],
            'date': pd.to_datetime(['2023-01-01', None, '2023-01-03'])
        })
        
        issues, anomalies = validation_agent._detect_anomalies(df)
        
        assert len(anomalies) >= 2  # At least negative sales and invalid date
        assert len(issues) >= 2
        assert any(a.type == "negative_sales" for a in anomalies)
        assert any(a.type == "invalid_date" for a in anomalies)
    
    def test_detect_anomalies_empty_dataframe(self, validation_agent):
        """Test anomaly detection with empty DataFrame."""
        df = pd.DataFrame()
        
        issues, anomalies = validation_agent._detect_anomalies(df)
        
        assert issues == []
        assert anomalies == []
    
    def test_detect_anomalies_affected_rows_limited(self, validation_agent):
        """Test that affected_rows is limited to 10 entries."""
        # Create DataFrame with many negative values
        df = pd.DataFrame({
            'sales': [-1.0] * 20  # 20 negative values
        })
        
        issues, anomalies = validation_agent._detect_anomalies(df)
        
        assert len(anomalies) > 0
        negative_anomalies = [a for a in anomalies if a.type == "negative_sales"]
        assert len(negative_anomalies) > 0
        # Should be limited to 10 rows
        assert len(negative_anomalies[0].affected_rows) <= 10




class TestBusinessRuleValidation:
    """Tests for business rule validation functionality."""
    
    def test_check_business_rules_valid_categories(self, validation_agent):
        """Test that valid categories pass validation."""
        df = pd.DataFrame({
            'category': ['Electronics', 'Clothing', 'Food'],
            'sales': [100, 200, 300]
        })
        results = QueryResult(
            data=df,
            row_count=3,
            execution_time=0.1,
            query=StructuredQuery(
                operation_type="sql",
                operation="SELECT * FROM sales",
                explanation="Test query"
            )
        )
        
        anomalies = validation_agent.check_business_rules(results)
        assert len(anomalies) == 0
    
    def test_check_business_rules_invalid_categories(self, validation_agent):
        """Test that invalid categories are flagged."""
        df = pd.DataFrame({
            'category': ['Electronics', 'InvalidCategory', 'Food', 'AnotherInvalid'],
            'sales': [100, 200, 300, 400]
        })
        results = QueryResult(
            data=df,
            row_count=4,
            execution_time=0.1,
            query=StructuredQuery(
                operation_type="sql",
                operation="SELECT * FROM sales",
                explanation="Test query"
            )
        )
        
        anomalies = validation_agent.check_business_rules(results)
        assert len(anomalies) == 1
        assert anomalies[0].type == "invalid_category"
        assert anomalies[0].severity == "error"
        assert "2" in anomalies[0].description  # 2 invalid rows
    
    def test_check_business_rules_sales_order_match_valid(self, validation_agent):
        """Test that valid sales/order relationships pass validation."""
        df = pd.DataFrame({
            'sales': [100, 200, 300],
            'order_count': [1, 2, 3]
        })
        results = QueryResult(
            data=df,
            row_count=3,
            execution_time=0.1,
            query=StructuredQuery(
                operation_type="sql",
                operation="SELECT * FROM sales",
                explanation="Test query"
            )
        )
        
        anomalies = validation_agent.check_business_rules(results)
        assert len(anomalies) == 0
    
    def test_check_business_rules_sales_order_mismatch_no_sales(self, validation_agent):
        """Test that orders without sales are flagged."""
        df = pd.DataFrame({
            'sales': [100, 0, -10],
            'order_count': [1, 2, 3]
        })
        results = QueryResult(
            data=df,
            row_count=3,
            execution_time=0.1,
            query=StructuredQuery(
                operation_type="sql",
                operation="SELECT * FROM sales",
                explanation="Test query"
            )
        )
        
        anomalies = validation_agent.check_business_rules(results)
        # Should flag rows where order_count > 0 but sales <= 0
        sales_order_anomalies = [a for a in anomalies if a.type == "sales_order_mismatch"]
        assert len(sales_order_anomalies) >= 1
        assert sales_order_anomalies[0].severity == "error"
    
    def test_check_business_rules_sales_order_mismatch_no_orders(self, validation_agent):
        """Test that sales without orders are flagged."""
        df = pd.DataFrame({
            'sales': [100, 200, 300],
            'order_count': [1, 0, -1]
        })
        results = QueryResult(
            data=df,
            row_count=3,
            execution_time=0.1,
            query=StructuredQuery(
                operation_type="sql",
                operation="SELECT * FROM sales",
                explanation="Test query"
            )
        )
        
        anomalies = validation_agent.check_business_rules(results)
        # Should flag rows where sales > 0 but order_count <= 0
        sales_order_anomalies = [a for a in anomalies if a.type == "sales_order_mismatch"]
        assert len(sales_order_anomalies) >= 1
        assert sales_order_anomalies[0].severity == "error"
    
    def test_check_business_rules_sales_exceeds_maximum(self, validation_agent):
        """Test that sales exceeding maximum are flagged."""
        df = pd.DataFrame({
            'sales': [100, 2000000, 300]  # 2M exceeds default max of 1M
        })
        results = QueryResult(
            data=df,
            row_count=3,
            execution_time=0.1,
            query=StructuredQuery(
                operation_type="sql",
                operation="SELECT * FROM sales",
                explanation="Test query"
            )
        )
        
        anomalies = validation_agent.check_business_rules(results)
        sales_bounds_anomalies = [a for a in anomalies if a.type == "sales_out_of_bounds"]
        assert len(sales_bounds_anomalies) >= 1
        assert sales_bounds_anomalies[0].severity == "warning"
    
    def test_check_business_rules_sales_below_minimum(self, validation_agent):
        """Test that sales below minimum are flagged."""
        df = pd.DataFrame({
            'sales': [100, -50, 300]  # -50 is below minimum of 0
        })
        results = QueryResult(
            data=df,
            row_count=3,
            execution_time=0.1,
            query=StructuredQuery(
                operation_type="sql",
                operation="SELECT * FROM sales",
                explanation="Test query"
            )
        )
        
        anomalies = validation_agent.check_business_rules(results)
        sales_bounds_anomalies = [a for a in anomalies if a.type == "sales_out_of_bounds"]
        assert len(sales_bounds_anomalies) >= 1
        assert sales_bounds_anomalies[0].severity == "error"
    
    def test_check_business_rules_empty_dataframe(self, validation_agent):
        """Test that empty dataframes don't cause errors."""
        df = pd.DataFrame()
        results = QueryResult(
            data=df,
            row_count=0,
            execution_time=0.1,
            query=StructuredQuery(
                operation_type="sql",
                operation="SELECT * FROM sales",
                explanation="Test query"
            )
        )
        
        anomalies = validation_agent.check_business_rules(results)
        assert len(anomalies) == 0
    
    def test_check_business_rules_custom_rules(self):
        """Test that custom business rules can be configured."""
        custom_rules = {
            'valid_categories': ['CustomCat1', 'CustomCat2'],
            'check_sales_order_match': False,
            'max_sales_per_order': 500,
            'min_sales_per_order': 10
        }
        agent = ValidationAgent(business_rules=custom_rules)
        
        df = pd.DataFrame({
            'category': ['CustomCat1', 'CustomCat2'],
            'sales': [100, 200]
        })
        results = QueryResult(
            data=df,
            row_count=2,
            execution_time=0.1,
            query=StructuredQuery(
                operation_type="sql",
                operation="SELECT * FROM sales",
                explanation="Test query"
            )
        )
        
        anomalies = agent.check_business_rules(results)
        assert len(anomalies) == 0
    
    def test_check_business_rules_multiple_violations(self, validation_agent):
        """Test that multiple business rule violations are all detected."""
        df = pd.DataFrame({
            'category': ['Electronics', 'InvalidCat', 'Food'],
            'sales': [100, -50, 2000000],
            'order_count': [1, 2, 0]
        })
        results = QueryResult(
            data=df,
            row_count=3,
            execution_time=0.1,
            query=StructuredQuery(
                operation_type="sql",
                operation="SELECT * FROM sales",
                explanation="Test query"
            )
        )
        
        anomalies = validation_agent.check_business_rules(results)
        # Should have: invalid category, sales below min, sales above max, sales without orders
        assert len(anomalies) >= 3
        
        anomaly_types = [a.type for a in anomalies]
        assert "invalid_category" in anomaly_types
        assert "sales_out_of_bounds" in anomaly_types
        assert "sales_order_mismatch" in anomaly_types


class TestBusinessRuleIntegration:
    """Tests for business rule integration with validate_results."""
    
    def test_validate_results_includes_business_rules(self, validation_agent, sample_query):
        """Test that validate_results includes business rule checks."""
        df = pd.DataFrame({
            'category': ['Electronics', 'InvalidCategory'],
            'sales': [100, 200]
        })
        results = QueryResult(
            data=df,
            row_count=2,
            execution_time=0.1,
            query=sample_query
        )
        
        validation_result = validation_agent.validate_results(results, sample_query)
        
        # Should fail due to invalid category
        assert validation_result.passed is False
        assert len(validation_result.anomalies) > 0
        
        # Check that invalid_category anomaly is present
        anomaly_types = [a.type for a in validation_result.anomalies]
        assert "invalid_category" in anomaly_types
