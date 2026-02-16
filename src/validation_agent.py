"""Validation Agent module for verifying query results.

This module implements the ValidationAgent responsible for verifying
query results for accuracy, consistency, and data quality.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
"""

import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from src.models import QueryResult, StructuredQuery, ValidationResult, Anomaly

logger = logging.getLogger(__name__)


class ValidationAgent:
    """
    Validation Agent responsible for verifying query results.
    
    This agent:
    - Verifies data types match expected schema
    - Checks mathematical consistency in aggregations
    - Validates empty results
    - Detects anomalies (negative sales, invalid dates)
    - Flags data anomalies with severity levels
    - Applies configurable business rules (category validation, sales/order matching)
    """
    
    def __init__(self, business_rules: Dict[str, Any] = None):
        """
        Initialize the Validation Agent.
        
        Args:
            business_rules: Optional dictionary of configurable business rules.
                           If not provided, uses default rules.
        """
        self.business_rules = business_rules or self._get_default_business_rules()
        logger.info("ValidationAgent initialized with business rules")
    
    def validate_results(self, results: QueryResult, query: StructuredQuery) -> ValidationResult:
        """
        Validate query results for correctness.
        
        This method performs multiple validation checks:
        1. Data type verification against schema
        2. Mathematical consistency checks for aggregations
        3. Empty result validation
        4. Anomaly detection
        5. Business rule validation
        
        Args:
            results: Query results from Extraction Agent
            query: Original structured query
            
        Returns:
            ValidationResult with pass/fail status and any issues found
            
        **Validates: Requirements 5.1, 5.2, 5.3, 5.5**
        """
        if not isinstance(results, QueryResult):
            raise ValueError("results must be a QueryResult instance")
        if not isinstance(query, StructuredQuery):
            raise ValueError("query must be a StructuredQuery instance")
        
        logger.info(f"Validating results for query: {query.explanation}")
        
        issues: List[str] = []
        anomalies: List[Anomaly] = []
        
        # Check 1: Validate data types against schema (Requirement 5.1)
        type_issues = self._validate_data_types(results.data)
        issues.extend(type_issues)
        
        # Check 2: Mathematical consistency checks (Requirement 5.2)
        math_issues, math_anomalies = self._check_mathematical_consistency(results.data)
        issues.extend(math_issues)
        anomalies.extend(math_anomalies)
        
        # Check 3: Empty result validation (Requirement 5.3)
        empty_issues = self._validate_empty_results(results, query)
        issues.extend(empty_issues)
        
        # Check 4: Anomaly detection (Requirement 5.4)
        anomaly_issues, detected_anomalies = self._detect_anomalies(results.data)
        issues.extend(anomaly_issues)
        anomalies.extend(detected_anomalies)
        
        # Check 5: Business rule validation (Requirement 5.5)
        business_rule_anomalies = self.check_business_rules(results)
        anomalies.extend(business_rule_anomalies)
        
        # Determine if validation passed
        # Validation fails if there are any critical issues
        # Anomalies with severity "warning" don't fail validation
        critical_issues = [issue for issue in issues if "error" in issue.lower()]
        error_anomalies = [a for a in anomalies if a.severity == "error"]
        
        passed = len(critical_issues) == 0 and len(error_anomalies) == 0
        
        # Calculate confidence score
        confidence = self._calculate_confidence(issues, anomalies)
        
        result = ValidationResult(
            passed=passed,
            issues=issues,
            anomalies=anomalies,
            confidence=confidence
        )
        
        if passed:
            logger.info(
                f"Validation passed with confidence {confidence:.2f}. "
                f"Warnings: {len(issues)}, Anomalies: {len(anomalies)}"
            )
        else:
            logger.warning(
                f"Validation failed. Issues: {len(issues)}, "
                f"Error anomalies: {len(error_anomalies)}"
            )
        
        return result
    
    def _validate_data_types(self, df: pd.DataFrame) -> List[str]:
        """
        Verify data types match expected schema.
        
        Checks for:
        - Numeric columns containing non-numeric values
        - Date columns with invalid dates
        - Unexpected data type conversions
        
        Args:
            df: DataFrame to validate
            
        Returns:
            List of data type issues found
            
        **Validates: Requirement 5.1**
        """
        issues = []
        
        if df.empty:
            return issues
        
        for col in df.columns:
            dtype = df[col].dtype
            
            # Check for object dtype that might contain mixed types
            if dtype == 'object':
                # Check if column should be numeric
                try:
                    pd.to_numeric(df[col], errors='raise')
                    issues.append(
                        f"Data type validation failure: Column '{col}' has object dtype but contains only numeric values. "
                        f"Consider converting to numeric type for better performance and accuracy."
                    )
                except (ValueError, TypeError):
                    # Mixed types or truly non-numeric, which is fine
                    pass
            
            # Check for datetime columns that might have invalid dates
            if pd.api.types.is_datetime64_any_dtype(dtype):
                if df[col].isna().any():
                    na_count = df[col].isna().sum()
                    na_rows = df[df[col].isna()].index.tolist()
                    issues.append(
                        f"Data type validation failure: Column '{col}' is datetime type but contains {na_count} invalid/missing dates. "
                        f"Affected rows: {na_rows[:10]}" + (" (showing first 10)" if len(na_rows) > 10 else "")
                    )
        
        logger.debug(f"Data type validation found {len(issues)} issues")
        return issues

    
    def _check_mathematical_consistency(self, df: pd.DataFrame) -> tuple[List[str], List[Anomaly]]:
        """
        Check for mathematical consistency in aggregations.
        
        Validates:
        - Sum consistency (e.g., total = subtotal + tax)
        - Non-negative values for metrics that shouldn't be negative
        - Reasonable value ranges
        
        Args:
            df: DataFrame to check
            
        Returns:
            Tuple of (issues list, anomalies list)
            
        **Validates: Requirement 5.2**
        """
        issues = []
        anomalies = []
        
        if df.empty:
            return issues, anomalies
        
        # Check for negative values in common sales/financial columns
        negative_check_columns = ['sales', 'revenue', 'price', 'quantity', 'amount', 'total']
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check if this is a column that shouldn't have negative values
            if any(keyword in col_lower for keyword in negative_check_columns):
                if pd.api.types.is_numeric_dtype(df[col]):
                    negative_mask = df[col] < 0
                    if negative_mask.any():
                        negative_rows = df[negative_mask].index.tolist()
                        anomalies.append(Anomaly(
                            type="negative_value",
                            description=f"Column '{col}' contains {negative_mask.sum()} negative values",
                            severity="warning",
                            affected_rows=negative_rows[:10]  # Limit to first 10 rows
                        ))
        
        # Check for sum consistency patterns
        # Look for common patterns like: total = subtotal + tax, or total = price * quantity
        if 'total' in df.columns and 'subtotal' in df.columns and 'tax' in df.columns:
            if all(pd.api.types.is_numeric_dtype(df[col]) for col in ['total', 'subtotal', 'tax']):
                # Allow small floating point differences (0.01)
                diff = abs(df['total'] - (df['subtotal'] + df['tax']))
                inconsistent_mask = diff > 0.01
                if inconsistent_mask.any():
                    inconsistent_rows = df[inconsistent_mask].index.tolist()
                    max_diff = diff[inconsistent_mask].max()
                    issues.append(
                        f"Mathematical inconsistency validation failure: {inconsistent_mask.sum()} rows where "
                        f"total != subtotal + tax (tolerance: 0.01). Maximum difference: {max_diff:.2f}. "
                        f"Affected rows: {inconsistent_rows[:10]}" + (" (showing first 10)" if len(inconsistent_rows) > 10 else "")
                    )
                    anomalies.append(Anomaly(
                        type="sum_inconsistency",
                        description=f"Total does not equal subtotal + tax. Maximum difference: {max_diff:.2f}",
                        severity="error",
                        affected_rows=inconsistent_rows[:10]
                    ))
        
        # Check for price * quantity = amount patterns
        if all(col in df.columns for col in ['price', 'quantity', 'amount']):
            if all(pd.api.types.is_numeric_dtype(df[col]) for col in ['price', 'quantity', 'amount']):
                diff = abs(df['amount'] - (df['price'] * df['quantity']))
                inconsistent_mask = diff > 0.01
                if inconsistent_mask.any():
                    inconsistent_rows = df[inconsistent_mask].index.tolist()
                    max_diff = diff[inconsistent_mask].max()
                    issues.append(
                        f"Mathematical inconsistency validation failure: {inconsistent_mask.sum()} rows where "
                        f"amount != price * quantity (tolerance: 0.01). Maximum difference: {max_diff:.2f}. "
                        f"Affected rows: {inconsistent_rows[:10]}" + (" (showing first 10)" if len(inconsistent_rows) > 10 else "")
                    )
                    anomalies.append(Anomaly(
                        type="multiplication_inconsistency",
                        description=f"Amount does not equal price * quantity. Maximum difference: {max_diff:.2f}",
                        severity="error",
                        affected_rows=inconsistent_rows[:10]
                    ))
        
        logger.debug(
            f"Mathematical consistency check found {len(issues)} issues "
            f"and {len(anomalies)} anomalies"
        )
        return issues, anomalies
    
    def _validate_empty_results(self, results: QueryResult, query: StructuredQuery) -> List[str]:
        """
        Validate empty results to ensure query logic is correct.
        
        When results are empty, this method checks if:
        - The query syntax is valid
        - The query is logically sound
        - Empty result is expected or indicates an issue
        
        Args:
            results: QueryResult to validate
            query: Original structured query
            
        Returns:
            List of issues found
            
        **Validates: Requirement 5.3**
        """
        issues = []
        
        if results.row_count == 0:
            # Empty result detected
            logger.info("Empty result set detected, validating query logic")
            
            # Check if query has overly restrictive filters
            query_lower = query.operation.lower()
            
            # Look for potential issues in SQL queries
            if query.operation_type == "sql":
                # Check for WHERE clauses that might be too restrictive
                if "where" in query_lower:
                    issues.append(
                        f"Empty result validation failure: Query contains WHERE clause that may be too restrictive. "
                        f"Query: {query.operation[:100]}... "
                        f"Suggestion: Review filter conditions to ensure they match existing data."
                    )
                
                # Check for JOINs that might eliminate all rows
                if "join" in query_lower:
                    issues.append(
                        f"Empty result validation failure: Query contains JOIN that may eliminate all rows. "
                        f"Query: {query.operation[:100]}... "
                        f"Suggestion: Verify join conditions match existing data or consider using LEFT JOIN."
                    )
                
                # Check for GROUP BY with HAVING that might filter out all groups
                if "having" in query_lower:
                    issues.append(
                        f"Empty result validation failure: Query contains HAVING clause that may filter out all groups. "
                        f"Query: {query.operation[:100]}... "
                        f"Suggestion: Review aggregation filters to ensure they are not too restrictive."
                    )
            
            # If no specific issues found, add general warning
            if not issues:
                issues.append(
                    f"Empty result validation warning: Query returned no rows. "
                    f"Query type: {query.operation_type}. "
                    f"This may be expected or indicate an issue with query logic. "
                    f"Suggestion: Verify the query matches the available data."
                )
        
        logger.debug(f"Empty result validation found {len(issues)} issues")
        return issues
    
    def _detect_anomalies(self, df: pd.DataFrame) -> tuple[List[str], List[Anomaly]]:
        """
        Detect anomalies in the data.
        
        Detects:
        - Negative sales values
        - Invalid dates (NaT, out-of-range dates)
        - Other data quality issues
        
        Args:
            df: DataFrame to check for anomalies
            
        Returns:
            Tuple of (issues list, anomalies list)
            
        **Validates: Requirement 5.4**
        """
        issues = []
        anomalies = []
        
        if df.empty:
            return issues, anomalies
        
        # Detect negative sales values
        sales_columns = ['sales', 'revenue', 'amount', 'total', 'price']
        for col in df.columns:
            col_lower = col.lower()
            
            # Check for negative values in sales-related columns
            if any(keyword in col_lower for keyword in sales_columns):
                if pd.api.types.is_numeric_dtype(df[col]):
                    negative_mask = df[col] < 0
                    if negative_mask.any():
                        negative_count = negative_mask.sum()
                        negative_rows = df[negative_mask].index.tolist()
                        negative_values = df.loc[negative_mask, col].tolist()[:5]  # Get first 5 values
                        
                        # Determine severity: error for sales, warning for others
                        severity = "error" if "sales" in col_lower or "revenue" in col_lower else "warning"
                        
                        anomalies.append(Anomaly(
                            type="negative_sales",
                            description=(
                                f"Column '{col}' contains {negative_count} negative values. "
                                f"Sample values: {negative_values}. "
                                f"Affected rows: {negative_rows[:10]}" + (" (showing first 10)" if len(negative_rows) > 10 else "")
                            ),
                            severity=severity,
                            affected_rows=negative_rows[:10]  # Limit to first 10 rows
                        ))
                        
                        issues.append(
                            f"Anomaly validation failure ({severity}): Column '{col}' contains {negative_count} negative values. "
                            f"Sample values: {negative_values}. "
                            f"Affected rows: {negative_rows[:10]}" + (" (showing first 10)" if len(negative_rows) > 10 else "")
                        )
        
        # Detect invalid dates
        date_columns = ['date', 'timestamp', 'created_at', 'updated_at', 'order_date', 'sale_date']
        for col in df.columns:
            col_lower = col.lower()
            
            # Check for date-related columns
            if any(keyword in col_lower for keyword in date_columns) or pd.api.types.is_datetime64_any_dtype(df[col]):
                # Check for NaT (Not a Time) values
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    nat_mask = pd.isna(df[col])
                    if nat_mask.any():
                        nat_count = nat_mask.sum()
                        nat_rows = df[nat_mask].index.tolist()
                        
                        anomalies.append(Anomaly(
                            type="invalid_date",
                            description=(
                                f"Column '{col}' contains {nat_count} invalid/missing dates (NaT). "
                                f"Affected rows: {nat_rows[:10]}" + (" (showing first 10)" if len(nat_rows) > 10 else "")
                            ),
                            severity="warning",
                            affected_rows=nat_rows[:10]
                        ))
                        
                        issues.append(
                            f"Anomaly validation failure (warning): Column '{col}' contains {nat_count} invalid/missing dates (NaT). "
                            f"Affected rows: {nat_rows[:10]}" + (" (showing first 10)" if len(nat_rows) > 10 else "")
                        )
                    
                    # Check for dates outside reasonable range (e.g., year < 1900 or > 2100)
                    valid_dates = df[col].dropna()
                    if not valid_dates.empty:
                        min_year = pd.Timestamp('1900-01-01')
                        max_year = pd.Timestamp('2100-12-31')
                        
                        out_of_range_mask = (valid_dates < min_year) | (valid_dates > max_year)
                        if out_of_range_mask.any():
                            out_of_range_count = out_of_range_mask.sum()
                            out_of_range_rows = valid_dates[out_of_range_mask].index.tolist()
                            out_of_range_values = valid_dates[out_of_range_mask].tolist()[:5]
                            
                            anomalies.append(Anomaly(
                                type="invalid_date",
                                description=(
                                    f"Column '{col}' contains {out_of_range_count} dates outside reasonable range (1900-2100). "
                                    f"Sample values: {[str(d) for d in out_of_range_values]}. "
                                    f"Affected rows: {out_of_range_rows[:10]}" + (" (showing first 10)" if len(out_of_range_rows) > 10 else "")
                                ),
                                severity="warning",
                                affected_rows=out_of_range_rows[:10]
                            ))
                            
                            issues.append(
                                f"Anomaly validation failure (warning): Column '{col}' contains {out_of_range_count} dates outside reasonable range (1900-2100). "
                                f"Sample values: {[str(d) for d in out_of_range_values]}. "
                                f"Affected rows: {out_of_range_rows[:10]}" + (" (showing first 10)" if len(out_of_range_rows) > 10 else "")
                            )
        
        logger.debug(
            f"Anomaly detection found {len(issues)} issues and {len(anomalies)} anomalies"
        )
        return issues, anomalies
    
    def _calculate_confidence(self, issues: List[str], anomalies: List[Anomaly]) -> float:
        """
        Calculate confidence score for validation.
        
        Confidence decreases based on:
        - Number of issues found
        - Severity of anomalies
        
        Args:
            issues: List of validation issues
            anomalies: List of detected anomalies
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 1.0
        
        # Decrease confidence for each issue (0.1 per issue, min 0.0)
        confidence -= len(issues) * 0.1
        
        # Decrease confidence for anomalies based on severity
        for anomaly in anomalies:
            if anomaly.severity == "error":
                confidence -= 0.2
            elif anomaly.severity == "warning":
                confidence -= 0.05
        
        # Ensure confidence is between 0.0 and 1.0
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    def _get_default_business_rules(self) -> Dict[str, Any]:
        """
        Get default business rules configuration.

        Returns:
            Dictionary containing default business rules
        """
        return {
            'valid_categories': [
                'Electronics', 'Clothing', 'Food', 'Home', 'Sports',
                'Books', 'Toys', 'Beauty', 'Automotive', 'Garden'
            ],
            'check_sales_order_match': True,
            'max_sales_per_order': 1000000,  # Maximum reasonable sales amount per order
            'min_sales_per_order': 0,  # Minimum sales amount
        }

    def check_business_rules(self, results: QueryResult) -> List[Anomaly]:
        """
        Check results against configured business rules.

        Validates:
        - Sales totals match order counts (if applicable)
        - Category values are in the known valid list
        - Other configurable business rules

        Args:
            results: Query results to validate

        Returns:
            List of anomalies found during business rule validation

        **Validates: Requirement 5.5**
        """
        anomalies = []
        df = results.data

        if df.empty:
            return anomalies

        # Rule 1: Validate category values against known list
        if 'category' in df.columns:
            valid_categories = self.business_rules.get('valid_categories', [])
            if valid_categories:
                invalid_mask = ~df['category'].isin(valid_categories)
                if invalid_mask.any():
                    invalid_count = invalid_mask.sum()
                    invalid_rows = df[invalid_mask].index.tolist()
                    invalid_values = df.loc[invalid_mask, 'category'].unique().tolist()

                    anomalies.append(Anomaly(
                        type="invalid_category",
                        description=(
                            f"Found {invalid_count} rows with invalid category values. "
                            f"Invalid categories: {invalid_values[:5]}"
                        ),
                        severity="error",
                        affected_rows=invalid_rows[:10]
                    ))
                    logger.warning(f"Business rule violation: {invalid_count} invalid categories found")

        # Rule 2: Check sales totals match order counts
        # This checks if the relationship between sales and orders is reasonable
        if self.business_rules.get('check_sales_order_match', False):
            if 'sales' in df.columns and 'order_count' in df.columns:
                # Check if sales and order_count have a reasonable relationship
                # Sales should be positive when order_count is positive
                if pd.api.types.is_numeric_dtype(df['sales']) and pd.api.types.is_numeric_dtype(df['order_count']):
                    # Check for cases where order_count > 0 but sales <= 0
                    mismatch_mask = (df['order_count'] > 0) & (df['sales'] <= 0)
                    if mismatch_mask.any():
                        mismatch_count = mismatch_mask.sum()
                        mismatch_rows = df[mismatch_mask].index.tolist()

                        anomalies.append(Anomaly(
                            type="sales_order_mismatch",
                            description=(
                                f"Found {mismatch_count} rows where order_count > 0 but sales <= 0. "
                                f"Sales should be positive when orders exist."
                            ),
                            severity="error",
                            affected_rows=mismatch_rows[:10]
                        ))
                        logger.warning(f"Business rule violation: {mismatch_count} sales/order mismatches found")

                    # Check for cases where sales > 0 but order_count <= 0
                    reverse_mismatch_mask = (df['sales'] > 0) & (df['order_count'] <= 0)
                    if reverse_mismatch_mask.any():
                        reverse_count = reverse_mismatch_mask.sum()
                        reverse_rows = df[reverse_mismatch_mask].index.tolist()

                        anomalies.append(Anomaly(
                            type="sales_order_mismatch",
                            description=(
                                f"Found {reverse_count} rows where sales > 0 but order_count <= 0. "
                                f"Orders should exist when sales are recorded."
                            ),
                            severity="error",
                            affected_rows=reverse_rows[:10]
                        ))
                        logger.warning(f"Business rule violation: {reverse_count} reverse sales/order mismatches found")

        # Rule 3: Check sales amounts are within reasonable bounds
        if 'sales' in df.columns:
            if pd.api.types.is_numeric_dtype(df['sales']):
                max_sales = self.business_rules.get('max_sales_per_order', float('inf'))
                min_sales = self.business_rules.get('min_sales_per_order', 0)

                # Check for sales exceeding maximum
                if max_sales < float('inf'):
                    over_max_mask = df['sales'] > max_sales
                    if over_max_mask.any():
                        over_count = over_max_mask.sum()
                        over_rows = df[over_max_mask].index.tolist()

                        anomalies.append(Anomaly(
                            type="sales_out_of_bounds",
                            description=(
                                f"Found {over_count} rows with sales exceeding maximum allowed "
                                f"value of {max_sales}"
                            ),
                            severity="warning",
                            affected_rows=over_rows[:10]
                        ))
                        logger.warning(f"Business rule violation: {over_count} sales values exceed maximum")

                # Check for sales below minimum
                under_min_mask = df['sales'] < min_sales
                if under_min_mask.any():
                    under_count = under_min_mask.sum()
                    under_rows = df[under_min_mask].index.tolist()

                    anomalies.append(Anomaly(
                        type="sales_out_of_bounds",
                        description=(
                            f"Found {under_count} rows with sales below minimum allowed "
                            f"value of {min_sales}"
                        ),
                        severity="error",
                        affected_rows=under_rows[:10]
                    ))
                    logger.warning(f"Business rule violation: {under_count} sales values below minimum")

        logger.info(f"Business rule validation found {len(anomalies)} anomalies")
        return anomalies

    def _get_default_business_rules(self) -> Dict[str, Any]:
        """
        Get default business rules configuration.
        
        Returns:
            Dictionary containing default business rules
        """
        return {
            'valid_categories': [
                'Electronics', 'Clothing', 'Food', 'Home', 'Sports',
                'Books', 'Toys', 'Beauty', 'Automotive', 'Garden'
            ],
            'check_sales_order_match': True,
            'max_sales_per_order': 1000000,  # Maximum reasonable sales amount per order
            'min_sales_per_order': 0,  # Minimum sales amount
        }
    
    def check_business_rules(self, results: QueryResult) -> List[Anomaly]:
        """
        Check results against configured business rules.
        
        Validates:
        - Sales totals match order counts (if applicable)
        - Category values are in the known valid list
        - Other configurable business rules
        
        Args:
            results: Query results to validate
            
        Returns:
            List of anomalies found during business rule validation
            
        **Validates: Requirement 5.5**
        """
        anomalies = []
        df = results.data
        
        if df.empty:
            return anomalies
        
        # Rule 1: Validate category values against known list
        if 'category' in df.columns:
            valid_categories = self.business_rules.get('valid_categories', [])
            if valid_categories:
                invalid_mask = ~df['category'].isin(valid_categories)
                if invalid_mask.any():
                    invalid_count = invalid_mask.sum()
                    invalid_rows = df[invalid_mask].index.tolist()
                    invalid_values = df.loc[invalid_mask, 'category'].unique().tolist()

                    anomalies.append(Anomaly(
                        type="invalid_category",
                        description=(
                            f"Business rule validation failure: Found {invalid_count} rows with invalid category values. "
                            f"Invalid categories: {invalid_values[:5]}" + (" (showing first 5)" if len(invalid_values) > 5 else "") + ". "
                            f"Valid categories are: {valid_categories}. "
                            f"Affected rows: {invalid_rows[:10]}" + (" (showing first 10)" if len(invalid_rows) > 10 else "")
                        ),
                        severity="error",
                        affected_rows=invalid_rows[:10]
                    ))
                    logger.warning(f"Business rule violation: {invalid_count} invalid categories found")
        
        # Rule 2: Check sales totals match order counts
        # This checks if the relationship between sales and orders is reasonable
        if self.business_rules.get('check_sales_order_match', False):
            if 'sales' in df.columns and 'order_count' in df.columns:
                # Check if sales and order_count have a reasonable relationship
                # Sales should be positive when order_count is positive
                if pd.api.types.is_numeric_dtype(df['sales']) and pd.api.types.is_numeric_dtype(df['order_count']):
                    # Check for cases where order_count > 0 but sales <= 0
                    mismatch_mask = (df['order_count'] > 0) & (df['sales'] <= 0)
                    if mismatch_mask.any():
                        mismatch_count = mismatch_mask.sum()
                        mismatch_rows = df[mismatch_mask].index.tolist()
                        sample_data = df.loc[mismatch_mask, ['order_count', 'sales']].head(3).to_dict('records')

                        anomalies.append(Anomaly(
                            type="sales_order_mismatch",
                            description=(
                                f"Business rule validation failure: Found {mismatch_count} rows where order_count > 0 but sales <= 0. "
                                f"Sales should be positive when orders exist. "
                                f"Sample data: {sample_data}. "
                                f"Affected rows: {mismatch_rows[:10]}" + (" (showing first 10)" if len(mismatch_rows) > 10 else "")
                            ),
                            severity="error",
                            affected_rows=mismatch_rows[:10]
                        ))
                        logger.warning(f"Business rule violation: {mismatch_count} sales/order mismatches found")
                    
                    # Check for cases where sales > 0 but order_count <= 0
                    reverse_mismatch_mask = (df['sales'] > 0) & (df['order_count'] <= 0)
                    if reverse_mismatch_mask.any():
                        reverse_count = reverse_mismatch_mask.sum()
                        reverse_rows = df[reverse_mismatch_mask].index.tolist()
                        sample_data = df.loc[reverse_mismatch_mask, ['sales', 'order_count']].head(3).to_dict('records')

                        anomalies.append(Anomaly(
                            type="sales_order_mismatch",
                            description=(
                                f"Business rule validation failure: Found {reverse_count} rows where sales > 0 but order_count <= 0. "
                                f"Orders should exist when sales are recorded. "
                                f"Sample data: {sample_data}. "
                                f"Affected rows: {reverse_rows[:10]}" + (" (showing first 10)" if len(reverse_rows) > 10 else "")
                            ),
                            severity="error",
                            affected_rows=reverse_rows[:10]
                        ))
                        logger.warning(f"Business rule violation: {reverse_count} reverse sales/order mismatches found")
        
        # Rule 3: Check sales amounts are within reasonable bounds
        if 'sales' in df.columns:
            if pd.api.types.is_numeric_dtype(df['sales']):
                max_sales = self.business_rules.get('max_sales_per_order', float('inf'))
                min_sales = self.business_rules.get('min_sales_per_order', 0)
                
                # Check for sales exceeding maximum
                if max_sales < float('inf'):
                    over_max_mask = df['sales'] > max_sales
                    if over_max_mask.any():
                        over_count = over_max_mask.sum()
                        over_rows = df[over_max_mask].index.tolist()
                        max_value = df.loc[over_max_mask, 'sales'].max()

                        anomalies.append(Anomaly(
                            type="sales_out_of_bounds",
                            description=(
                                f"Business rule validation failure: Found {over_count} rows with sales exceeding maximum allowed "
                                f"value of {max_sales}. Maximum value found: {max_value}. "
                                f"Affected rows: {over_rows[:10]}" + (" (showing first 10)" if len(over_rows) > 10 else "")
                            ),
                            severity="warning",
                            affected_rows=over_rows[:10]
                        ))
                        logger.warning(f"Business rule violation: {over_count} sales values exceed maximum")
                
                # Check for sales below minimum
                under_min_mask = df['sales'] < min_sales
                if under_min_mask.any():
                    under_count = under_min_mask.sum()
                    under_rows = df[under_min_mask].index.tolist()
                    min_value = df.loc[under_min_mask, 'sales'].min()

                    anomalies.append(Anomaly(
                        type="sales_out_of_bounds",
                        description=(
                            f"Business rule validation failure: Found {under_count} rows with sales below minimum allowed "
                            f"value of {min_sales}. Minimum value found: {min_value}. "
                            f"Affected rows: {under_rows[:10]}" + (" (showing first 10)" if len(under_rows) > 10 else "")
                        ),
                        severity="error",
                        affected_rows=under_rows[:10]
                    ))
                    logger.warning(f"Business rule violation: {under_count} sales values below minimum")
        
        logger.info(f"Business rule validation found {len(anomalies)} anomalies")
        return anomalies
