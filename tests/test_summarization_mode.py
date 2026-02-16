"""
Unit tests for SummarizationMode.

Tests summary generation, YoY growth calculation, trend identification,
and multi-dataset comparison.

**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7**
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from src.summarization_mode import SummarizationMode
from src.orchestrator import Orchestrator
from src.extraction_agent import ExtractionAgent
from src.llm_provider import LLMProvider, LLMResponse
from src.data_store import DataStore
from src.models import Response


@pytest.fixture
def mock_llm_provider():
    """Create mock LLM provider."""
    llm = Mock(spec=LLMProvider)
    llm.generate_with_cache.return_value = LLMResponse(
        content="This is a test summary with key metrics and insights.",
        tokens_used=100,
        model="test-model"
    )
    return llm


@pytest.fixture
def data_store_with_sales():
    """Create data store with sample sales data."""
    data_store = DataStore()
    
    # Create sample sales data
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Toys']
    
    data = {
        'date': dates[:365],
        'category': [categories[i % len(categories)] for i in range(365)],
        'sales': [100 + i * 2 for i in range(365)],
        'quantity': [10 + i for i in range(365)]
    }
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    data_store.tables['sales'] = df
    
    return data_store


@pytest.fixture
def data_store_with_yoy():
    """Create data store with multi-year data for YoY testing."""
    data_store = DataStore()
    
    # Create 2-year data
    dates_2022 = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
    dates_2023 = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    data_2022 = {
        'date': dates_2022,
        'category': ['Electronics'] * len(dates_2022),
        'sales': [100] * len(dates_2022),  # Constant for simplicity
        'quantity': [10] * len(dates_2022)
    }
    
    data_2023 = {
        'date': dates_2023,
        'category': ['Electronics'] * len(dates_2023),
        'sales': [150] * len(dates_2023),  # 50% increase
        'quantity': [15] * len(dates_2023)
    }
    
    df_2022 = pd.DataFrame(data_2022)
    df_2023 = pd.DataFrame(data_2023)
    
    df = pd.concat([df_2022, df_2023], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    
    data_store.tables['sales_yoy'] = df
    
    return data_store


@pytest.fixture
def extraction_agent(data_store_with_sales):
    """Create extraction agent with sample data."""
    return ExtractionAgent(data_store_with_sales)


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator."""
    return Mock(spec=Orchestrator)


@pytest.fixture
def summarization_mode(mock_orchestrator, extraction_agent, mock_llm_provider):
    """Create SummarizationMode instance."""
    return SummarizationMode(
        orchestrator=mock_orchestrator,
        extraction_agent=extraction_agent,
        llm_provider=mock_llm_provider
    )


class TestSummarizationMode:
    """Test suite for SummarizationMode."""
    
    def test_initialization(self, summarization_mode):
        """Test SummarizationMode initialization."""
        assert summarization_mode.orchestrator is not None
        assert summarization_mode.extraction_agent is not None
        assert summarization_mode.llm_provider is not None
    
    def test_generate_summary_basic(self, summarization_mode, mock_llm_provider):
        """
        Test basic summary generation.
        
        **Validates: Requirement 6.1**
        """
        # Generate summary
        response = summarization_mode.generate_summary()
        
        # Verify response
        assert isinstance(response, Response)
        assert response.answer is not None
        assert len(response.answer) > 0
        assert response.metadata['mode'] == 'summarization'
        assert response.metadata['table_name'] == 'sales'
        
        # Verify LLM was called
        mock_llm_provider.generate_with_cache.assert_called_once()
    
    def test_generate_summary_with_table_name(self, summarization_mode):
        """Test summary generation with specific table name."""
        response = summarization_mode.generate_summary(table_name='sales')
        
        assert response.metadata['table_name'] == 'sales'
    
    def test_generate_summary_no_tables(self, mock_orchestrator, mock_llm_provider):
        """Test error when no tables are loaded."""
        empty_data_store = DataStore()
        extraction_agent = ExtractionAgent(empty_data_store)
        
        summarization_mode = SummarizationMode(
            orchestrator=mock_orchestrator,
            extraction_agent=extraction_agent,
            llm_provider=mock_llm_provider
        )
        
        with pytest.raises(ValueError, match="No datasets loaded"):
            summarization_mode.generate_summary()
    
    def test_generate_summary_invalid_table(self, summarization_mode):
        """Test error when table not found."""
        with pytest.raises(ValueError, match="Table 'nonexistent' not found"):
            summarization_mode.generate_summary(table_name='nonexistent')
    
    def test_get_dataset_info(self, summarization_mode):
        """Test dataset information extraction."""
        info = summarization_mode._get_dataset_info('sales')
        
        assert info['table_name'] == 'sales'
        assert info['row_count'] == 365
        assert info['column_count'] == 4
        assert 'date' in info['columns']
        assert 'category' in info['columns']
        assert 'sales' in info['columns']
        assert info['date_range'] is not None
    
    def test_calculate_key_metrics(self, summarization_mode):
        """
        Test key metrics calculation.
        
        **Validates: Requirement 6.1**
        """
        metrics = summarization_mode._calculate_key_metrics('sales')
        
        # Verify numeric metrics are calculated
        assert 'sales_total' in metrics
        assert 'sales_average' in metrics
        assert 'sales_min' in metrics
        assert 'sales_max' in metrics
        
        assert metrics['sales_total'] > 0
        assert metrics['sales_average'] > 0
    
    def test_calculate_yoy_growth(self, mock_orchestrator, mock_llm_provider):
        """
        Test year-over-year growth calculation.
        
        **Validates: Requirement 6.2**
        """
        # Create data store with multi-year data
        data_store = DataStore()
        
        dates_2022 = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        dates_2023 = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        data = {
            'date': list(dates_2022) + list(dates_2023),
            'sales': [100] * len(dates_2022) + [150] * len(dates_2023)
        }
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        data_store.tables['sales_yoy'] = df
        
        extraction_agent = ExtractionAgent(data_store)
        summarization_mode = SummarizationMode(
            orchestrator=mock_orchestrator,
            extraction_agent=extraction_agent,
            llm_provider=mock_llm_provider
        )
        
        # Calculate YoY growth
        yoy_growth = summarization_mode._calculate_yoy_growth(df)
        
        assert yoy_growth is not None
        assert '2022_to_2023' in yoy_growth
        # 50% growth: (150-100)/100 * 100 = 50%
        assert abs(yoy_growth['2022_to_2023'] - 50.0) < 1.0
    
    def test_calculate_yoy_growth_no_date_column(self, summarization_mode):
        """Test YoY growth calculation with no date column."""
        # Create DataFrame without date column
        df = pd.DataFrame({
            'sales': [100, 200, 300],
            'category': ['A', 'B', 'C']
        })
        
        yoy_growth = summarization_mode._calculate_yoy_growth(df)
        
        # Should return None when no date column
        assert yoy_growth is None
    
    def test_identify_performers(self, summarization_mode):
        """
        Test identification of top and bottom performers.
        
        **Validates: Requirement 6.3**
        """
        performers = summarization_mode._identify_performers('sales')
        
        # Verify top performers
        assert 'top_performers' in performers
        assert 'category_column' in performers['top_performers']
        assert 'metric_column' in performers['top_performers']
        assert 'categories' in performers['top_performers']
        assert len(performers['top_performers']['categories']) > 0
        
        # Verify bottom performers
        assert 'bottom_performers' in performers
        assert len(performers['bottom_performers']['categories']) > 0
    
    def test_detect_trends(self, summarization_mode):
        """
        Test trend and anomaly detection.
        
        **Validates: Requirement 6.4**
        """
        trends = summarization_mode._detect_trends('sales')
        
        # Verify anomalies are detected
        assert 'anomalies' in trends
        assert isinstance(trends['anomalies'], list)
        
        # Verify time trends are detected
        assert 'time_trend' in trends
        assert 'direction' in trends['time_trend']
        assert trends['time_trend']['direction'] in ['increasing', 'decreasing', 'stable']
    
    def test_detect_trends_negative_values(self, mock_orchestrator, mock_llm_provider):
        """Test anomaly detection for negative values."""
        # Create data with negative sales
        data_store = DataStore()
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10),
            'sales': [100, 200, -50, 300, 400, -100, 500, 600, 700, 800],
            'category': ['A'] * 10
        })
        data_store.tables['sales_negative'] = df
        
        extraction_agent = ExtractionAgent(data_store)
        summarization_mode = SummarizationMode(
            orchestrator=mock_orchestrator,
            extraction_agent=extraction_agent,
            llm_provider=mock_llm_provider
        )
        
        trends = summarization_mode._detect_trends('sales_negative')
        
        # Should detect negative values
        anomalies = trends['anomalies']
        negative_anomaly = [a for a in anomalies if a['type'] == 'negative_values']
        assert len(negative_anomaly) > 0
        assert negative_anomaly[0]['count'] == 2
    
    def test_format_summary(self, summarization_mode, mock_llm_provider):
        """
        Test summary formatting with LLM.
        
        **Validates: Requirements 6.5, 6.6**
        """
        dataset_info = {'table_name': 'sales', 'row_count': 365}
        metrics = {'sales_total': 100000, 'sales_average': 274}
        performers = {}
        trends = {}
        
        summary = summarization_mode._format_summary(
            dataset_info=dataset_info,
            metrics=metrics,
            performers=performers,
            trends=trends,
            max_words=500
        )
        
        # Verify summary is generated
        assert summary is not None
        assert len(summary) > 0
        
        # Verify LLM was called
        mock_llm_provider.generate_with_cache.assert_called()
        
        # Verify word limit instruction was included
        call_args = mock_llm_provider.generate_with_cache.call_args
        prompt = call_args[1]['prompt']
        assert '500 words' in prompt
    
    def test_format_summary_fallback(self, summarization_mode, mock_llm_provider):
        """Test fallback summary generation when LLM fails."""
        # Make LLM raise an exception
        mock_llm_provider.generate_with_cache.side_effect = Exception("LLM error")
        
        dataset_info = {'table_name': 'sales', 'row_count': 365, 'column_count': 4}
        metrics = {'sales_total': 100000}
        performers = {
            'top_performers': {
                'categories': [
                    {'name': 'Electronics', 'value': 50000},
                    {'name': 'Clothing', 'value': 30000}
                ]
            }
        }
        trends = {
            'anomalies': [
                {'description': 'Found 5 outliers in sales'}
            ]
        }
        
        summary = summarization_mode._generate_fallback_summary(
            dataset_info, metrics, performers, trends
        )
        
        # Verify fallback summary is generated
        assert summary is not None
        assert 'sales' in summary.lower()
        assert '365' in summary
    
    def test_generate_comparative_summary(self, mock_orchestrator, mock_llm_provider):
        """
        Test multi-dataset comparison.
        
        **Validates: Requirement 6.7**
        """
        # Create data store with multiple tables
        data_store = DataStore()
        
        df1 = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=100),
            'sales': [100 + i for i in range(100)],
            'category': ['A'] * 100
        })
        
        df2 = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=100),
            'sales': [200 + i * 2 for i in range(100)],
            'category': ['B'] * 100
        })
        
        data_store.tables['sales_2022'] = df1
        data_store.tables['sales_2023'] = df2
        
        extraction_agent = ExtractionAgent(data_store)
        summarization_mode = SummarizationMode(
            orchestrator=mock_orchestrator,
            extraction_agent=extraction_agent,
            llm_provider=mock_llm_provider
        )
        
        # Generate comparative summary
        response = summarization_mode.generate_comparative_summary(
            table_names=['sales_2022', 'sales_2023']
        )
        
        # Verify response
        assert isinstance(response, Response)
        assert response.answer is not None
        assert response.metadata['comparison'] is True
        assert len(response.metadata['table_names']) == 2
        assert 'sales_2022' in response.metadata['table_names']
        assert 'sales_2023' in response.metadata['table_names']
    
    def test_generate_comparative_summary_insufficient_tables(self, summarization_mode):
        """Test error when fewer than 2 tables provided."""
        with pytest.raises(ValueError, match="at least 2 tables"):
            summarization_mode.generate_comparative_summary(table_names=['sales'])
    
    def test_generate_comparative_summary_invalid_table(self, summarization_mode):
        """Test error when table not found in comparison."""
        with pytest.raises(ValueError, match="not found"):
            summarization_mode.generate_comparative_summary(
                table_names=['sales', 'nonexistent']
            )
    
    def test_summary_includes_numerical_evidence(self, summarization_mode):
        """
        Test that summary includes specific numerical evidence.
        
        **Validates: Requirement 6.6**
        """
        response = summarization_mode.generate_summary()
        
        # Verify metadata contains numerical metrics
        assert 'metrics' in response.metadata
        metrics = response.metadata['metrics']
        
        # Should have numeric values
        numeric_metrics = [v for v in metrics.values() if isinstance(v, (int, float))]
        assert len(numeric_metrics) > 0
    
    def test_summary_word_limit(self, summarization_mode, mock_llm_provider):
        """
        Test that summary respects word limit.
        
        **Validates: Requirement 6.5**
        """
        # Set a short word limit
        max_words = 100
        
        response = summarization_mode.generate_summary(max_words=max_words)
        
        # Verify max_words is passed to metadata
        assert response.metadata['max_words'] == max_words
        
        # Verify LLM prompt includes word limit
        call_args = mock_llm_provider.generate_with_cache.call_args
        prompt = call_args[1]['prompt']
        assert str(max_words) in prompt
    
    def test_get_date_range_with_datetime_column(self, summarization_mode):
        """Test date range extraction with datetime column."""
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10),
            'sales': [100] * 10
        })
        
        date_range = summarization_mode._get_date_range(df)
        
        assert date_range is not None
        assert 'start_date' in date_range
        assert 'end_date' in date_range
        assert 'column' in date_range
        assert '2023-01-01' in date_range['start_date']
    
    def test_get_date_range_no_date_column(self, summarization_mode):
        """Test date range extraction with no date column."""
        df = pd.DataFrame({
            'sales': [100, 200, 300],
            'category': ['A', 'B', 'C']
        })
        
        date_range = summarization_mode._get_date_range(df)
        
        # Should return None when no date column
        assert date_range is None
    
    def test_detect_outliers(self, mock_orchestrator, mock_llm_provider):
        """Test outlier detection using IQR method."""
        # Create data with outliers
        data_store = DataStore()
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=20),
            'sales': [100] * 18 + [1000, 2000],  # Last 2 are outliers
            'category': ['A'] * 20
        })
        data_store.tables['sales_outliers'] = df
        
        extraction_agent = ExtractionAgent(data_store)
        summarization_mode = SummarizationMode(
            orchestrator=mock_orchestrator,
            extraction_agent=extraction_agent,
            llm_provider=mock_llm_provider
        )
        
        trends = summarization_mode._detect_trends('sales_outliers')
        
        # Should detect outliers
        anomalies = trends['anomalies']
        outlier_anomaly = [a for a in anomalies if a['type'] == 'outliers']
        assert len(outlier_anomaly) > 0
        assert outlier_anomaly[0]['count'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
