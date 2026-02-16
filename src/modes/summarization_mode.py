"""
Summarization Mode for generating dataset summaries.

This module implements the SummarizationMode that analyzes loaded datasets
and generates comprehensive summaries with key metrics, trends, and anomalies.

**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7**
"""

import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.core.orchestrator import Orchestrator
from src.agents.extraction_agent import ExtractionAgent
from src.llm.llm_provider import LLMProvider
from src.llm.prompt_templates import PromptTemplates
from src.core.models import Response, DataSchema

logger = logging.getLogger(__name__)


class SummarizationMode:
    """
    Summarization Mode for generating dataset summaries.
    
    This class:
    - Analyzes loaded datasets to identify key metrics
    - Calculates year-over-year growth rates
    - Identifies top-performing and underperforming categories
    - Detects significant trends and anomalies
    - Formats summaries in natural language (max 500 words)
    - Supports multi-dataset comparison
    """
    
    def __init__(
        self,
        orchestrator: Orchestrator,
        extraction_agent: ExtractionAgent,
        llm_provider: LLMProvider
    ):
        """
        Initialize SummarizationMode.
        
        Args:
            orchestrator: Orchestrator for agent coordination
            extraction_agent: ExtractionAgent for data analysis
            llm_provider: LLM provider for natural language formatting
        """
        self.orchestrator = orchestrator
        self.extraction_agent = extraction_agent
        self.llm_provider = llm_provider
        
        logger.info("SummarizationMode initialized")
    
    def generate_summary(
        self,
        table_name: Optional[str] = None,
        max_words: int = 500
    ) -> Response:
        """
        Generate a comprehensive summary of the loaded dataset.
        
        This method:
        1. Analyzes the dataset to identify key metrics
        2. Calculates year-over-year growth rates
        3. Identifies top-performing and underperforming categories
        4. Highlights significant trends and anomalies
        5. Formats the summary in natural language (max 500 words)
        6. Includes specific numerical evidence
        
        Args:
            table_name: Name of table to summarize (uses first table if None)
            max_words: Maximum words for summary (default 500)
            
        Returns:
            Response object with summary and metadata
            
        Raises:
            ValueError: If no tables are loaded or table not found
            
        **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6**
        """
        logger.info(f"Generating summary for table: {table_name or 'default'}")
        
        # Get available tables
        tables = self.extraction_agent.data_store.list_tables()
        if not tables:
            raise ValueError("No datasets loaded. Please load a dataset first.")
        
        # Use specified table or first available table
        if table_name is None:
            table_name = tables[0]
            logger.info(f"No table specified, using first table: {table_name}")
        elif table_name not in tables:
            raise ValueError(
                f"Table '{table_name}' not found. "
                f"Available tables: {', '.join(tables)}"
            )
        
        # Get dataset information
        dataset_info = self._get_dataset_info(table_name)
        
        # Calculate key metrics
        metrics = self._calculate_key_metrics(table_name)
        
        # Identify top and bottom performers
        performers = self._identify_performers(table_name)
        
        # Detect trends and anomalies
        trends = self._detect_trends(table_name)
        
        # Format summary using LLM
        summary = self._format_summary(
            dataset_info=dataset_info,
            metrics=metrics,
            performers=performers,
            trends=trends,
            max_words=max_words
        )
        
        # Create response
        response = Response(
            answer=summary,
            metadata={
                "mode": "summarization",
                "table_name": table_name,
                "dataset_info": dataset_info,
                "metrics": metrics,
                "performers": performers,
                "trends": trends,
                "max_words": max_words
            }
        )
        
        logger.info(f"Summary generated successfully for {table_name}")
        
        return response
    
    def _get_dataset_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get basic information about the dataset.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with dataset information
        """
        schema = self.extraction_agent.data_store.get_table_schema(table_name)
        df = self.extraction_agent.data_store.tables[table_name]
        
        info = {
            "table_name": table_name,
            "row_count": schema["row_count"],
            "column_count": len(schema["columns"]),
            "columns": [col["name"] if isinstance(col, dict) else col 
                       for col in schema["columns"]],
            "date_range": self._get_date_range(df),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        return info
    
    def _get_date_range(self, df: pd.DataFrame) -> Optional[Dict[str, str]]:
        """
        Get date range from dataset if date columns exist.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with start_date and end_date, or None
        """
        # Look for date columns
        date_columns = df.select_dtypes(include=['datetime64']).columns
        
        if len(date_columns) == 0:
            # Try to find columns with 'date' in name
            date_like_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_like_cols:
                try:
                    df[date_like_cols[0]] = pd.to_datetime(df[date_like_cols[0]])
                    date_columns = [date_like_cols[0]]
                except:
                    return None
            else:
                return None
        
        if len(date_columns) > 0:
            date_col = date_columns[0]
            return {
                "start_date": str(df[date_col].min()),
                "end_date": str(df[date_col].max()),
                "column": date_col
            }
        
        return None
    
    def _calculate_key_metrics(self, table_name: str) -> Dict[str, Any]:
        """
        Calculate key metrics from the dataset.
        
        Includes:
        - Total sales/revenue
        - Average values
        - Year-over-year growth rates
        - Count metrics
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with calculated metrics
            
        **Validates: Requirements 6.1, 6.2**
        """
        df = self.extraction_agent.data_store.tables[table_name]
        metrics = {}
        
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Calculate basic statistics for numeric columns
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            metrics[f"{col}_total"] = float(df[col].sum())
            metrics[f"{col}_average"] = float(df[col].mean())
            metrics[f"{col}_min"] = float(df[col].min())
            metrics[f"{col}_max"] = float(df[col].max())
        
        # Calculate year-over-year growth if date column exists
        yoy_growth = self._calculate_yoy_growth(df)
        if yoy_growth:
            metrics["yoy_growth"] = yoy_growth
        
        # Count unique values in categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
            metrics[f"{col}_unique_count"] = int(df[col].nunique())
        
        return metrics
    
    def _calculate_yoy_growth(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        Calculate year-over-year growth rates.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with YoY growth rates, or None
            
        **Validates: Requirement 6.2**
        """
        # Find date column
        date_columns = df.select_dtypes(include=['datetime64']).columns
        
        if len(date_columns) == 0:
            # Try to find columns with 'date' in name
            date_like_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_like_cols:
                try:
                    df = df.copy()
                    df[date_like_cols[0]] = pd.to_datetime(df[date_like_cols[0]])
                    date_columns = [date_like_cols[0]]
                except:
                    return None
            else:
                return None
        
        if len(date_columns) == 0:
            return None
        
        date_col = date_columns[0]
        
        # Find numeric columns (likely sales/revenue)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return None
        
        # Use first numeric column as the metric
        metric_col = numeric_cols[0]
        
        try:
            # Extract year from date
            df_copy = df.copy()
            df_copy['year'] = df_copy[date_col].dt.year
            
            # Group by year and sum
            yearly_totals = df_copy.groupby('year')[metric_col].sum()
            
            if len(yearly_totals) < 2:
                return None
            
            # Calculate YoY growth for each year
            yoy_growth = {}
            years = sorted(yearly_totals.index)
            
            for i in range(1, len(years)):
                prev_year = years[i-1]
                curr_year = years[i]
                
                prev_value = yearly_totals[prev_year]
                curr_value = yearly_totals[curr_year]
                
                if prev_value != 0:
                    growth_rate = ((curr_value - prev_value) / prev_value) * 100
                    yoy_growth[f"{prev_year}_to_{curr_year}"] = round(growth_rate, 2)
            
            return yoy_growth if yoy_growth else None
            
        except Exception as e:
            logger.warning(f"Failed to calculate YoY growth: {str(e)}")
            return None
    
    def _identify_performers(self, table_name: str) -> Dict[str, Any]:
        """
        Identify top-performing and underperforming categories.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with top and bottom performers
            
        **Validates: Requirement 6.3**
        """
        df = self.extraction_agent.data_store.tables[table_name]
        performers = {}
        
        # Find categorical columns (potential categories)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Find numeric columns (potential metrics)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not categorical_cols or not numeric_cols:
            return performers
        
        # Use first categorical column as category
        category_col = categorical_cols[0]
        
        # Use first numeric column as metric
        metric_col = numeric_cols[0]
        
        try:
            # Group by category and sum metric
            category_totals = df.groupby(category_col)[metric_col].sum().sort_values(ascending=False)
            
            # Top 5 performers
            top_performers = category_totals.head(5)
            performers["top_performers"] = {
                "category_column": category_col,
                "metric_column": metric_col,
                "categories": [
                    {"name": str(cat), "value": float(val)}
                    for cat, val in top_performers.items()
                ]
            }
            
            # Bottom 5 performers
            bottom_performers = category_totals.tail(5)
            performers["bottom_performers"] = {
                "category_column": category_col,
                "metric_column": metric_col,
                "categories": [
                    {"name": str(cat), "value": float(val)}
                    for cat, val in bottom_performers.items()
                ]
            }
            
        except Exception as e:
            logger.warning(f"Failed to identify performers: {str(e)}")
        
        return performers
    
    def _detect_trends(self, table_name: str) -> Dict[str, Any]:
        """
        Detect significant trends and anomalies in the data.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with detected trends and anomalies
            
        **Validates: Requirement 6.4**
        """
        df = self.extraction_agent.data_store.tables[table_name]
        trends = {}
        
        # Detect anomalies in numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        anomalies = []
        for col in numeric_cols[:3]:  # Check first 3 numeric columns
            # Check for negative values where they shouldn't be
            if 'sales' in col.lower() or 'revenue' in col.lower() or 'amount' in col.lower():
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    anomalies.append({
                        "type": "negative_values",
                        "column": col,
                        "count": int(negative_count),
                        "description": f"Found {negative_count} negative values in {col}"
                    })
            
            # Check for outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                anomalies.append({
                    "type": "outliers",
                    "column": col,
                    "count": len(outliers),
                    "description": f"Found {len(outliers)} outliers in {col}"
                })
        
        trends["anomalies"] = anomalies
        
        # Detect time-based trends if date column exists
        date_columns = df.select_dtypes(include=['datetime64']).columns
        
        if len(date_columns) > 0 and len(numeric_cols) > 0:
            date_col = date_columns[0]
            metric_col = numeric_cols[0]
            
            try:
                # Sort by date
                df_sorted = df.sort_values(date_col)
                
                # Calculate rolling average
                df_sorted['rolling_avg'] = df_sorted[metric_col].rolling(window=7, min_periods=1).mean()
                
                # Detect trend direction
                first_half_avg = df_sorted[metric_col].iloc[:len(df_sorted)//2].mean()
                second_half_avg = df_sorted[metric_col].iloc[len(df_sorted)//2:].mean()
                
                if second_half_avg > first_half_avg * 1.1:
                    trend_direction = "increasing"
                elif second_half_avg < first_half_avg * 0.9:
                    trend_direction = "decreasing"
                else:
                    trend_direction = "stable"
                
                trends["time_trend"] = {
                    "direction": trend_direction,
                    "first_half_avg": float(first_half_avg),
                    "second_half_avg": float(second_half_avg),
                    "change_percent": float(((second_half_avg - first_half_avg) / first_half_avg) * 100)
                }
                
            except Exception as e:
                logger.warning(f"Failed to detect time trends: {str(e)}")
        
        return trends
    
    def _format_summary(
        self,
        dataset_info: Dict[str, Any],
        metrics: Dict[str, Any],
        performers: Dict[str, Any],
        trends: Dict[str, Any],
        max_words: int
    ) -> str:
        """
        Format summary using LLM in natural language.
        
        Args:
            dataset_info: Dataset information
            metrics: Calculated metrics
            performers: Top and bottom performers
            trends: Detected trends and anomalies
            max_words: Maximum words for summary
            
        Returns:
            Formatted summary string
            
        **Validates: Requirements 6.5, 6.6**
        """
        logger.info("Formatting summary using LLM")
        
        try:
            # Prepare prompt
            prompt = PromptTemplates.format_summarization_prompt(
                dataset_info=dataset_info,
                metrics={**metrics, **performers, **trends}
            )
            
            # Add word limit instruction
            prompt += f"\n\nIMPORTANT: Keep the summary under {max_words} words."
            
            # Generate summary
            response = self.llm_provider.generate_with_cache(
                prompt=prompt,
                temperature=0.7
            )
            
            summary = response.content.strip()
            
            # Check word count
            word_count = len(summary.split())
            logger.info(f"Generated summary with {word_count} words (limit: {max_words})")
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to format summary with LLM: {str(e)}")
            # Fallback to basic summary
            return self._generate_fallback_summary(
                dataset_info, metrics, performers, trends
            )
    
    def _generate_fallback_summary(
        self,
        dataset_info: Dict[str, Any],
        metrics: Dict[str, Any],
        performers: Dict[str, Any],
        trends: Dict[str, Any]
    ) -> str:
        """Generate basic fallback summary without LLM."""
        summary_parts = []
        
        # Overview
        summary_parts.append(f"Dataset Summary: {dataset_info['table_name']}")
        summary_parts.append(f"Total Records: {dataset_info['row_count']:,}")
        summary_parts.append(f"Columns: {dataset_info['column_count']}")
        
        # Key metrics
        if metrics:
            summary_parts.append("\nKey Metrics:")
            for key, value in list(metrics.items())[:5]:
                if isinstance(value, (int, float)):
                    summary_parts.append(f"  - {key}: {value:,.2f}")
        
        # Top performers
        if "top_performers" in performers:
            summary_parts.append("\nTop Performers:")
            for cat in performers["top_performers"]["categories"][:3]:
                summary_parts.append(f"  - {cat['name']}: {cat['value']:,.2f}")
        
        # Trends
        if "anomalies" in trends and trends["anomalies"]:
            summary_parts.append("\nAnomalies Detected:")
            for anomaly in trends["anomalies"][:3]:
                summary_parts.append(f"  - {anomaly['description']}")
        
        return "\n".join(summary_parts)
    
    def generate_comparative_summary(
        self,
        table_names: List[str],
        max_words: int = 500
    ) -> Response:
        """
        Generate comparative summary across multiple datasets.
        
        Args:
            table_names: List of table names to compare
            max_words: Maximum words for summary
            
        Returns:
            Response object with comparative summary
            
        Raises:
            ValueError: If fewer than 2 tables provided or tables not found
            
        **Validates: Requirement 6.7**
        """
        logger.info(f"Generating comparative summary for {len(table_names)} tables")
        
        if len(table_names) < 2:
            raise ValueError("Comparative summary requires at least 2 tables")
        
        # Verify all tables exist
        available_tables = self.extraction_agent.data_store.list_tables()
        for table_name in table_names:
            if table_name not in available_tables:
                raise ValueError(
                    f"Table '{table_name}' not found. "
                    f"Available tables: {', '.join(available_tables)}"
                )
        
        # Generate summaries for each table
        table_summaries = {}
        for table_name in table_names:
            dataset_info = self._get_dataset_info(table_name)
            metrics = self._calculate_key_metrics(table_name)
            performers = self._identify_performers(table_name)
            trends = self._detect_trends(table_name)
            
            table_summaries[table_name] = {
                "dataset_info": dataset_info,
                "metrics": metrics,
                "performers": performers,
                "trends": trends
            }
        
        # Format comparative summary
        comparative_summary = self._format_comparative_summary(
            table_summaries, max_words
        )
        
        # Create response
        response = Response(
            answer=comparative_summary,
            metadata={
                "mode": "summarization",
                "comparison": True,
                "table_names": table_names,
                "table_summaries": table_summaries,
                "max_words": max_words
            }
        )
        
        logger.info(f"Comparative summary generated for {len(table_names)} tables")
        
        return response
    
    def _format_comparative_summary(
        self,
        table_summaries: Dict[str, Dict[str, Any]],
        max_words: int
    ) -> str:
        """
        Format comparative summary using LLM.
        
        Args:
            table_summaries: Dictionary of table summaries
            max_words: Maximum words for summary
            
        Returns:
            Formatted comparative summary string
        """
        logger.info("Formatting comparative summary using LLM")
        
        try:
            # Prepare comparison data
            comparison_data = []
            
            for table_name, summary_data in table_summaries.items():
                comparison_data.append(f"\n=== {table_name} ===")
                comparison_data.append(f"Records: {summary_data['dataset_info']['row_count']:,}")
                
                # Add key metrics
                if summary_data['metrics']:
                    comparison_data.append("Key Metrics:")
                    for key, value in list(summary_data['metrics'].items())[:3]:
                        if isinstance(value, (int, float)):
                            comparison_data.append(f"  {key}: {value:,.2f}")
            
            comparison_text = "\n".join(comparison_data)
            
            # Create prompt
            prompt = f"""You are a business intelligence analyst. Generate a comparative summary of multiple datasets.

Datasets to Compare:
{comparison_text}

Instructions:
1. Compare key metrics across datasets
2. Highlight similarities and differences
3. Identify which dataset performs better in which areas
4. Provide specific numerical evidence
5. Keep the summary concise (maximum {max_words} words)
6. Use clear, business-friendly language

Format the summary with:
- Overview: High-level comparison
- Key Differences: Major variations between datasets
- Insights: What the comparison reveals
- Recommendations: Actionable insights

Comparative Summary:"""
            
            # Generate summary
            response = self.llm_provider.generate_with_cache(
                prompt=prompt,
                temperature=0.7
            )
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to format comparative summary: {str(e)}")
            # Fallback to basic comparison
            return self._generate_fallback_comparative_summary(table_summaries)
    
    def _generate_fallback_comparative_summary(
        self,
        table_summaries: Dict[str, Dict[str, Any]]
    ) -> str:
        """Generate basic fallback comparative summary."""
        summary_parts = ["Comparative Dataset Summary\n"]
        
        for table_name, summary_data in table_summaries.items():
            summary_parts.append(f"\n{table_name}:")
            summary_parts.append(f"  Records: {summary_data['dataset_info']['row_count']:,}")
            summary_parts.append(f"  Columns: {summary_data['dataset_info']['column_count']}")
            
            if summary_data['metrics']:
                first_metric = list(summary_data['metrics'].items())[0]
                summary_parts.append(f"  {first_metric[0]}: {first_metric[1]:,.2f}")
        
        return "\n".join(summary_parts)
