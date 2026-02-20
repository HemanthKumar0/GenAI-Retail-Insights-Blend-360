"""
Structured prompt templates for LLM interactions.

This module provides prompt templates for query parsing, response formatting,
and summarization with few-shot examples for improved accuracy.
"""

from typing import Dict, Any, List


class PromptTemplates:
    """Collection of structured prompt templates."""
    
    # Query parsing prompt with few-shot examples
    QUERY_PARSING_PROMPT = """You are a data query expert. Convert the user's natural language query into a structured data operation.

Available Schema:
{schema}

Conversation Context:
{context}

User Query: {query}

Generate one of:
1. SQL query (for aggregations, filters, joins)
2. Pandas operation (for transformations)
3. Semantic search (for finding similar records)

IMPORTANT SQL SYNTAX RULES:
- Use DOUBLE QUOTES for column names with spaces (e.g., "Courier Status", not `Courier Status`)
- DuckDB does not support backticks for identifiers
- Always quote column names that contain spaces or special characters
- When aggregating columns that might be stored as text, use CAST or TRY_CAST to convert to numeric:
  * CAST("column_name" AS DOUBLE) for guaranteed numeric columns
  * TRY_CAST("column_name" AS DOUBLE) for columns that might have non-numeric values
  * Example: SELECT SUM(TRY_CAST("GROSS AMT" AS DOUBLE)) FROM table
- When performing arithmetic operations (-, +, *, /) on columns, always cast text columns to numeric first:
  * Example: TRY_CAST("Final MRP Old" AS DOUBLE) - TRY_CAST("Ajio MRP" AS DOUBLE)
  * This applies to all mathematical operations, not just aggregations
- When performing arithmetic operations (-, +, *, /) on columns, always cast text columns to numeric first:
  * Example: TRY_CAST("Final MRP Old" AS DOUBLE) - TRY_CAST("Ajio MRP" AS DOUBLE)
  * This applies to all mathematical operations on potentially text-stored columns

Few-shot Examples:

Example 1:
User Query: "What were the total sales in Q4 2022?"
Output:
{{
  "operation_type": "sql",
  "operation": "SELECT SUM(sales) as total_sales FROM sales WHERE date >= '2022-10-01' AND date <= '2022-12-31'",
  "explanation": "Aggregates sales for Q4 2022 (October-December)"
}}

Example 2:
User Query: "Show me the top 5 products by revenue"
Output:
{{
  "operation_type": "sql",
  "operation": "SELECT product_name, SUM(sales) as revenue FROM sales GROUP BY product_name ORDER BY revenue DESC LIMIT 5",
  "explanation": "Groups by product and returns top 5 by total revenue"
}}

Example 3:
User Query: "Search based on the courier status"
Output:
{{
  "operation_type": "sql",
  "operation": "SELECT * FROM Amazon_Sale_Report WHERE \"Courier Status\" IS NOT NULL",
  "explanation": "Retrieves all records where there is a specified courier status. Note: Column name with space uses double quotes."
}}

Example 4:
User Query: "What are the total sales by category?"
Output:
{{
  "operation_type": "sql",
  "operation": "SELECT Category, SUM(TRY_CAST(\"GROSS AMT\" AS DOUBLE)) as total_sales FROM International_sale_Report GROUP BY Category ORDER BY total_sales DESC",
  "explanation": "Groups by category and sums the gross amount. Uses TRY_CAST to handle text-stored numeric values."
}}

Example 5:
User Query: "What is the total profit and loss?"
Output:
{{
  "operation_type": "sql",
  "operation": "SELECT SUM(TRY_CAST(\"Final MRP Old\" AS DOUBLE) - TRY_CAST(\"Ajio MRP\" AS DOUBLE)) as total_profit_loss FROM P__L_March_2021",
  "explanation": "Calculates profit/loss by subtracting two columns. Uses TRY_CAST on both columns before arithmetic operation."
}}

Example 6:
User Query: "Calculate year-over-year growth for each category"
Output:
{{
  "operation_type": "pandas",
  "operation": "df.groupby(['category', df['date'].dt.year])['sales'].sum().unstack().pct_change(axis=1)",
  "explanation": "Groups by category and year, calculates percentage change between years"
}}

Example 7:
User Query: "Find products similar to 'laptop'"
Output:
{{
  "operation_type": "semantic",
  "operation": "laptop",
  "explanation": "Performs semantic search to find products similar to 'laptop'"
}}

Now generate the structured operation for the user's query. Return ONLY valid JSON in the format shown above.

Output:"""

    # Summarization prompt
    SUMMARIZATION_PROMPT = """You are a business intelligence analyst. Generate a comprehensive summary of the sales dataset.

Dataset Information:
{dataset_info}

Key Metrics:
{metrics}

Instructions:
1. Analyze the dataset and identify key performance indicators
2. Calculate year-over-year growth rates where applicable
3. Identify top-performing and underperforming product categories
4. Highlight significant trends or anomalies
5. Provide specific numerical evidence for each insight
6. Keep the summary concise (maximum 500 words)
7. Use clear, business-friendly language

Format the summary with the following sections:
- Overview: High-level summary of the dataset
- Key Metrics: Important numbers and trends
- Top Performers: Best-performing categories/products
- Areas of Concern: Underperforming areas or anomalies
- Recommendations: Actionable insights based on the data

Summary:"""

    # Context summarization prompt (for managing conversation history)
    CONTEXT_SUMMARIZATION_PROMPT = """Summarize the following conversation history into a concise context summary that preserves key information.

Conversation History:
{history}

Create a brief summary (max 200 words) that captures:
1. The main topics discussed
2. Key data points or findings mentioned
3. Any ongoing analysis or questions
4. Current dataset context

Summary:"""

    @staticmethod
    def format_query_parsing_prompt(query: str, schema: Dict[str, Any], 
                                    context: str = "") -> str:
        """
        Format query parsing prompt with parameters.
        
        Args:
            query: User's natural language query
            schema: Database schema information
            context: Conversation context
            
        Returns:
            Formatted prompt string
        """
        schema_str = PromptTemplates._format_schema(schema)
        context_str = context if context else "No previous context"
        
        return PromptTemplates.QUERY_PARSING_PROMPT.format(
            schema=schema_str,
            context=context_str,
            query=query
        )
    
    @staticmethod
    def format_response_formatting_prompt(user_query: str, data_summary: str, 
                                         warnings: List[str] = None) -> str:
        """
        Format response formatting prompt with data summary and warnings.
        
        Args:
            user_query: User's original query
            data_summary: Summary of query results
            warnings: List of validation warnings
            
        Returns:
            Formatted prompt string
        """
        warnings_str = ""
        if warnings:
            warnings_str = "\n\nValidation Warnings:\n" + "\n".join(f"- {w}" for w in warnings)
        
        prompt = f"""You are a helpful data analyst assistant. Format the query results into a clear, natural language response.

User Query: {user_query}

Query Results Summary:
{data_summary}{warnings_str}

Instructions:
1. Provide a clear, concise answer to the user's question
2. Reference specific data points from the results
3. Include relevant numbers and percentages
4. Highlight key insights or trends
5. Keep the response conversational and easy to understand
6. If there are warnings, mention them appropriately
7. If the results are empty, explain why and suggest alternatives

Response:"""
        
        return prompt
    
    @staticmethod
    def format_summarization_prompt(dataset_info: Dict[str, Any], 
                                    metrics: Dict[str, Any]) -> str:
        """
        Format summarization prompt.
        
        Args:
            dataset_info: Information about the dataset
            metrics: Calculated metrics
            
        Returns:
            Formatted prompt string
        """
        dataset_str = PromptTemplates._format_dataset_info(dataset_info)
        metrics_str = PromptTemplates._format_metrics(metrics)
        
        return PromptTemplates.SUMMARIZATION_PROMPT.format(
            dataset_info=dataset_str,
            metrics=metrics_str
        )
    
    @staticmethod
    def format_context_summary_prompt(history: List[Dict[str, str]]) -> str:
        """
        Format context summarization prompt.
        
        Args:
            history: List of conversation messages
            
        Returns:
            Formatted prompt string
        """
        history_str = PromptTemplates._format_conversation_history(history)
        
        return PromptTemplates.CONTEXT_SUMMARIZATION_PROMPT.format(
            history=history_str
        )
    
    
    # Helper methods for formatting
    
    @staticmethod
    def _format_schema(schema: Dict[str, Any]) -> str:
        """Format schema dictionary into readable string."""
        if not schema:
            return "No schema available"
        
        lines = []
        for table_name, table_info in schema.items():
            lines.append(f"Table: {table_name}")
            if isinstance(table_info, dict) and 'columns' in table_info:
                for col in table_info['columns']:
                    if isinstance(col, dict):
                        col_name = col.get('name', 'unknown')
                        col_type = col.get('dtype', 'unknown')
                        lines.append(f"  - {col_name} ({col_type})")
                    else:
                        lines.append(f"  - {col}")
            lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_dataset_info(dataset_info: Dict[str, Any]) -> str:
        """Format dataset information into readable string."""
        lines = []
        for key, value in dataset_info.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)
    
    @staticmethod
    def _format_metrics(metrics: Dict[str, Any]) -> str:
        """Format metrics into readable string."""
        lines = []
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.2f}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)
    
    @staticmethod
    def _format_conversation_history(history: List[Dict[str, str]]) -> str:
        """Format conversation history into readable string."""
        lines = []
        for msg in history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            lines.append(f"{role.capitalize()}: {content}")
        return "\n\n".join(lines)
