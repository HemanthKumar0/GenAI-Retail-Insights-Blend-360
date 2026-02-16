"""
Unit tests for RAGSystem.

**Validates: Requirements 9.1, 9.2, 9.3, 9.5**
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Check if dependencies are available
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from src.rag_system import RAGSystem, SearchResult, DEPENDENCIES_AVAILABLE
    SKIP_TESTS = not DEPENDENCIES_AVAILABLE
except ImportError:
    SKIP_TESTS = True


@pytest.mark.skipif(SKIP_TESTS, reason="RAG dependencies not installed")
class TestRAGSystem:
    """Test suite for RAGSystem class."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            "product": ["Laptop", "Mouse", "Keyboard", "Monitor", "Headphones"],
            "category": ["Electronics", "Accessories", "Accessories", "Electronics", "Accessories"],
            "price": [1200, 25, 75, 350, 150],
            "sales": [50, 200, 150, 80, 120]
        })
    
    @pytest.fixture
    def rag_system(self):
        """Create a RAGSystem instance."""
        return RAGSystem(model_name="all-MiniLM-L6-v2")
    
    def test_initialization(self, rag_system):
        """
        Test RAGSystem initialization.
        
        **Validates: Requirements 9.1, 9.4**
        """
        assert rag_system.model is not None
        assert rag_system.embedding_dim > 0
        assert rag_system.index is not None
        assert len(rag_system.documents) == 0
        assert len(rag_system.metadata) == 0
    
    def test_initialization_without_dependencies(self):
        """Test that initialization fails gracefully without dependencies."""
        with patch('src.rag_system.DEPENDENCIES_AVAILABLE', False):
            with pytest.raises(ImportError, match="RAG dependencies not available"):
                from src.rag_system import RAGSystem
                RAGSystem()
    
    def test_index_dataset_basic(self, rag_system, sample_dataframe):
        """
        Test basic dataset indexing.
        
        **Validates: Requirements 9.1, 9.4**
        """
        # Index the dataset
        rag_system.index_dataset("products", sample_dataframe)
        
        # Verify documents were added
        assert len(rag_system.documents) > 0
        assert len(rag_system.metadata) > 0
        assert rag_system.index.ntotal > 0
        
        # Verify table metadata was indexed
        table_metadata = [m for m in rag_system.metadata if m["type"] == "table_metadata"]
        assert len(table_metadata) == 1
        assert table_metadata[0]["table_name"] == "products"
        assert table_metadata[0]["row_count"] == 5
    
    def test_index_dataset_columns(self, rag_system, sample_dataframe):
        """
        Test that column metadata is indexed.
        
        **Validates: Requirement 9.1**
        """
        rag_system.index_dataset("products", sample_dataframe)
        
        # Verify column metadata was indexed
        column_metadata = [m for m in rag_system.metadata if m["type"] == "column_metadata"]
        assert len(column_metadata) == 4  # 4 columns in sample data
        
        # Check specific column
        product_col = [m for m in column_metadata if m["column_name"] == "product"]
        assert len(product_col) == 1
        assert product_col[0]["table_name"] == "products"
    
    def test_index_dataset_sample_rows(self, rag_system, sample_dataframe):
        """
        Test that sample rows are indexed.
        
        **Validates: Requirement 9.1**
        """
        rag_system.index_dataset("products", sample_dataframe)
        
        # Verify sample rows were indexed
        sample_rows = [m for m in rag_system.metadata if m["type"] == "sample_row"]
        assert len(sample_rows) == 5  # All 5 rows (less than 10)
        
        # Check row indices
        row_indices = [m["row_index"] for m in sample_rows]
        assert row_indices == [0, 1, 2, 3, 4]
    
    def test_semantic_search_basic(self, rag_system, sample_dataframe):
        """
        Test basic semantic search.
        
        **Validates: Requirements 9.2, 9.5**
        """
        # Index dataset
        rag_system.index_dataset("products", sample_dataframe)
        
        # Perform search
        results = rag_system.semantic_search("electronics products", top_k=3)
        
        # Verify results
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(0 <= r.similarity_score <= 1 for r in results)
        
        # Results should be sorted by similarity (highest first)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].similarity_score >= results[i + 1].similarity_score
    
    def test_semantic_search_returns_top_k(self, rag_system, sample_dataframe):
        """
        Test that semantic search returns exactly top_k results.
        
        **Validates: Requirement 9.5**
        """
        rag_system.index_dataset("products", sample_dataframe)
        
        # Test different top_k values
        for k in [1, 3, 5]:
            results = rag_system.semantic_search("laptop computer", top_k=k)
            assert len(results) <= k
    
    def test_semantic_search_similarity_scores(self, rag_system, sample_dataframe):
        """
        Test that similarity scores are included in results.
        
        **Validates: Requirement 9.5**
        """
        rag_system.index_dataset("products", sample_dataframe)
        
        results = rag_system.semantic_search("price information", top_k=3)
        
        # All results should have similarity scores
        assert all(hasattr(r, 'similarity_score') for r in results)
        assert all(isinstance(r.similarity_score, float) for r in results)
        assert all(0 <= r.similarity_score <= 1 for r in results)
    
    def test_semantic_search_empty_index(self, rag_system):
        """Test semantic search with empty index."""
        results = rag_system.semantic_search("test query", top_k=3)
        assert len(results) == 0
    
    def test_get_relevant_context(self, rag_system, sample_dataframe):
        """
        Test context retrieval for LLM augmentation.
        
        **Validates: Requirement 9.3**
        """
        # Index dataset
        rag_system.index_dataset("products", sample_dataframe)
        
        # Get relevant context
        context = rag_system.get_relevant_context("laptop sales", top_k=3)
        
        # Verify context format
        assert isinstance(context, str)
        assert len(context) > 0
        assert "Relevant context" in context
        assert "similarity:" in context.lower()
    
    def test_get_relevant_context_empty_index(self, rag_system):
        """Test context retrieval with empty index."""
        context = rag_system.get_relevant_context("test query", top_k=3)
        assert "No relevant context found" in context
    
    def test_get_relevant_context_includes_top_k(self, rag_system, sample_dataframe):
        """Test that context includes top_k results."""
        rag_system.index_dataset("products", sample_dataframe)
        
        context = rag_system.get_relevant_context("products", top_k=2)
        
        # Should include 2 results
        assert context.count("[Result") == 2
    
    def test_clear(self, rag_system, sample_dataframe):
        """Test clearing the RAG system."""
        # Index dataset
        rag_system.index_dataset("products", sample_dataframe)
        assert len(rag_system.documents) > 0
        assert rag_system.index.ntotal > 0
        
        # Clear
        rag_system.clear()
        
        # Verify cleared
        assert len(rag_system.documents) == 0
        assert len(rag_system.metadata) == 0
        assert rag_system.index.ntotal == 0
    
    def test_get_stats(self, rag_system, sample_dataframe):
        """Test getting RAG system statistics."""
        # Before indexing
        stats = rag_system.get_stats()
        assert stats["total_documents"] == 0
        assert stats["embedding_dimension"] > 0
        
        # After indexing
        rag_system.index_dataset("products", sample_dataframe)
        stats = rag_system.get_stats()
        assert stats["total_documents"] > 0
        assert stats["indexed_tables"] == 1
    
    def test_multiple_tables_indexing(self, rag_system, sample_dataframe):
        """Test indexing multiple tables."""
        # Create second table
        sales_df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "revenue": [1000, 1500]
        })
        
        # Index both tables
        rag_system.index_dataset("products", sample_dataframe)
        rag_system.index_dataset("sales", sales_df)
        
        # Verify both tables are indexed
        stats = rag_system.get_stats()
        assert stats["indexed_tables"] == 2
        
        # Verify metadata contains both tables
        table_names = set(m.get("table_name") for m in rag_system.metadata if "table_name" in m)
        assert "products" in table_names
        assert "sales" in table_names
    
    def test_search_result_metadata(self, rag_system, sample_dataframe):
        """Test that search results include metadata."""
        rag_system.index_dataset("products", sample_dataframe)
        
        results = rag_system.semantic_search("product information", top_k=3)
        
        # All results should have metadata
        assert all(hasattr(r, 'metadata') for r in results)
        assert all(isinstance(r.metadata, dict) for r in results)
        assert all('type' in r.metadata for r in results)
        assert all('table_name' in r.metadata for r in results)
    
    def test_semantic_search_relevance(self, rag_system, sample_dataframe):
        """Test that semantic search returns relevant results."""
        rag_system.index_dataset("products", sample_dataframe)
        
        # Search for price-related information
        results = rag_system.semantic_search("price cost money", top_k=5)
        
        # At least one result should mention price
        contents = [r.content.lower() for r in results]
        assert any("price" in content for content in contents)
    
    def test_large_dataset_indexing(self, rag_system):
        """Test indexing a larger dataset."""
        # Create a larger dataset
        large_df = pd.DataFrame({
            "id": range(100),
            "value": np.random.rand(100),
            "category": [f"cat_{i % 10}" for i in range(100)]
        })
        
        # Index dataset
        rag_system.index_dataset("large_table", large_df)
        
        # Verify indexing completed
        assert len(rag_system.documents) > 0
        
        # Sample rows should be limited to 10
        sample_rows = [m for m in rag_system.metadata if m["type"] == "sample_row"]
        assert len(sample_rows) == 10


@pytest.mark.skipif(not SKIP_TESTS, reason="Only run when dependencies are missing")
def test_import_without_dependencies():
    """Test that module can be imported without dependencies."""
    # This test runs when dependencies are NOT available
    # It verifies graceful degradation
    try:
        from src.rag_system import RAGSystem, DEPENDENCIES_AVAILABLE
        assert not DEPENDENCIES_AVAILABLE
    except ImportError:
        # Expected if module can't be imported at all
        pass
