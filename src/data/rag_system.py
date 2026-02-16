"""
RAG (Retrieval-Augmented Generation) System for semantic search.

This module implements a RAG system using FAISS for vector storage and
sentence-transformers for embeddings to enable semantic search capabilities.

**Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    logging.warning(
        "RAG dependencies not available. Install with: "
        "pip install faiss-cpu sentence-transformers"
    )

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Represents a single search result from semantic search.
    
    Attributes:
        content: The text content of the result
        similarity_score: Similarity score (0.0 to 1.0)
        metadata: Additional metadata about the result
    """
    content: str
    similarity_score: float
    metadata: Dict[str, Any]


class RAGSystem:
    """
    RAG System for semantic search and context retrieval.
    
    This class provides:
    - Vector embeddings generation using sentence-transformers
    - FAISS vector store for efficient similarity search
    - Semantic search returning top-k most relevant results
    - Context retrieval for LLM prompt augmentation
    - Dataset metadata and sample indexing
    
    **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG system.
        
        Args:
            model_name: Name of the sentence-transformers model to use
                       Default: "all-MiniLM-L6-v2" (fast and efficient)
        
        Raises:
            ImportError: If required dependencies are not installed
            
        **Validates: Requirements 9.1, 9.4**
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "RAG dependencies not available. Install with: "
                "pip install faiss-cpu sentence-transformers"
            )
        
        logger.info(f"Initializing RAG system with model: {model_name}")
        
        # Load sentence transformer model
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index (using L2 distance)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Store document contents and metadata
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        
        logger.info(
            f"RAG system initialized with embedding dimension: {self.embedding_dim}"
        )
    
    def index_dataset(self, table_name: str, df: pd.DataFrame) -> None:
        """
        Generate embeddings for dataset metadata and samples.
        
        This method indexes:
        - Table name and description
        - Column names and descriptions
        - Sample rows from the dataset
        
        Args:
            table_name: Name of the table to index
            df: DataFrame containing the dataset
            
        **Validates: Requirements 9.1, 9.4**
        """
        logger.info(f"Indexing dataset: {table_name} ({len(df)} rows)")
        
        documents_to_add = []
        metadata_to_add = []
        
        # 1. Index table metadata
        table_doc = f"Table: {table_name}\nColumns: {', '.join(df.columns.tolist())}\nRow count: {len(df)}"
        documents_to_add.append(table_doc)
        metadata_to_add.append({
            "type": "table_metadata",
            "table_name": table_name,
            "row_count": len(df)
        })
        
        # 2. Index column descriptions
        for col in df.columns:
            col_dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            
            # Get sample values
            sample_values = df[col].dropna().head(5).tolist()
            sample_str = ", ".join([str(v) for v in sample_values])
            
            col_doc = (
                f"Column: {col} in table {table_name}\n"
                f"Type: {col_dtype}\n"
                f"Unique values: {unique_count}\n"
                f"Sample values: {sample_str}"
            )
            
            documents_to_add.append(col_doc)
            metadata_to_add.append({
                "type": "column_metadata",
                "table_name": table_name,
                "column_name": col,
                "dtype": col_dtype,
                "unique_count": unique_count
            })
        
        # 3. Index sample rows (first 10 rows)
        sample_size = min(10, len(df))
        for idx in range(sample_size):
            row = df.iloc[idx]
            row_doc = f"Sample row from {table_name}:\n"
            for col in df.columns:
                row_doc += f"  {col}: {row[col]}\n"
            
            documents_to_add.append(row_doc)
            metadata_to_add.append({
                "type": "sample_row",
                "table_name": table_name,
                "row_index": idx
            })
        
        # Generate embeddings and add to index
        if documents_to_add:
            embeddings = self.model.encode(documents_to_add, show_progress_bar=False)
            self.index.add(embeddings.astype('float32'))
            self.documents.extend(documents_to_add)
            self.metadata.extend(metadata_to_add)
            
            logger.info(
                f"Indexed {len(documents_to_add)} documents for table {table_name}"
            )
    
    def semantic_search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        Perform semantic search and return top-k most relevant results.
        
        Args:
            query: Search query text
            top_k: Number of top results to return (default: 3)
            
        Returns:
            List of SearchResult objects with similarity scores
            
        **Validates: Requirements 9.2, 9.5**
        """
        if self.index.ntotal == 0:
            logger.warning("No documents indexed, returning empty results")
            return []
        
        logger.info(f"Performing semantic search: '{query}' (top_k={top_k})")
        
        # Generate query embedding
        query_embedding = self.model.encode([query], show_progress_bar=False)
        
        # Search in FAISS index
        # Note: FAISS returns L2 distances, we convert to similarity scores
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            min(top_k, self.index.ntotal)
        )
        
        # Convert distances to similarity scores (0 to 1)
        # Lower L2 distance = higher similarity
        # We use: similarity = 1 / (1 + distance)
        similarities = 1.0 / (1.0 + distances[0])
        
        # Create SearchResult objects
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if idx < len(self.documents):  # Safety check
                results.append(SearchResult(
                    content=self.documents[idx],
                    similarity_score=float(similarity),
                    metadata=self.metadata[idx]
                ))
        
        logger.info(f"Found {len(results)} results")
        
        return results
    
    def get_relevant_context(self, query: str, top_k: int = 3) -> str:
        """
        Get relevant context string to augment LLM prompts.
        
        This method performs semantic search and formats the results
        into a context string suitable for including in LLM prompts.
        
        Args:
            query: Query text to find relevant context for
            top_k: Number of top results to include (default: 3)
            
        Returns:
            Formatted context string
            
        **Validates: Requirement 9.3**
        """
        logger.info(f"Retrieving relevant context for: '{query}'")
        
        results = self.semantic_search(query, top_k=top_k)
        
        if not results:
            return "No relevant context found."
        
        # Format results into context string
        context_parts = ["Relevant context from dataset:"]
        
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"\n[Result {i}] (similarity: {result.similarity_score:.3f})\n"
                f"{result.content}"
            )
        
        context = "\n".join(context_parts)
        
        logger.info(f"Retrieved context with {len(results)} results")
        
        return context
    
    def clear(self) -> None:
        """Clear all indexed documents and reset the index."""
        logger.info("Clearing RAG system")
        
        # Reset FAISS index
        self.index.reset()
        
        # Clear documents and metadata
        self.documents.clear()
        self.metadata.clear()
        
        logger.info("RAG system cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system.
        
        Returns:
            Dictionary with RAG system statistics
        """
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embedding_dim,
            "model_name": self.model._model_card_data.model_name if hasattr(self.model, '_model_card_data') else "unknown",
            "indexed_tables": len(set(
                m.get("table_name") for m in self.metadata if "table_name" in m
            ))
        }
