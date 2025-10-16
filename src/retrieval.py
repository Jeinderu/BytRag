"""
Hybrid Retrieval and Re-ranking implementation.
Combines Dense (semantic) and Sparse (BM25) retrieval with Reciprocal Rank Fusion.
"""

import logging
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Implements Hybrid Retrieval combining dense semantic search and sparse BM25 search.
    Uses Reciprocal Rank Fusion (RRF) for re-ranking.
    """
    
    def __init__(
        self,
        vectorstore,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        final_top_k: int = 5,
        rrf_k: int = 60
    ):
        """
        Initialize the Hybrid Retriever.
        
        Args:
            vectorstore: ChromaDB vector store instance
            dense_top_k: Number of documents to retrieve via dense search
            sparse_top_k: Number of documents to retrieve via BM25
            final_top_k: Final number of documents after re-ranking
            rrf_k: RRF constant (typically 60)
        """
        self.vectorstore = vectorstore
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k
        
        # BM25 index will be initialized when documents are loaded
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_metadatas = []
        
        logger.info("HybridRetriever initialized")
    
    def initialize_bm25_index(self, documents: List[Any]) -> None:
        """
        Initialize the BM25 index from documents.
        
        Args:
            documents: List of Document objects with page_content and metadata
        """
        logger.info("Initializing BM25 index...")
        
        # Extract text content and metadata
        self.bm25_documents = []
        self.bm25_metadatas = []
        tokenized_corpus = []
        
        for doc in documents:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            self.bm25_documents.append(content)
            self.bm25_metadatas.append(metadata)
            
            # Simple tokenization (split by whitespace and lowercase)
            tokenized_corpus.append(content.lower().split())
        
        # Create BM25 index
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        logger.info(f"BM25 index created with {len(self.bm25_documents)} documents")
    
    def dense_retrieval(self, query: str, k: int = None) -> List[Tuple[Any, float]]:
        """
        Perform dense semantic retrieval using the vector store.
        
        Args:
            query: Search query
            k: Number of documents to retrieve (defaults to self.dense_top_k)
            
        Returns:
            List of (document, score) tuples
        """
        k = k or self.dense_top_k
        
        try:
            # Perform similarity search with scores
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            logger.info(f"Dense retrieval found {len(results)} documents")
            return results
        
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []
    
    def sparse_retrieval(self, query: str, k: int = None) -> List[Tuple[str, int, float]]:
        """
        Perform sparse BM25 retrieval.
        
        Args:
            query: Search query
            k: Number of documents to retrieve (defaults to self.sparse_top_k)
            
        Returns:
            List of (document_content, doc_index, score) tuples
        """
        k = k or self.sparse_top_k
        
        if self.bm25_index is None:
            logger.error("BM25 index not initialized")
            return []
        
        try:
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top K indices
            top_k_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for idx in top_k_indices:
                if scores[idx] > 0:  # Only include documents with positive scores
                    results.append((
                        self.bm25_documents[idx],
                        int(idx),
                        float(scores[idx])
                    ))
            
            logger.info(f"Sparse retrieval found {len(results)} documents")
            return results
        
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            return []
    
    def reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[Any, float]],
        sparse_results: List[Tuple[str, int, float]]
    ) -> List[Any]:
        """
        Apply Reciprocal Rank Fusion to combine and re-rank results.
        
        Formula: RRF_score(d) = Σ(1 / (k + rank_i(d))) for all retrievers i
        
        Args:
            dense_results: Results from dense retrieval (document, score)
            sparse_results: Results from sparse retrieval (content, index, score)
            
        Returns:
            Re-ranked list of documents (top K)
        """
        logger.info("Applying Reciprocal Rank Fusion...")
        
        # Dictionary to store RRF scores: doc_id -> (score, document)
        rrf_scores = defaultdict(float)
        doc_map = {}  # doc_id -> actual document object
        
        # Process dense results
        for rank, (doc, score) in enumerate(dense_results, start=1):
            # Create a unique ID for the document
            doc_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            doc_id = hash(doc_content[:200])  # Use hash of first 200 chars as ID
            
            # Calculate RRF contribution
            rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank)
            
            # Store document
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
        
        # Process sparse results
        for rank, (content, idx, score) in enumerate(sparse_results, start=1):
            doc_id = hash(content[:200])
            
            # Calculate RRF contribution
            rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank)
            
            # Store document if not already present (create from BM25 result)
            if doc_id not in doc_map:
                # Create a simple document-like object
                from types import SimpleNamespace
                doc = SimpleNamespace(
                    page_content=content,
                    metadata=self.bm25_metadatas[idx] if idx < len(self.bm25_metadatas) else {}
                )
                doc_map[doc_id] = doc
        
        # Sort by RRF score (descending)
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top K documents
        top_k_doc_ids = [doc_id for doc_id, score in sorted_docs[:self.final_top_k]]
        top_k_docs = [doc_map[doc_id] for doc_id in top_k_doc_ids]
        
        logger.info(f"RRF produced top {len(top_k_docs)} documents")
        
        # Log RRF scores for debugging
        for i, (doc_id, score) in enumerate(sorted_docs[:self.final_top_k], 1):
            logger.debug(f"  Rank {i}: RRF Score = {score:.4f}")
        
        return top_k_docs
    
    def retrieve(self, query: str) -> List[Any]:
        """
        Perform hybrid retrieval with RRF re-ranking.
        
        Args:
            query: User query
            
        Returns:
            Top K re-ranked documents
        """
        logger.info(f"Hybrid retrieval for query: '{query[:100]}...'")
        
        # Step 1: Dense retrieval
        dense_results = self.dense_retrieval(query)
        
        # Step 2: Sparse retrieval
        sparse_results = self.sparse_retrieval(query)
        
        # Step 3: Re-ranking with RRF
        if not dense_results and not sparse_results:
            logger.warning("No results from either retriever")
            return []
        
        final_docs = self.reciprocal_rank_fusion(dense_results, sparse_results)
        
        return final_docs


def create_hybrid_retriever(
    vectorstore,
    documents: List[Any],
    dense_top_k: int = 50,
    sparse_top_k: int = 50,
    final_top_k: int = 5,
    rrf_k: int = 60
) -> HybridRetriever:
    """
    Factory function to create and initialize a HybridRetriever.
    
    Args:
        vectorstore: ChromaDB vector store
        documents: All documents for BM25 indexing
        dense_top_k: Number of dense retrieval results
        sparse_top_k: Number of sparse retrieval results
        final_top_k: Final number after re-ranking
        rrf_k: RRF constant
        
    Returns:
        Initialized HybridRetriever instance
    """
    retriever = HybridRetriever(
        vectorstore=vectorstore,
        dense_top_k=dense_top_k,
        sparse_top_k=sparse_top_k,
        final_top_k=final_top_k,
        rrf_k=rrf_k
    )
    
    # Initialize BM25 index
    retriever.initialize_bm25_index(documents)
    
    return retriever

