"""
Retrieval evaluation metrics for the DeepEval framework.

This module contains specialized metrics for evaluating retrieval systems,
including Recall@K, MRR (Mean Reciprocal Rank), and nDCG@K (Normalized
Discounted Cumulative Gain).
"""

import time
import logging
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np

from .base import RetrievalMetric, register_metric
from ..models import RetrievalTestCase, EvaluationResult


class RecallAtKMetric(RetrievalMetric):
    """
    Recall@K metric for retrieval evaluation.
    
    Recall@K measures the proportion of relevant documents that are retrieved
    in the top K results.
    """

    def __init__(self, k: int = 10, threshold: float = 0.5):
        super().__init__("recall_at_k", k, threshold)

    async def evaluate_retrieval(
        self, 
        test_case: RetrievalTestCase, 
        actual_output: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate Recall@K for retrieval."""
        start_time = time.time()
        
        # Get relevant and retrieved documents
        relevant_docs = self._extract_document_ids(test_case.relevant_documents)
        retrieved_docs = self._extract_document_ids(
            actual_output.get("retrieved_documents", [])
        )
        
        # Calculate Recall@K
        score = self._calculate_recall_at_k(relevant_docs, retrieved_docs, self.k)
        passed = score >= self.threshold
        
        execution_time = time.time() - start_time
        
        # Calculate intersection for explanation
        relevant_set = set(relevant_docs)
        relevant_retrieved_count = len(relevant_set.intersection(set(retrieved_docs[:self.k])))
        
        explanation = (
            f"Recall@{self.k}: {score:.3f}. "
            f"Relevant docs: {len(relevant_docs)}, "
            f"Retrieved in top {self.k}: {relevant_retrieved_count}"
        )
        
        return self._create_result(
            test_case_id=test_case.id,
            score=score,
            explanation=explanation,
            metadata={
                "k": self.k,
                "relevant_count": len(relevant_docs),
                "retrieved_count": len(retrieved_docs),
                "relevant_retrieved": relevant_retrieved_count
            },
            execution_time=execution_time
        )

    def _extract_document_ids(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract document IDs from document list."""
        doc_ids = []
        for doc in documents:
            if isinstance(doc, dict):
                # Try different possible ID fields
                doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id")
                if doc_id:
                    doc_ids.append(str(doc_id))
                else:
                    # Use content hash as fallback
                    content = doc.get("content", "") or doc.get("text", "")
                    doc_ids.append(str(hash(content)))
            else:
                # If document is just a string, use it as ID
                doc_ids.append(str(doc))
        return doc_ids

    def _calculate_recall_at_k(self, relevant_docs: List[str], retrieved_docs: List[str], k: int) -> float:
        """Calculate Recall@K score."""
        if not relevant_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        retrieved_top_k = retrieved_docs[:k]
        
        # Count how many relevant documents are in the top K retrieved
        relevant_retrieved = len(relevant_set.intersection(set(retrieved_top_k)))
        
        # Recall@K = |Relevant ∩ Retrieved@K| / |Relevant|
        recall = relevant_retrieved / len(relevant_set)
        return recall


class MRRMetric(RetrievalMetric):
    """
    Mean Reciprocal Rank (MRR) metric for retrieval evaluation.
    
    MRR measures the average of the reciprocal ranks of the first relevant
    document for each query.
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__("mrr", k=1, threshold=threshold)  # MRR doesn't use K

    async def evaluate_retrieval(
        self, 
        test_case: RetrievalTestCase, 
        actual_output: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate MRR for retrieval."""
        start_time = time.time()
        
        # Get relevant and retrieved documents
        relevant_docs = self._extract_document_ids(test_case.relevant_documents)
        retrieved_docs = self._extract_document_ids(
            actual_output.get("retrieved_documents", [])
        )
        
        # Calculate MRR
        score = self._calculate_mrr(relevant_docs, retrieved_docs)
        passed = score >= self.threshold
        
        execution_time = time.time() - start_time
        
        # Find the rank of the first relevant document
        relevant_set = set(relevant_docs)
        first_relevant_rank = None
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_set:
                first_relevant_rank = i + 1  # 1-indexed rank
                break
        
        explanation = (
            f"MRR: {score:.3f}. "
            f"First relevant document at rank: {first_relevant_rank or 'N/A'}"
        )
        
        return self._create_result(
            test_case_id=test_case.id,
            score=score,
            explanation=explanation,
            metadata={
                "first_relevant_rank": first_relevant_rank,
                "relevant_count": len(relevant_docs),
                "retrieved_count": len(retrieved_docs)
            },
            execution_time=execution_time
        )

    def _extract_document_ids(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract document IDs from document list."""
        doc_ids = []
        for doc in documents:
            if isinstance(doc, dict):
                doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id")
                if doc_id:
                    doc_ids.append(str(doc_id))
                else:
                    content = doc.get("content", "") or doc.get("text", "")
                    doc_ids.append(str(hash(content)))
            else:
                doc_ids.append(str(doc))
        return doc_ids

    def _calculate_mrr(self, relevant_docs: List[str], retrieved_docs: List[str]) -> float:
        """Calculate MRR score."""
        if not relevant_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        
        # Find the rank of the first relevant document
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)  # Reciprocal rank (1-indexed)
        
        # No relevant documents found
        return 0.0


class NDCGAtKMetric(RetrievalMetric):
    """
    Normalized Discounted Cumulative Gain at K (nDCG@K) metric for retrieval evaluation.
    
    nDCG@K measures the quality of ranking by considering both relevance and position,
    normalized by the ideal ranking.
    """

    def __init__(self, k: int = 10, threshold: float = 0.5):
        super().__init__("ndcg_at_k", k, threshold)

    async def evaluate_retrieval(
        self, 
        test_case: RetrievalTestCase, 
        actual_output: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate nDCG@K for retrieval."""
        start_time = time.time()
        
        # Get relevant and retrieved documents with relevance scores
        relevant_docs = self._extract_documents_with_relevance(test_case.relevant_documents)
        retrieved_docs = self._extract_documents_with_relevance(
            actual_output.get("retrieved_documents", [])
        )
        
        # Calculate nDCG@K
        score = self._calculate_ndcg_at_k(relevant_docs, retrieved_docs, self.k)
        passed = score >= self.threshold
        
        execution_time = time.time() - start_time
        
        explanation = (
            f"nDCG@{self.k}: {score:.3f}. "
            f"Relevant docs: {len(relevant_docs)}, "
            f"Retrieved docs: {len(retrieved_docs)}"
        )
        
        return self._create_result(
            test_case_id=test_case.id,
            score=score,
            explanation=explanation,
            metadata={
                "k": self.k,
                "relevant_count": len(relevant_docs),
                "retrieved_count": len(retrieved_docs)
            },
            execution_time=execution_time
        )

    def _extract_documents_with_relevance(self, documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract documents with their relevance scores."""
        doc_relevance = {}
        
        for doc in documents:
            if isinstance(doc, dict):
                # Get document ID
                doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id")
                if not doc_id:
                    content = doc.get("content", "") or doc.get("text", "")
                    doc_id = str(hash(content))
                
                # Get relevance score (default to 1.0 if not specified)
                relevance = doc.get("relevance", 1.0)
                if not isinstance(relevance, (int, float)):
                    relevance = 1.0
                
                doc_relevance[str(doc_id)] = float(relevance)
            else:
                # If document is just a string, treat as ID with relevance 1.0
                doc_relevance[str(doc)] = 1.0
        
        return doc_relevance

    def _calculate_ndcg_at_k(self, relevant_docs: Dict[str, float], retrieved_docs: Dict[str, float], k: int) -> float:
        """Calculate nDCG@K score."""
        if not relevant_docs:
            return 0.0
        
        # Get top K retrieved documents
        retrieved_list = list(retrieved_docs.keys())[:k]
        
        # Calculate DCG@K
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_list):
            relevance = relevant_docs.get(doc_id, 0.0)
            if relevance > 0:
                dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG@K (Ideal DCG)
        ideal_relevances = sorted(relevant_docs.values(), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances):
            idcg += relevance / np.log2(i + 2)
        
        # Calculate nDCG@K
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        return ndcg


class PrecisionAtKMetric(RetrievalMetric):
    """
    Precision@K metric for retrieval evaluation.
    
    Precision@K measures the proportion of retrieved documents in the top K
    that are relevant.
    """

    def __init__(self, k: int = 10, threshold: float = 0.5):
        super().__init__("precision_at_k", k, threshold)

    async def evaluate_retrieval(
        self, 
        test_case: RetrievalTestCase, 
        actual_output: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate Precision@K for retrieval."""
        start_time = time.time()
        
        # Get relevant and retrieved documents
        relevant_docs = self._extract_document_ids(test_case.relevant_documents)
        retrieved_docs = self._extract_document_ids(
            actual_output.get("retrieved_documents", [])
        )
        
        # Calculate Precision@K
        score = self._calculate_precision_at_k(relevant_docs, retrieved_docs, self.k)
        passed = score >= self.threshold
        
        execution_time = time.time() - start_time
        
        relevant_retrieved = len(set(relevant_docs).intersection(set(retrieved_docs[:self.k])))
        
        explanation = (
            f"Precision@{self.k}: {score:.3f}. "
            f"Relevant in top {self.k}: {relevant_retrieved}/{min(self.k, len(retrieved_docs))}"
        )
        
        return self._create_result(
            test_case_id=test_case.id,
            score=score,
            explanation=explanation,
            metadata={
                "k": self.k,
                "relevant_count": len(relevant_docs),
                "retrieved_count": len(retrieved_docs),
                "relevant_retrieved": relevant_retrieved
            },
            execution_time=execution_time
        )

    def _extract_document_ids(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract document IDs from document list."""
        doc_ids = []
        for doc in documents:
            if isinstance(doc, dict):
                doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id")
                if doc_id:
                    doc_ids.append(str(doc_id))
                else:
                    content = doc.get("content", "") or doc.get("text", "")
                    doc_ids.append(str(hash(content)))
            else:
                doc_ids.append(str(doc))
        return doc_ids

    def _calculate_precision_at_k(self, relevant_docs: List[str], retrieved_docs: List[str], k: int) -> float:
        """Calculate Precision@K score."""
        if not retrieved_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        retrieved_top_k = retrieved_docs[:k]
        
        # Count how many relevant documents are in the top K retrieved
        relevant_retrieved = len(relevant_set.intersection(set(retrieved_top_k)))
        
        # Precision@K = |Relevant ∩ Retrieved@K| / |Retrieved@K|
        precision = relevant_retrieved / len(retrieved_top_k)
        return precision


class F1AtKMetric(RetrievalMetric):
    """
    F1@K metric for retrieval evaluation.
    
    F1@K is the harmonic mean of Precision@K and Recall@K.
    """

    def __init__(self, k: int = 10, threshold: float = 0.5):
        super().__init__("f1_at_k", k, threshold)

    async def evaluate_retrieval(
        self, 
        test_case: RetrievalTestCase, 
        actual_output: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate F1@K for retrieval."""
        start_time = time.time()
        
        # Get relevant and retrieved documents
        relevant_docs = self._extract_document_ids(test_case.relevant_documents)
        retrieved_docs = self._extract_document_ids(
            actual_output.get("retrieved_documents", [])
        )
        
        # Calculate Precision@K and Recall@K
        precision = self._calculate_precision_at_k(relevant_docs, retrieved_docs, self.k)
        recall = self._calculate_recall_at_k(relevant_docs, retrieved_docs, self.k)
        
        # Calculate F1@K
        if precision + recall == 0:
            score = 0.0
        else:
            score = 2 * (precision * recall) / (precision + recall)
        
        passed = score >= self.threshold
        
        execution_time = time.time() - start_time
        
        explanation = (
            f"F1@{self.k}: {score:.3f} "
            f"(Precision: {precision:.3f}, Recall: {recall:.3f})"
        )
        
        return self._create_result(
            test_case_id=test_case.id,
            score=score,
            explanation=explanation,
            metadata={
                "k": self.k,
                "precision": precision,
                "recall": recall,
                "relevant_count": len(relevant_docs),
                "retrieved_count": len(retrieved_docs)
            },
            execution_time=execution_time
        )

    def _extract_document_ids(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract document IDs from document list."""
        doc_ids = []
        for doc in documents:
            if isinstance(doc, dict):
                doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id")
                if doc_id:
                    doc_ids.append(str(doc_id))
                else:
                    content = doc.get("content", "") or doc.get("text", "")
                    doc_ids.append(str(hash(content)))
            else:
                doc_ids.append(str(doc))
        return doc_ids

    def _calculate_precision_at_k(self, relevant_docs: List[str], retrieved_docs: List[str], k: int) -> float:
        """Calculate Precision@K score."""
        if not retrieved_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        retrieved_top_k = retrieved_docs[:k]
        
        relevant_retrieved = len(relevant_set.intersection(set(retrieved_top_k)))
        precision = relevant_retrieved / len(retrieved_top_k)
        return precision

    def _calculate_recall_at_k(self, relevant_docs: List[str], retrieved_docs: List[str], k: int) -> float:
        """Calculate Recall@K score."""
        if not relevant_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        retrieved_top_k = retrieved_docs[:k]
        
        relevant_retrieved = len(relevant_set.intersection(set(retrieved_top_k)))
        recall = relevant_retrieved / len(relevant_set)
        return recall


# Register all retrieval metrics
register_metric(RecallAtKMetric, "retrieval")
register_metric(MRRMetric, "retrieval")
register_metric(NDCGAtKMetric, "retrieval")
register_metric(PrecisionAtKMetric, "retrieval")
register_metric(F1AtKMetric, "retrieval")
