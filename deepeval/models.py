"""
Core data models and structures for the DeepEval framework.

This module contains all the fundamental data structures, enums, and models
used throughout the evaluation framework.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Protocol
import numpy as np


class EvaluationStatus(Enum):
    """Status of an evaluation run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MetricType(Enum):
    """Types of metrics supported by the framework."""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    FACTUAL_CONSISTENCY = "factual_consistency"
    BIAS = "bias"
    TOXICITY = "toxicity"
    HALLUCINATION = "hallucination"
    CUSTOM = "custom"
    LATENCY = "latency"
    # Retrieval metrics
    RECALL_AT_K = "recall_at_k"
    MRR = "mrr"
    NDCG_AT_K = "ndcg_at_k"


@dataclass
class TestCase:
    """Individual test case for evaluation."""
    id: str
    input_data: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []


@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    test_case_id: str
    metric_name: str
    score: float
    passed: bool
    threshold: Optional[float] = None
    explanation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TestSuite:
    """Collection of test cases with configuration."""
    
    def __init__(
        self,
        name: str,
        test_cases: List[TestCase],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        id: Optional[str] = None
    ):
        self.id = id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.test_cases = test_cases
        self.tags = tags or []
        self.created_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert TestSuite to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "test_cases": [asdict(tc) for tc in self.test_cases],
            "tags": self.tags,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestSuite":
        """Create TestSuite from dictionary."""
        test_cases = [
            TestCase(
                id=tc_data.get("id", str(uuid.uuid4())),
                input_data=tc_data["input_data"],
                expected_output=tc_data.get("expected_output"),
                metadata=tc_data.get("metadata"),
                tags=tc_data.get("tags")
            )
            for tc_data in data.get("test_cases", [])
        ]
        
        return cls(
            id=data.get("id"),
            name=data["name"],
            description=data.get("description"),
            test_cases=test_cases,
            tags=data.get("tags")
        )


class Metric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, name: str, threshold: Optional[float] = None):
        self.name = name
        self.threshold = threshold
        self.metadata = {}

    @abstractmethod
    async def evaluate(self, test_case: TestCase, actual_output: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a test case and return result."""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate metric configuration."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary representation."""
        return {
            "name": self.name,
            "threshold": self.threshold,
            "metadata": self.metadata
        }


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM."""
        pass


class RetrievalTestCase(TestCase):
    """Specialized test case for retrieval evaluation."""
    
    def __init__(
        self,
        id: str,
        query: str,
        relevant_documents: List[Dict[str, Any]],
        retrieved_documents: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        # Convert to standard TestCase format
        input_data = {"query": query}
        expected_output = {
            "relevant_documents": relevant_documents,
            "retrieved_documents": retrieved_documents
        }
        
        super().__init__(
            id=id,
            input_data=input_data,
            expected_output=expected_output,
            metadata=metadata,
            tags=tags
        )
        
        self.query = query
        self.relevant_documents = relevant_documents
        self.retrieved_documents = retrieved_documents


class RetrievalMetric(Metric):
    """Abstract base class for retrieval-specific metrics."""
    
    def __init__(self, name: str, k: int = 10, threshold: Optional[float] = None):
        super().__init__(name, threshold)
        self.k = k

    @abstractmethod
    async def evaluate_retrieval(
        self, 
        test_case: RetrievalTestCase, 
        actual_output: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate retrieval performance."""
        pass

    async def evaluate(self, test_case: TestCase, actual_output: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a test case - converts to retrieval evaluation if needed."""
        if isinstance(test_case, RetrievalTestCase):
            return await self.evaluate_retrieval(test_case, actual_output)
        
        # Convert standard test case to retrieval test case
        retrieval_tc = RetrievalTestCase(
            id=test_case.id,
            query=test_case.input_data.get("query", ""),
            relevant_documents=test_case.expected_output.get("relevant_documents", []),
            retrieved_documents=actual_output.get("retrieved_documents", []),
            metadata=test_case.metadata,
            tags=test_case.tags
        )
        
        return await self.evaluate_retrieval(retrieval_tc, actual_output)

    def validate_config(self) -> bool:
        """Validate retrieval metric configuration."""
        return isinstance(self.k, int) and self.k > 0


class EvaluationSummary:
    """Summary of evaluation results."""
    
    def __init__(self, results: List[EvaluationResult]):
        self.results = results
        self.total_evaluations = len(results)
        self.passed = sum(1 for r in results if r.passed)
        self.failed = self.total_evaluations - self.passed
        self.pass_rate = self.passed / self.total_evaluations if self.total_evaluations > 0 else 0.0
        self.average_score = sum(r.score for r in results) / self.total_evaluations if self.total_evaluations > 0 else 0.0
        self.metrics_summary = self._calculate_metrics_summary()

    def _calculate_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Calculate summary statistics by metric."""
        metrics_summary = {}
        
        for result in self.results:
            metric_name = result.metric_name
            if metric_name not in metrics_summary:
                metrics_summary[metric_name] = {
                    "count": 0,
                    "passed": 0,
                    "scores": [],
                    "execution_times": []
                }
            
            metrics_summary[metric_name]["count"] += 1
            if result.passed:
                metrics_summary[metric_name]["passed"] += 1
            metrics_summary[metric_name]["scores"].append(result.score)
            if result.execution_time:
                metrics_summary[metric_name]["execution_times"].append(result.execution_time)
        
        # Calculate averages
        for metric_name, data in metrics_summary.items():
            data["pass_rate"] = data["passed"] / data["count"] if data["count"] > 0 else 0.0
            data["average_score"] = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0.0
            data["min_score"] = min(data["scores"]) if data["scores"] else 0.0
            data["max_score"] = max(data["scores"]) if data["scores"] else 0.0
            data["average_execution_time"] = sum(data["execution_times"]) / len(data["execution_times"]) if data["execution_times"] else 0.0
            
            # Remove raw lists to keep summary clean
            del data["scores"]
            del data["execution_times"]
        
        return metrics_summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "total_evaluations": self.total_evaluations,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "average_score": self.average_score,
            "metrics": self.metrics_summary
        }
