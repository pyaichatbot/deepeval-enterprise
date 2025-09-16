"""
Base metric interfaces and abstract classes for the DeepEval framework.

This module contains the fundamental interfaces and abstract base classes
that all metrics must implement.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Protocol
from dataclasses import dataclass

from ..models import TestCase, EvaluationResult, RetrievalTestCase, RetrievalMetric


class Metric(ABC):
    """Abstract base class for all evaluation metrics."""

    def __init__(self, name: str, threshold: Optional[float] = None):
        self.name = name
        self.threshold = threshold
        self.metadata = {}
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    async def evaluate(self, test_case: TestCase, actual_output: Dict[str, Any]) -> EvaluationResult:
        """
        Evaluate a test case and return the result.
        
        Args:
            test_case: The test case to evaluate
            actual_output: The actual output from the system under test
            
        Returns:
            EvaluationResult containing the evaluation outcome
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate the metric configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary representation."""
        return {
            "name": self.name,
            "threshold": self.threshold,
            "metadata": self.metadata,
            "type": self.__class__.__name__
        }

    def _create_result(
        self,
        test_case_id: str,
        score: float,
        explanation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None
    ) -> EvaluationResult:
        """Helper method to create evaluation results."""
        passed = score >= self.threshold if self.threshold is not None else True
        
        return EvaluationResult(
            test_case_id=test_case_id,
            metric_name=self.name,
            score=score,
            passed=passed,
            threshold=self.threshold,
            explanation=explanation,
            metadata=metadata or {},
            execution_time=execution_time
        )

    def _measure_execution_time(self, func: Callable) -> Callable:
        """Decorator to measure execution time of metric evaluation."""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                if hasattr(result, 'execution_time'):
                    result.execution_time = execution_time
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(f"Metric {self.name} failed after {execution_time:.3f}s: {e}")
                raise
        
        return wrapper


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
        """
        Evaluate retrieval performance.
        
        Args:
            test_case: The retrieval test case
            actual_output: The actual retrieval results
            
        Returns:
            EvaluationResult containing the retrieval evaluation
        """
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary representation."""
        base_dict = super().to_dict()
        base_dict["k"] = self.k
        return base_dict


class LLMBasedMetric(Metric):
    """Abstract base class for metrics that require LLM evaluation."""
    
    def __init__(self, name: str, llm_provider: 'LLMProvider', threshold: Optional[float] = None):
        super().__init__(name, threshold)
        self.llm_provider = llm_provider

    def validate_config(self) -> bool:
        """Validate LLM-based metric configuration."""
        return self.llm_provider is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary representation."""
        base_dict = super().to_dict()
        base_dict["llm_provider"] = str(type(self.llm_provider).__name__)
        return base_dict


class CompositeMetric(Metric):
    """Abstract base class for metrics that combine multiple sub-metrics."""
    
    def __init__(self, name: str, sub_metrics: List[Metric], threshold: Optional[float] = None):
        super().__init__(name, threshold)
        self.sub_metrics = sub_metrics

    @abstractmethod
    def combine_results(self, results: List[EvaluationResult]) -> EvaluationResult:
        """
        Combine results from sub-metrics into a single result.
        
        Args:
            results: List of results from sub-metrics
            
        Returns:
            Combined evaluation result
        """
        pass

    async def evaluate(self, test_case: TestCase, actual_output: Dict[str, Any]) -> EvaluationResult:
        """Evaluate using all sub-metrics and combine results."""
        sub_results = []
        
        for metric in self.sub_metrics:
            try:
                result = await metric.evaluate(test_case, actual_output)
                sub_results.append(result)
            except Exception as e:
                self.logger.error(f"Sub-metric {metric.name} failed: {e}")
                # Create a failed result for the sub-metric
                failed_result = self._create_result(
                    test_case_id=test_case.id,
                    score=0.0,
                    explanation=f"Sub-metric failed: {str(e)}"
                )
                sub_results.append(failed_result)
        
        return self.combine_results(sub_results)

    def validate_config(self) -> bool:
        """Validate composite metric configuration."""
        return len(self.sub_metrics) > 0 and all(metric.validate_config() for metric in self.sub_metrics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary representation."""
        base_dict = super().to_dict()
        base_dict["sub_metrics"] = [metric.to_dict() for metric in self.sub_metrics]
        return base_dict


class ThresholdMetric(Metric):
    """Base class for metrics that use simple threshold-based evaluation."""
    
    def __init__(self, name: str, threshold: float = 0.5):
        super().__init__(name, threshold)

    def validate_config(self) -> bool:
        """Validate threshold metric configuration."""
        return isinstance(self.threshold, (int, float)) and 0 <= self.threshold <= 1

    def _normalize_score(self, score: float) -> float:
        """Normalize score to 0-1 range."""
        return max(0.0, min(1.0, score))


class StatisticalMetric(Metric):
    """Base class for metrics that perform statistical analysis."""
    
    def __init__(self, name: str, threshold: Optional[float] = None):
        super().__init__(name, threshold)
        self.statistics = {}

    def update_statistics(self, result: EvaluationResult):
        """Update internal statistics with new result."""
        if self.name not in self.statistics:
            self.statistics[self.name] = {
                "count": 0,
                "total_score": 0.0,
                "passed_count": 0,
                "execution_times": []
            }
        
        stats = self.statistics[self.name]
        stats["count"] += 1
        stats["total_score"] += result.score
        if result.passed:
            stats["passed_count"] += 1
        if result.execution_time:
            stats["execution_times"].append(result.execution_time)

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics for this metric."""
        if self.name not in self.statistics:
            return {}
        
        stats = self.statistics[self.name]
        return {
            "count": stats["count"],
            "average_score": stats["total_score"] / stats["count"] if stats["count"] > 0 else 0.0,
            "pass_rate": stats["passed_count"] / stats["count"] if stats["count"] > 0 else 0.0,
            "average_execution_time": sum(stats["execution_times"]) / len(stats["execution_times"]) if stats["execution_times"] else 0.0
        }

    def validate_config(self) -> bool:
        """Validate statistical metric configuration."""
        return True


class MetricRegistry:
    """Registry for managing available metrics."""
    
    def __init__(self):
        self._metrics = {}
        self._categories = {}

    def register(self, metric_class: type, category: str = "general"):
        """Register a metric class."""
        if not issubclass(metric_class, Metric):
            raise ValueError("Metric must inherit from Metric base class")
        
        self._metrics[metric_class.__name__] = metric_class
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(metric_class.__name__)

    def create_metric(self, metric_name: str, **kwargs) -> Metric:
        """Create an instance of a registered metric."""
        if metric_name not in self._metrics:
            raise ValueError(f"Metric {metric_name} not registered")
        
        return self._metrics[metric_name](**kwargs)

    def list_metrics(self, category: Optional[str] = None) -> List[str]:
        """List available metrics, optionally filtered by category."""
        if category:
            return self._categories.get(category, [])
        return list(self._metrics.keys())

    def get_metric_info(self, metric_name: str) -> Dict[str, Any]:
        """Get information about a metric."""
        if metric_name not in self._metrics:
            raise ValueError(f"Metric {metric_name} not registered")
        
        metric_class = self._metrics[metric_name]
        return {
            "name": metric_name,
            "class": metric_class,
            "description": metric_class.__doc__ or "No description available",
            "parameters": list(metric_class.__init__.__code__.co_varnames[1:])
        }


# Global metric registry
_metric_registry = MetricRegistry()


def get_metric_registry() -> MetricRegistry:
    """Get the global metric registry."""
    return _metric_registry


def register_metric(metric_class: type, category: str = "general"):
    """Register a metric class in the global registry."""
    _metric_registry.register(metric_class, category)
