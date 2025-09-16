"""
Metrics module for the DeepEval framework.

This module provides all evaluation metrics including built-in metrics,
retrieval metrics, and custom metric registration capabilities.
"""

from .base import (
    Metric,
    RetrievalMetric,
    LLMBasedMetric,
    CompositeMetric,
    ThresholdMetric,
    StatisticalMetric,
    MetricRegistry,
    get_metric_registry,
    register_metric
)

from .builtin import (
    AccuracyMetric,
    RelevanceMetric,
    CoherenceMetric,
    LatencyMetric,
    FluencyMetric,
    FactualConsistencyMetric,
    BiasDetectionMetric,
    ToxicityMetric,
    SemanticSimilarityMetric
)

from .retrieval import (
    RecallAtKMetric,
    MRRMetric,
    NDCGAtKMetric,
    PrecisionAtKMetric,
    F1AtKMetric
)

__all__ = [
    # Base classes
    "Metric",
    "RetrievalMetric",
    "LLMBasedMetric",
    "CompositeMetric",
    "ThresholdMetric",
    "StatisticalMetric",
    "MetricRegistry",
    "get_metric_registry",
    "register_metric",
    
    # Built-in metrics
    "AccuracyMetric",
    "RelevanceMetric",
    "CoherenceMetric",
    "LatencyMetric",
    "FluencyMetric",
    "FactualConsistencyMetric",
    "BiasDetectionMetric",
    "ToxicityMetric",
    "SemanticSimilarityMetric",
    
    # Retrieval metrics
    "RecallAtKMetric",
    "MRRMetric",
    "NDCGAtKMetric",
    "PrecisionAtKMetric",
    "F1AtKMetric"
]
