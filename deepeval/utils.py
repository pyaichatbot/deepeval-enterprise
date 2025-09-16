"""
Utility functions and helper classes for the DeepEval framework.

This module contains common utility functions, data processing helpers,
and other shared functionality.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Iterator
from dataclasses import dataclass, asdict
import asyncio
import threading
from contextlib import contextmanager, asynccontextmanager

import pandas as pd
import numpy as np

from .models import TestCase, TestSuite, EvaluationResult


def create_test_suite(name: str, test_cases: List[Dict[str, Any]]) -> TestSuite:
    """Helper function to create test suite from dictionary data."""
    cases: List[TestCase] = []
    for i, d in enumerate(test_cases):
        cases.append(TestCase(
            id=d.get("id", f"test_{i}"),
            input_data=d["input"],
            expected_output=d.get("expected"),
            metadata=d.get("metadata"),
            tags=d.get("tags"),
        ))
    return TestSuite(name=name, test_cases=cases)


def load_test_suite_from_file(file_path: Union[str, Path]) -> TestSuite:
    """Load test suite from JSON file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Test suite file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return TestSuite.from_dict(data)


def save_test_suite_to_file(test_suite: TestSuite, file_path: Union[str, Path]):
    """Save test suite to JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(test_suite.to_dict(), f, indent=2, default=str)


def load_evaluation_results_from_file(file_path: Union[str, Path]) -> List[EvaluationResult]:
    """Load evaluation results from JSON file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return [EvaluationResult(**result) for result in data]


def save_evaluation_results_to_file(results: List[EvaluationResult], file_path: Union[str, Path]):
    """Save evaluation results to JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump([asdict(result) for result in results], f, indent=2, default=str)


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        logging.info(f"{self.name} took {self.duration:.3f} seconds")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or time.time()
        return end_time - self.start_time


class AsyncTimer:
    """Async context manager for timing operations."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        logging.info(f"{self.name} took {self.duration:.3f} seconds")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or time.time()
        return end_time - self.start_time


class RateLimiter:
    """Rate limiter for controlling request frequency."""
    
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """Acquire permission to make a request."""
        with self.lock:
            now = time.time()
            # Remove old requests outside the time window
            self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False
    
    async def acquire_async(self) -> bool:
        """Async version of acquire."""
        return self.acquire()
    
    def wait_time(self) -> float:
        """Get time to wait before next request is allowed."""
        with self.lock:
            if len(self.requests) < self.max_requests:
                return 0.0
            
            oldest_request = min(self.requests)
            return self.time_window - (time.time() - oldest_request)


class BatchProcessor:
    """Process items in batches with configurable batch size and concurrency."""
    
    def __init__(self, batch_size: int = 10, max_concurrent: int = 5):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
    
    def process_batches(self, items: List[Any], processor: Callable) -> List[Any]:
        """Process items in batches synchronously."""
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = processor(batch)
            results.extend(batch_results)
        
        return results
    
    async def process_batches_async(self, items: List[Any], processor: Callable) -> List[Any]:
        """Process items in batches asynchronously."""
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_batch(batch):
            async with semaphore:
                if asyncio.iscoroutinefunction(processor):
                    return await processor(batch)
                else:
                    return processor(batch)
        
        tasks = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            tasks.append(process_batch(batch))
        
        batch_results = await asyncio.gather(*tasks)
        for batch_result in batch_results:
            results.extend(batch_result)
        
        return results


class DataValidator:
    """Validate data structures and formats."""
    
    @staticmethod
    def validate_test_case(test_case_data: Dict[str, Any]) -> List[str]:
        """Validate test case data structure."""
        errors = []
        
        if "input" not in test_case_data:
            errors.append("Test case must have 'input' field")
        
        if "expected" not in test_case_data:
            errors.append("Test case must have 'expected' field")
        
        if not isinstance(test_case_data.get("input"), dict):
            errors.append("Test case 'input' must be a dictionary")
        
        if not isinstance(test_case_data.get("expected"), dict):
            errors.append("Test case 'expected' must be a dictionary")
        
        return errors
    
    @staticmethod
    def validate_test_suite(test_suite_data: Dict[str, Any]) -> List[str]:
        """Validate test suite data structure."""
        errors = []
        
        if "name" not in test_suite_data:
            errors.append("Test suite must have 'name' field")
        
        if "test_cases" not in test_suite_data:
            errors.append("Test suite must have 'test_cases' field")
        
        if not isinstance(test_suite_data.get("test_cases"), list):
            errors.append("Test suite 'test_cases' must be a list")
        
        # Validate individual test cases
        for i, test_case in enumerate(test_suite_data.get("test_cases", [])):
            case_errors = DataValidator.validate_test_case(test_case)
            for error in case_errors:
                errors.append(f"Test case {i}: {error}")
        
        return errors


class ResultAnalyzer:
    """Analyze evaluation results and generate insights."""
    
    def __init__(self):
        self.cache = {}
    
    def analyze_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Comprehensive analysis of evaluation results."""
        
        if not results:
            return {"error": "No results to analyze"}
        
        df = pd.DataFrame([asdict(r) for r in results])
        
        analysis = {
            "overall_statistics": self._calculate_overall_stats(df),
            "metric_analysis": self._analyze_by_metric(df),
            "performance_trends": self._analyze_performance_trends(df),
            "outlier_detection": self._detect_outliers(df),
            "correlation_analysis": self._correlation_analysis(df)
        }
        
        return analysis
    
    def _calculate_overall_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall statistics."""
        return {
            "total_evaluations": len(df),
            "pass_rate": df['passed'].mean(),
            "average_score": df['score'].mean(),
            "score_std": df['score'].std(),
            "score_distribution": {
                "min": df['score'].min(),
                "q25": df['score'].quantile(0.25),
                "median": df['score'].median(),
                "q75": df['score'].quantile(0.75),
                "max": df['score'].max()
            },
            "execution_time_stats": {
                "mean": df['execution_time'].mean(),
                "std": df['execution_time'].std(),
                "min": df['execution_time'].min(),
                "max": df['execution_time'].max()
            }
        }
    
    def _analyze_by_metric(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze results by metric type."""
        metric_analysis = {}
        
        for metric in df['metric_name'].unique():
            metric_data = df[df['metric_name'] == metric]
            metric_analysis[metric] = {
                "count": len(metric_data),
                "pass_rate": metric_data['passed'].mean(),
                "average_score": metric_data['score'].mean(),
                "score_std": metric_data['score'].std(),
                "average_execution_time": metric_data['execution_time'].mean()
            }
        
        return metric_analysis
    
    def _analyze_performance_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance trends over test cases."""
        # Sort by test_case_id to see trends
        df_sorted = df.sort_values('test_case_id')
        
        # Example: Simple moving average of scores
        if len(df_sorted) > 5:
            df_sorted['rolling_avg_score'] = df_sorted['score'].rolling(window=5).mean()
            df_sorted['rolling_avg_latency'] = df_sorted['execution_time'].rolling(window=5).mean()
        
        return {
            "scores_over_time": df_sorted[['test_case_id', 'score', 'rolling_avg_score']].to_dict(orient='records') if 'rolling_avg_score' in df_sorted.columns else df_sorted[['test_case_id', 'score']].to_dict(orient='records'),
            "latency_over_time": df_sorted[['test_case_id', 'execution_time', 'rolling_avg_latency']].to_dict(orient='records') if 'rolling_avg_latency' in df_sorted.columns else df_sorted[['test_case_id', 'execution_time']].to_dict(orient='records'),
        }
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in scores and execution times."""
        outliers = {}
        
        # Score outliers (e.g., below 1st percentile)
        score_lower_bound = df['score'].quantile(0.01)
        outliers['low_score_test_cases'] = df[df['score'] < score_lower_bound][['test_case_id', 'metric_name', 'score']].to_dict(orient='records')
        
        # Latency outliers (e.g., above 99th percentile)
        latency_upper_bound = df['execution_time'].quantile(0.99)
        outliers['high_latency_test_cases'] = df[df['execution_time'] > latency_upper_bound][['test_case_id', 'metric_name', 'execution_time']].to_dict(orient='records')
        
        return outliers
    
    def _correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between metrics or other factors."""
        correlations = {}
        
        # Example: Correlation between score and execution time
        if 'score' in df.columns and 'execution_time' in df.columns:
            correlations['score_vs_execution_time'] = df['score'].corr(df['execution_time'])
        
        # Example: Correlation matrix for numerical metrics
        numerical_metrics = df.select_dtypes(include=[np.number])
        if not numerical_metrics.empty:
            correlations['metric_correlation_matrix'] = numerical_metrics.corr().to_dict()
            
        return correlations


class ProgressTracker:
    """Track progress of long-running operations."""
    
    def __init__(self, total: int, name: str = "operation"):
        self.total = total
        self.current = 0
        self.name = name
        self.start_time = time.time()
        self.callbacks = []
    
    def add_callback(self, callback: Callable[[int, int, float], None]):
        """Add a progress callback function."""
        self.callbacks.append(callback)
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        self._notify_callbacks()
    
    def set_progress(self, current: int):
        """Set current progress."""
        self.current = current
        self._notify_callbacks()
    
    def _notify_callbacks(self):
        """Notify all callbacks of progress update."""
        elapsed = time.time() - self.start_time
        for callback in self.callbacks:
            try:
                callback(self.current, self.total, elapsed)
            except Exception as e:
                logging.error(f"Progress callback error: {e}")
    
    @property
    def percentage(self) -> float:
        """Get progress percentage."""
        return (self.current / self.total) * 100 if self.total > 0 else 0.0
    
    @property
    def eta(self) -> float:
        """Get estimated time to completion."""
        if self.current == 0:
            return float('inf')
        
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed
        remaining = self.total - self.current
        return remaining / rate if rate > 0 else float('inf')


def generate_unique_id(prefix: str = "") -> str:
    """Generate a unique identifier."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uuid_part = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{uuid_part}" if prefix else f"{timestamp}_{uuid_part}"


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Safely serialize object to JSON with proper error handling."""
    try:
        return json.dumps(obj, default=str, **kwargs)
    except (TypeError, ValueError) as e:
        logging.error(f"JSON serialization error: {e}")
        return json.dumps({"error": "Serialization failed", "type": str(type(obj))})


def safe_json_loads(json_str: str) -> Any:
    """Safely deserialize JSON string with proper error handling."""
    try:
        return json.loads(json_str)
    except (TypeError, ValueError) as e:
        logging.error(f"JSON deserialization error: {e}")
        return None


@contextmanager
def suppress_logging(level: int = logging.CRITICAL):
    """Context manager to suppress logging below specified level."""
    logging.disable(level)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


@asynccontextmanager
async def async_suppress_logging(level: int = logging.CRITICAL):
    """Async context manager to suppress logging below specified level."""
    logging.disable(level)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)
