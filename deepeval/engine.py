"""
Core evaluation engine for the DeepEval framework.

This module contains the main evaluation engine that orchestrates the evaluation
process, manages metrics, and handles parallel execution.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from contextlib import asynccontextmanager

from .models import TestCase, TestSuite, EvaluationResult, EvaluationSummary
from .metrics.base import Metric, get_metric_registry
from .database import DatabaseManager, DatabaseRepository, get_database_manager
from .config import EvaluationConfig


class EvaluationEngine:
    """Main evaluation engine for running evaluations."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics: List[Metric] = []
        self.db_manager: Optional[DatabaseManager] = None
        self.repository: Optional[DatabaseRepository] = None
        self.logger = self._setup_logging()
        self._lock = threading.Lock()
        
        # Initialize database if configured
        if config.database_url:
            self.db_manager = DatabaseManager(config.database_url, echo=config.debug)
            self.repository = DatabaseRepository(self.db_manager)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the evaluation engine."""
        logger = logging.getLogger("deepeval.engine")
        logger.setLevel(logging.INFO if not self.config.debug else logging.DEBUG)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def add_metric(self, metric: Metric):
        """Add a metric to the evaluation pipeline."""
        with self._lock:
            if not metric.validate_config():
                raise ValueError(f"Invalid configuration for metric: {metric.name}")
            
            self.metrics.append(metric)
            self.logger.info(f"Added metric: {metric.name}")

    def remove_metric(self, metric_name: str):
        """Remove a metric from the pipeline."""
        with self._lock:
            self.metrics = [m for m in self.metrics if m.name != metric_name]
            self.logger.info(f"Removed metric: {metric_name}")

    def clear_metrics(self):
        """Clear all metrics from the pipeline."""
        with self._lock:
            self.metrics.clear()
            self.logger.info("Cleared all metrics")

    def get_metrics(self) -> List[Metric]:
        """Get all metrics in the pipeline."""
        with self._lock:
            return self.metrics.copy()

    async def evaluate(
        self, 
        test_suite: TestSuite, 
        system_under_test: Callable,
        run_name: Optional[str] = None,
        save_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        Run evaluation on a test suite.
        
        Args:
            test_suite: The test suite to evaluate
            system_under_test: Function that takes input and returns output
            run_name: Optional name for this evaluation run
            save_to_db: Whether to save results to database
            
        Returns:
            Dictionary containing evaluation results and summary
        """
        run_id = str(uuid.uuid4())
        run_name = run_name or f"evaluation_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting evaluation run: {run_name} ({run_id})")
        
        # Create evaluation run record
        eval_run_data = {
            "id": run_id,
            "name": run_name,
            "status": "running",
            "test_suite_id": test_suite.id,
            "configuration": self.config.to_dict(),
            "started_at": datetime.now(timezone.utc),
            "total_tests": len(test_suite.test_cases)
        }
        
        if save_to_db and self.repository:
            self.repository.save_evaluation_run(eval_run_data)
        
        try:
            # Execute evaluation
            results = await self._execute_evaluation(test_suite, system_under_test, run_id)
            
            # Generate summary
            summary = EvaluationSummary(results)
            
            # Update run status
            updates = {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc),
                "passed_tests": summary.passed,
                "failed_tests": summary.failed,
                "average_score": summary.average_score,
                "results": [result.__dict__ for result in results]
            }
            
            if save_to_db and self.repository:
                self.repository.update_evaluation_run(run_id, updates)
            
            self.logger.info(f"Evaluation completed: {run_name}")
            
            return {
                "run_id": run_id,
                "run_name": run_name,
                "status": "completed",
                "results": results,
                "summary": summary.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            
            # Update run status to failed
            updates = {
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.now(timezone.utc)
            }
            
            if save_to_db and self.repository:
                self.repository.update_evaluation_run(run_id, updates)
            
            raise

    async def _execute_evaluation(
        self, 
        test_suite: TestSuite, 
        system_under_test: Callable,
        run_id: str
    ) -> List[EvaluationResult]:
        """Execute the actual evaluation."""
        all_results = []
        
        if self.config.parallel_execution:
            # Parallel execution
            semaphore = asyncio.Semaphore(self.config.max_workers)
            
            async def evaluate_test_case(test_case: TestCase) -> List[EvaluationResult]:
                async with semaphore:
                    return await self._evaluate_single_test_case(
                        test_case, system_under_test, run_id
                    )
            
            tasks = [evaluate_test_case(tc) for tc in test_suite.test_cases]
            results_lists = await asyncio.gather(*tasks, return_exceptions=True)
            
            for results in results_lists:
                if isinstance(results, Exception):
                    self.logger.error(f"Task failed: {results}")
                else:
                    all_results.extend(results)
        else:
            # Sequential execution
            for test_case in test_suite.test_cases:
                try:
                    results = await self._evaluate_single_test_case(
                        test_case, system_under_test, run_id
                    )
                    all_results.extend(results)
                except Exception as e:
                    self.logger.error(f"Test case {test_case.id} failed: {e}")
        
        return all_results

    async def _evaluate_single_test_case(
        self, 
        test_case: TestCase, 
        system_under_test: Callable,
        run_id: str
    ) -> List[EvaluationResult]:
        """Evaluate a single test case with all metrics."""
        
        # Get actual output from system under test
        try:
            if asyncio.iscoroutinefunction(system_under_test):
                actual_output = await asyncio.wait_for(
                    system_under_test(test_case.input_data),
                    timeout=self.config.timeout
                )
            else:
                actual_output = await asyncio.wait_for(
                    asyncio.to_thread(system_under_test, test_case.input_data),
                    timeout=self.config.timeout
                )
        except asyncio.TimeoutError:
            self.logger.error(f"Test case {test_case.id} timed out")
            return []
        except Exception as e:
            self.logger.error(f"System under test failed for {test_case.id}: {e}")
            return []
        
        results = []
        
        # Evaluate with each metric
        for metric in self.metrics:
            try:
                result = await metric.evaluate(test_case, actual_output)
                results.append(result)
                
                # Save to database if configured
                if self.repository:
                    result_data = {
                        "evaluation_run_id": run_id,
                        "test_case_id": result.test_case_id,
                        "metric_name": result.metric_name,
                        "score": result.score,
                        "passed": result.passed,
                        "threshold": result.threshold,
                        "explanation": result.explanation,
                        "metadata": result.metadata,
                        "execution_time": result.execution_time
                    }
                    self.repository.save_metric_result(result_data)
                
            except Exception as e:
                self.logger.error(f"Metric {metric.name} failed for test {test_case.id}: {e}")
        
        return results

    def generate_report(self, results: List[EvaluationResult], format: str = "json") -> str:
        """Generate evaluation report in specified format."""
        summary = EvaluationSummary(results)
        
        if format == "json":
            return json.dumps({
                "summary": summary.to_dict(),
                "detailed_results": [result.__dict__ for result in results]
            }, indent=2, default=str)
        
        elif format == "html":
            return self._generate_html_report(summary, results)
        
        else:
            raise ValueError(f"Unsupported report format: {format}")

    def _generate_html_report(self, summary: EvaluationSummary, results: List[EvaluationResult]) -> str:
        """Generate HTML report."""
        import html
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DeepEval Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007acc; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>DeepEval Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Evaluations: {summary.total_evaluations}</p>
                <p>Pass Rate: {summary.pass_rate:.2%}</p>
                <p>Average Score: {summary.average_score:.3f}</p>
            </div>
            
            <h2>Metric Details</h2>
        """
        
        for metric_name, metric_data in summary.metrics_summary.items():
            html_content += f"""
            <div class="metric">
                <h3>{metric_name}</h3>
                <p>Passed: {metric_data['passed']}/{metric_data['count']}</p>
                <p>Average Score: {metric_data['average_score']:.3f}</p>
            </div>
            """
        
        html_content += """
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Test Case</th>
                    <th>Metric</th>
                    <th>Score</th>
                    <th>Status</th>
                    <th>Explanation</th>
                </tr>
        """
        
        for result in results:
            status_class = "passed" if result.passed else "failed"
            status_text = "PASSED" if result.passed else "FAILED"
            
            # Escape and truncate explanation
            explanation = (result.explanation or 'N/A')
            if len(explanation) > 200:
                explanation = explanation[:197] + '...'
            explanation = html.escape(explanation)

            html_content += f"""
                <tr>
                    <td>{result.test_case_id}</td>
                    <td>{result.metric_name}</td>
                    <td>{result.score:.3f}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{explanation}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        return html_content

    async def evaluate_batch(
        self,
        test_suites: List[TestSuite],
        system_under_test: Callable,
        run_name_prefix: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple test suites in batch."""
        results = []
        
        for i, test_suite in enumerate(test_suites):
            run_name = f"{run_name_prefix}_batch_{i+1}" if run_name_prefix else f"batch_evaluation_{i+1}"
            try:
                result = await self.evaluate(test_suite, system_under_test, run_name)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch evaluation {i+1} failed: {e}")
                results.append({
                    "run_name": run_name,
                    "status": "failed",
                    "error": str(e)
                })
        
        return results

    def get_evaluation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get evaluation history from database."""
        if not self.repository:
            return []
        
        return self.repository.list_evaluation_runs(limit=limit)

    def get_evaluation_by_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get specific evaluation by ID."""
        if not self.repository:
            return None
        
        return self.repository.get_evaluation_run(run_id)

    def close(self):
        """Close database connections and cleanup resources."""
        if self.db_manager:
            self.db_manager.close()
            self.logger.info("Database connections closed")


class EvaluationRunner:
    """High-level runner for evaluations with additional features."""

    def __init__(self, engine: EvaluationEngine):
        self.engine = engine
        self.logger = logging.getLogger(__name__)

    async def run_with_retry(
        self,
        test_suite: TestSuite,
        system_under_test: Callable,
        max_retries: int = 3,
        run_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run evaluation with retry logic."""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return await self.engine.evaluate(test_suite, system_under_test, run_name)
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Evaluation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise last_exception

    async def run_with_monitoring(
        self,
        test_suite: TestSuite,
        system_under_test: Callable,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        run_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run evaluation with progress monitoring."""
        total_tests = len(test_suite.test_cases)
        completed_tests = 0
        
        async def monitored_sut(input_data):
            if asyncio.iscoroutinefunction(system_under_test):
                result = await system_under_test(input_data)
            else:
                result = system_under_test(input_data)
            nonlocal completed_tests
            completed_tests += 1
            if progress_callback:
                progress_callback(completed_tests, total_tests)
            return result
        
        return await self.engine.evaluate(test_suite, monitored_sut, run_name)
