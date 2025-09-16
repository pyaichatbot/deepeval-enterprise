
"""
Enterprise-grade Deep Evaluation Framework

A production-ready, scalable evaluation framework for AI systems with comprehensive
metrics, reporting, and enterprise features.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Protocol
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from contextlib import contextmanager
import html

# Core Dependencies

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
# from sqlalchemy.dialects.postgresql import UUID, JSONB # Commented out for SQLite portability

# Optional integrations

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# ==================== Core Models ====================

class EvaluationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MetricType(Enum):
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

@dataclass
class TestCase:
    """Individual test case for evaluation"""
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
    """Result of a single evaluation"""
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

class TestSuite(BaseModel):
    """Collection of test cases with configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    test_cases: List[TestCase] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        arbitrary_types_allowed = True

# ==================== Database Models ====================

Base = declarative_base()

class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False)
    test_suite_id = Column(String(255), nullable=False)
    configuration = Column(Text)                 # JSON serialized
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    total_tests = Column(Integer, default=0)
    passed_tests = Column(Integer, default=0)
    failed_tests = Column(Integer, default=0)
    average_score = Column(Float)
    results = Column(Text)                       # JSON serialized
    error_message = Column(Text)


class MetricResult(Base):
    __tablename__ = "metric_results"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    evaluation_run_id = Column(String(36), nullable=False)
    test_case_id = Column(String(255), nullable=False)
    metric_name = Column(String(100), nullable=False)
    score = Column(Float, nullable=False)
    passed = Column(Boolean, nullable=False)
    threshold = Column(Float)
    explanation = Column(Text)
    metric_metadata = Column(Text)               # JSON serialized
    execution_time = Column(Float)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

# ==================== Core Interfaces ====================

class Metric(ABC):
    """Abstract base class for all metrics"""

    def __init__(self, name: str, threshold: Optional[float] = None):
        self.name = name
        self.threshold = threshold
        self.metadata = {}

    @abstractmethod
    async def evaluate(self, test_case: TestCase, actual_output: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a test case and return result"""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate metric configuration"""
        pass

class LLMProvider(Protocol):
    """Protocol for LLM providers"""

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM"""
        pass

# ==================== Built-in Metrics ====================

class AccuracyMetric(Metric):
    """Exact match accuracy metric"""

    def __init__(self, threshold: float = 1.0):
        super().__init__("accuracy", threshold)

    async def evaluate(self, test_case: TestCase, actual_output: Dict[str, Any]) -> EvaluationResult:
        start_time = time.time()
        
        expected = test_case.expected_output
        if not expected:
            raise ValueError("Expected output required for accuracy metric")
        
        # Simple exact match for now - can be extended
        score = 1.0 if actual_output == expected else 0.0
        passed = score >= self.threshold if self.threshold else True
        
        execution_time = time.time() - start_time
        
        return EvaluationResult(
            test_case_id=test_case.id,
            metric_name=self.name,
            score=score,
            passed=passed,
            threshold=self.threshold,
            explanation=f"Exact match: {score == 1.0}",
            execution_time=execution_time
        )

    def validate_config(self) -> bool:
        return isinstance(self.threshold, (int, float)) and 0 <= self.threshold <= 1

class RelevanceMetric(Metric):
    """LLM-based relevance metric"""

    def __init__(self, llm_provider: LLMProvider, threshold: float = 0.7):
        super().__init__("relevance", threshold)
        self.llm_provider = llm_provider

    async def evaluate(self, test_case: TestCase, actual_output: Dict[str, Any]) -> EvaluationResult:
        start_time = time.time()
        
        prompt = self._create_relevance_prompt(test_case, actual_output)
        
        try:
            response = await self.llm_provider.generate(prompt)
            score = self._parse_score_from_response(response)
            passed = score >= self.threshold if self.threshold else True
            
            execution_time = time.time() - start_time
            
            return EvaluationResult(
                test_case_id=test_case.id,
                metric_name=self.name,
                score=score,
                passed=passed,
                threshold=self.threshold,
                explanation=f"LLM relevance assessment: {response[:200]}...",
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return EvaluationResult(
                test_case_id=test_case.id,
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=self.threshold,
                explanation=f"Error during evaluation: {str(e)}",
                execution_time=execution_time
            )

    def _create_relevance_prompt(self, test_case: TestCase, actual_output: Dict[str, Any]) -> str:
        return f"""
        Evaluate the relevance of the given response to the input query.
        
        Input: {test_case.input_data}
        Response: {actual_output}
        
        Rate the relevance on a scale of 0.0 to 1.0, where:
        - 1.0 = Perfectly relevant and addresses the input completely
        - 0.7-0.9 = Mostly relevant with minor gaps
        - 0.4-0.6 = Somewhat relevant but missing key aspects
        - 0.0-0.3 = Not relevant or completely off-topic
        
        Provide your score as a single number between 0.0 and 1.0.
        """

    def _parse_score_from_response(self, response: str) -> float:
        # Simple parsing - can be made more robust
        import re
        matches = re.findall(r'(\d\.\d+|\d+)', response)
        if matches:
            score = float(matches[0])
            return max(0.0, min(1.0, score))
        return 0.0

    def validate_config(self) -> bool:
        return (self.llm_provider is not None and 
                isinstance(self.threshold, (int, float)) and 
                0 <= self.threshold <= 1)


class CoherenceMetric(Metric):
    """Text coherence evaluation metric"""

    def __init__(self, threshold: float = 0.7):
        super().__init__("coherence", threshold)

    async def evaluate(self, test_case: TestCase, actual_output: Dict[str, Any]) -> EvaluationResult:
        start_time = time.time()
        
        # Simple coherence check based on sentence structure
        # In production, this would use more sophisticated NLP
        text = str(actual_output.get("text", ""))
        
        score = self._calculate_coherence_score(text)
        passed = score >= self.threshold if self.threshold else True
        
        execution_time = time.time() - start_time
        
        return EvaluationResult(
            test_case_id=test_case.id,
            metric_name=self.name,
            score=score,
            passed=passed,
            threshold=self.threshold,
            explanation=f"Coherence analysis based on structure and flow",
            execution_time=execution_time
        )

    def _calculate_coherence_score(self, text: str) -> float:
        if not text:
            return 0.0
        
        # Simple heuristic - can be enhanced with proper NLP
        sentences = text.split(".")
        if len(sentences) < 2:
            return 0.5
        
        # Check for repeated words, sentence length variation, etc.
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        length_variance = np.var([len(s.split()) for s in sentences])
        
        # Normalize score between 0 and 1
        coherence_score = min(1.0, max(0.0, (avg_length / 20) * (1 - length_variance / 100)))
        return coherence_score

    def validate_config(self) -> bool:
        return isinstance(self.threshold, (int, float)) and 0 <= self.threshold <= 1

class LatencyMetric(Metric):
    def __init__(self, threshold: float = 2.0):
        super().__init__("latency", threshold)

    def validate_config(self) -> bool:
        return isinstance(self.threshold, (int, float)) and self.threshold > 0

    async def evaluate(self, test_case: TestCase, actual_output: Dict[str, Any]) -> EvaluationResult:
        start = time.time()
        rt = float(actual_output.get("response_time", 0.0))
        score = max(0.0, 1.0 - (rt / max(self.threshold, 1e-6)))  # 1 fast, 0 slow
        return EvaluationResult(
            test_case_id=test_case.id,
            metric_name=self.name,
            score=score,
            passed=rt <= self.threshold,
            threshold=self.threshold,
            explanation=f"response_time={rt:.3f}s, threshold={self.threshold:.3f}s",
            execution_time=time.time() - start
        )


# ==================== LLM Providers ====================

class OpenAIProvider:
    """OpenAI LLM provider"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available")
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

class AnthropicProvider:
    """Anthropic LLM provider"""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not available")
        
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text

# ==================== Core Evaluation Engine ====================

class EvaluationConfig(BaseModel):
    """Configuration for evaluation runs"""
    max_workers: int = 10
    timeout: int = 300
    retry_attempts: int = 3
    parallel_execution: bool = True
    save_results: bool = True
    database_url: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

class EvaluationEngine:
    """Main evaluation engine"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics: List[Metric] = []
        self.db_session: Optional[Session] = None
        self.logger = self._setup_logging()
        
        if config.database_url:
            self.engine = create_engine(config.database_url)
            Base.metadata.create_all(self.engine)
            SessionLocal = sessionmaker(bind=self.engine)
            self.db_session = SessionLocal()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("deepeval")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def add_metric(self, metric: Metric):
        """Add a metric to the evaluation pipeline"""
        if not metric.validate_config():
            raise ValueError(f"Invalid configuration for metric: {metric.name}")
        
        self.metrics.append(metric)
        self.logger.info(f"Added metric: {metric.name}")

    def remove_metric(self, metric_name: str):
        """Remove a metric from the pipeline"""
        self.metrics = [m for m in self.metrics if m.name != metric_name]
        self.logger.info(f"Removed metric: {metric_name}")

    async def evaluate(self, 
                      test_suite: TestSuite, 
                      system_under_test: Callable,
                      run_name: Optional[str] = None) -> Dict[str, Any]:
        """Run evaluation on a test suite"""
        
        run_id = str(uuid.uuid4())
        run_name = run_name or f"evaluation_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting evaluation run: {run_name} ({run_id})")
        
        # Create evaluation run record
        eval_run = None
        if self.db_session:
            eval_run = EvaluationRun(
                id=run_id,
                name=run_name,
                status=EvaluationStatus.RUNNING.value,
                test_suite_id=test_suite.id,
                configuration=json.dumps(self.config.dict()), # Serialize config
                started_at=datetime.now(timezone.utc),
                total_tests=len(test_suite.test_cases)
            )
            self.db_session.add(eval_run)
            self.db_session.commit()
        
        try:
            results = await self._execute_evaluation(test_suite, system_under_test, run_id)
            
            # Update run status
            if eval_run:
                eval_run.status = EvaluationStatus.COMPLETED.value
                eval_run.completed_at = datetime.now(timezone.utc)
                eval_run.passed_tests = sum(1 for r in results if r.passed)
                eval_run.failed_tests = len(results) - eval_run.passed_tests
                eval_run.average_score = np.mean([r.score for r in results])
                eval_run.results = json.dumps([asdict(r) for r in results], default=str) # Serialize results
                self.db_session.commit()
            
            self.logger.info(f"Evaluation completed: {run_name}")
            
            return {
                "run_id": run_id,
                "run_name": run_name,
                "status": "completed",
                "results": results,
                "summary": self._generate_summary(results)
            }
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            
            if eval_run:
                eval_run.status = EvaluationStatus.FAILED.value
                eval_run.error_message = str(e)
                eval_run.completed_at = datetime.now(timezone.utc)
                self.db_session.commit()
            
            raise

    async def _execute_evaluation(self, 
                                 test_suite: TestSuite, 
                                 system_under_test: Callable,
                                 run_id: str) -> List[EvaluationResult]:
        """Execute the actual evaluation"""
        
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

    async def _evaluate_single_test_case(self, 
                                       test_case: TestCase, 
                                       system_under_test: Callable,
                                       run_id: str) -> List[EvaluationResult]:
        """Evaluate a single test case with all metrics"""
        
        # Get actual output from system under test
        try:
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
                if self.db_session:
                    metric_result = MetricResult(
                        evaluation_run_id=run_id,
                        test_case_id=result.test_case_id,
                        metric_name=result.metric_name,
                        score=result.score,
                        passed=result.passed,
                        threshold=result.threshold,
                        explanation=result.explanation,
                        metric_metadata=json.dumps(result.metadata or {}), # Serialize metadata
                        execution_time=result.execution_time
                    )
                    self.db_session.add(metric_result)
                
            except Exception as e:
                self.logger.error(f"Metric {metric.name} failed for test {test_case.id}: {e}")
        
        if self.db_session:
            self.db_session.commit()
        
        return results

    def _generate_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate evaluation summary"""
        if not results:
            return {}
        
        summary = {
            "total_evaluations": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "pass_rate": sum(1 for r in results if r.passed) / len(results),
            "average_score": np.mean([r.score for r in results]),
            "metrics": {}
        }
        
        # Per-metric summary
        for metric_name in set(r.metric_name for r in results):
            metric_results = [r for r in results if r.metric_name == metric_name]
            summary["metrics"][metric_name] = {
                "count": len(metric_results),
                "passed": sum(1 for r in metric_results if r.passed),
                "average_score": np.mean([r.score for r in metric_results]),
                "min_score": min(r.score for r in metric_results),
                "max_score": max(r.score for r in metric_results)
            }
        
        return summary

    def generate_report(self, results: List[EvaluationResult], format: str = "json") -> str:
        """Generate evaluation report"""
        summary = self._generate_summary(results)
        
        if format == "json":
            return json.dumps({
                "summary": summary,
                "detailed_results": [asdict(r) for r in results]
            }, indent=2, default=str)
        
        elif format == "html":
            return self._generate_html_report(summary, results)
        
        else:
            raise ValueError(f"Unsupported report format: {format}")

    def _generate_html_report(self, summary: Dict, results: List[EvaluationResult]) -> str:
        """Generate HTML report"""
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
                <p>Total Evaluations: {summary.get('total_evaluations', 0)}</p>
                <p>Pass Rate: {summary.get('pass_rate', 0):.2%}</p>
                <p>Average Score: {summary.get('average_score', 0):.3f}</p>
            </div>
            
            <h2>Metric Details</h2>
        """
        
        for metric_name, metric_data in summary.get("metrics", {}).items():
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
            if len(explanation) > 200: # Truncate long explanations
                explanation = explanation[:197] + '...'
            explanation = html.escape(explanation) # Escape HTML special characters

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

# ==================== Utility Functions ====================

def create_test_suite(name: str, test_cases: List[Dict[str, Any]]) -> TestSuite:
    """Helper function to create test suite from dictionary data"""
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

def my_ai_system(input_data: Dict[str, Any]) -> Dict[str, Any]:
    # Mock system - replace with your actual AI system
    question = input_data.get("question", "")
    if "France" in question:
        return {"answer": "Paris"}
    elif "quantum" in question:
        return {"answer": "Quantum computing uses quantum bits to process information."}
    else:
        return {"answer": "I don’t know"}

# Example usage block (commented out or moved to a separate example file)
# async def example_usage_block():
#     # This block is commented out as it was part of the original example usage
#     # and needs to be adapted or removed based on the final application structure.
#     pass

# ==================== REST API Server ====================

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel as PydanticBaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

if FASTAPI_AVAILABLE:
    app = FastAPI(title="DeepEval API", version="1.0.0")

class EvaluationRequest(PydanticBaseModel):
    test_suite: Dict[str, Any]
    system_endpoint: str
    metrics_config: List[Dict[str, Any]]
    run_name: Optional[str] = None

class EvaluationResponse(PydanticBaseModel):
    run_id: str
    status: str
    message: str

# Global engine instance
_global_engine = None

def get_engine():
    global _global_engine
    if _global_engine is None:
        # Placeholder for ConfigManager - assuming a default config for now
        config = EvaluationConfig()
        _global_engine = EvaluationEngine(config)
    return _global_engine

@app.post("/evaluate", response_model=EvaluationResponse)
async def run_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    engine: EvaluationEngine = Depends(get_engine)
):
    """Start an evaluation run"""
    try:
        run_id = str(uuid.uuid4())
        
        # Add background task to run evaluation
        background_tasks.add_task(
            _background_evaluation,
            engine,
            request,
            run_id
        )
        
        return EvaluationResponse(
            run_id=run_id,
            status="started",
            message="Evaluation started in background"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluation/{run_id}")
async def get_evaluation_status(run_id: str, engine: EvaluationEngine = Depends(get_engine)):
    """Get evaluation status"""
    if engine.db_session:
        eval_run = engine.db_session.query(EvaluationRun).filter_by(id=run_id).first()
        if eval_run:
            # Deserialize configuration and results
            config_data = json.loads(eval_run.configuration) if eval_run.configuration else {}
            results_data = json.loads(eval_run.results) if eval_run.results else []

            return {
                "run_id": run_id,
                "status": eval_run.status,
                "started_at": eval_run.started_at,
                "completed_at": eval_run.completed_at,
                "configuration": config_data,
                "summary": {
                    "total_tests": eval_run.total_tests,
                    "passed_tests": eval_run.passed_tests,
                    "failed_tests": eval_run.failed_tests,
                    "average_score": eval_run.average_score
                }
            }
    raise HTTPException(status_code=404, detail="Evaluation run not found")

@app.get("/evaluation/{run_id}/report")
async def get_evaluation_report(
    run_id: str, 
    format: str = "json",
    engine: EvaluationEngine = Depends(get_engine)
):
    """Get evaluation report"""
    if engine.db_session:
        eval_run = engine.db_session.query(EvaluationRun).filter_by(id=run_id).first()
        if eval_run and eval_run.results:
            results = [EvaluationResult(**r) for r in json.loads(eval_run.results)] # Deserialize results
            report = engine.generate_report(results, format=format)
            
            if format == "html":
                return HTMLResponse(content=report)
            else:
                return {"report": report}
    
    raise HTTPException(status_code=404, detail="Report not found")

@app.get("/catalog/metrics")
async def list_available_metrics():
    """List available metrics"""
    return {
        "built_in_metrics": [
            {
                "name": "accuracy",
                "description": "Exact match accuracy",
                "parameters": ["threshold"]
            },
            {
                "name": "coherence", 
                "description": "Text coherence evaluation",
                "parameters": ["threshold"]
            },
            {
                "name": "relevance",
                "description": "LLM-based relevance scoring",
                "parameters": ["threshold", "llm_provider"]
            },
            {
                "name": "hallucination",
                "description": "Hallucination detection",
                "parameters": ["threshold", "llm_provider"]
            },
            {
                "name": "bias_detection",
                "description": "Bias detection in responses", 
                "parameters": ["threshold", "llm_provider", "bias_types"]
            },
            {
                "name": "latency",
                "description": "Response latency measurement",
                "parameters": ["threshold"]
            }
        ]
    }

async def _background_evaluation(engine: EvaluationEngine, request: EvaluationRequest, run_id: str):
    """Background task for running evaluations"""
    try:
        engine.metrics = []  # reset per run
        for metric_config in request.metrics_config:
            metric = _create_metric_from_config(metric_config)
            engine.add_metric(metric)

        test_suite = TestSuite(**request.test_suite)

        def system_under_test(input_data: Dict[str, Any]) -> Dict[str, Any]:
            # TODO: call request.system_endpoint here
            return {"answer": "Mock response", "response_time": 0.05}

        await engine.evaluate(test_suite, system_under_test, request.run_name or f"API_Run_{run_id}")
    except Exception as e:
        if engine.db_session:
            eval_run = engine.db_session.query(EvaluationRun).filter_by(id=run_id).first()
            if eval_run:
                eval_run.status = EvaluationStatus.FAILED.value
                eval_run.error_message = str(e)
                engine.db_session.commit()

def _create_metric_from_config(config: Dict[str, Any]) -> Metric:
    """Create metric instance from configuration"""
    metric_type = config["type"]
    threshold = config.get("threshold")
    llm_provider_config = config.get("llm_provider")

    llm_provider = None
    if llm_provider_config:
        provider_name = llm_provider_config["name"]
        api_key = llm_provider_config["api_key"]
        model = llm_provider_config.get("model")
        if provider_name == "openai":
            llm_provider = OpenAIProvider(api_key=api_key, model=model)
        elif provider_name == "anthropic":
            llm_provider = AnthropicProvider(api_key=api_key, model=model)
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

    if metric_type == "accuracy":
        return AccuracyMetric(threshold=threshold)
    elif metric_type == "coherence":
        return CoherenceMetric(threshold=threshold)
    elif metric_type == "relevance":
        if not llm_provider:
            raise ValueError("LLM provider required for RelevanceMetric")
        return RelevanceMetric(llm_provider=llm_provider, threshold=threshold)
    elif metric_type == "latency":
        return LatencyMetric(threshold=threshold)
    # Add other metrics as needed
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

# ==================== Monitoring and Observability ====================

class MetricsCollector:
    """Collect and expose metrics for monitoring"""

    def __init__(self):
        self.metrics = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "average_execution_time": 0.0,
            "active_evaluations": 0
        }
        self.lock = threading.Lock()

    def increment_evaluations(self):
        with self.lock:
            self.metrics["total_evaluations"] += 1

    def increment_successful(self):
        with self.lock:
            self.metrics["successful_evaluations"] += 1

    def increment_failed(self):
        with self.lock:
            self.metrics["failed_evaluations"] += 1

    def update_execution_time(self, time: float):
        with self.lock:
            current_avg = self.metrics["average_execution_time"]
            total = self.metrics["total_evaluations"]
            # Running average
            self.metrics["average_execution_time"] = ((current_avg * (total - 1)) + time) / total

    def set_active_evaluations(self, count: int):
        with self.lock:
            self.metrics["active_evaluations"] = count

    def get_metrics(self) -> Dict[str, Any]:
        with self.lock:
            return self.metrics.copy()

class HealthChecker:
    """Health check for the evaluation system"""

    def __init__(self, engine: EvaluationEngine):
        self.engine = engine

    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {}
        }
        
        # Database connectivity
        try:
            if self.engine.db_session:
                self.engine.db_session.execute("SELECT 1")
                health_status["checks"]["database"] = "healthy"
            else:
                health_status["checks"]["database"] = "not_configured"
        except Exception as e:
            health_status["checks"]["database"] = f"unhealthy: {str(e)}"
            health_status["status"] = "unhealthy"
        
        # Memory usage
        try:
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                health_status["checks"]["memory"] = {
                    "usage_percent": memory.percent,
                    "available_gb": round(memory.available / (1024**3), 2)
                }
                
                # Disk space
                disk = psutil.disk_usage("/")
                health_status["checks"]["disk"] = {
                    "usage_percent": round((disk.used / disk.total) * 100, 2),
                    "free_gb": round(disk.free / (1024**3), 2)
                }
            else:
                health_status["checks"]["system_resources"] = "psutil_not_available"
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["checks"]["system_resources"] = f"error: {str(e)}"
            health_status["status"] = "degraded"

        # Redis cache status
        try:
            if REDIS_AVAILABLE and hasattr(self.engine, 'cache_manager') and self.engine.cache_manager.cache_type == 'redis':
                self.engine.cache_manager.redis_client.ping()
                health_status["checks"]["redis_cache"] = "healthy"
            elif not REDIS_AVAILABLE and hasattr(self.engine, 'cache_manager') and self.engine.cache_manager.cache_type == 'redis':
                health_status["checks"]["redis_cache"] = "redis_not_available"
                health_status["status"] = "degraded"
            else:
                health_status["checks"]["redis_cache"] = "not_configured_or_memory_cache"
        except Exception as e:
            health_status["checks"]["redis_cache"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"

        return health_status

if FASTAPI_AVAILABLE:
    metrics_collector = MetricsCollector()

    @app.get("/health")
    async def health_check(engine: EvaluationEngine = Depends(get_engine)):
        """Health check endpoint"""
        health_checker = HealthChecker(engine)
        return await health_checker.check_health()

    @app.get("/monitor/metrics")
    async def get_metrics():
        """Metrics endpoint for monitoring"""
        return metrics_collector.get_metrics()

# ==================== Custom Metric Framework ====================

class CustomMetricRegistry:
    """Registry for custom metrics"""

    def __init__(self):
        self.metrics = {}

    def register(self, name: str, metric_class: type):
        """Register a custom metric"""
        if not issubclass(metric_class, Metric):
            raise ValueError("Metric must inherit from Metric base class")
        
        self.metrics[name] = metric_class

    def create_metric(self, name: str, **kwargs) -> Metric:
        """Create instance of registered metric"""
        if name not in self.metrics:
            raise ValueError(f"Metric {name} not registered")
        
        return self.metrics[name](**kwargs)

    def list_metrics(self) -> List[str]:
        """List all registered metrics"""
        return list(self.metrics.keys())

# Example custom metric

class SemanticSimilarityMetric(Metric):
    """Custom metric for semantic similarity using embeddings"""

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", threshold: float = 0.7):
        super().__init__("semantic_similarity", threshold)
        self.embedding_model = embedding_model
        
        # Try to import sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embedding_model)
            self.available = True
        except ImportError:
            self.available = False

    async def evaluate(self, test_case: TestCase, actual_output: Dict[str, Any]) -> EvaluationResult:
        start_time = time.time()
        
        if not self.available:
            return EvaluationResult(
                test_case_id=test_case.id,
                metric_name=self.name,
                score=0.0,
                passed=False,
                explanation="sentence-transformers library not available"
            )
        
        expected_text = str(test_case.expected_output.get('answer', ''))
        actual_text = str(actual_output.get('answer', ''))
        
        # Calculate embeddings and similarity
        embeddings = self.model.encode([expected_text, actual_text])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        score = float(similarity)
        passed = score >= self.threshold if self.threshold else True
        execution_time = time.time() - start_time
        
        return EvaluationResult(
            test_case_id=test_case.id,
            metric_name=self.name,
            score=score,
            passed=passed,
            threshold=self.threshold,
            explanation=f"Semantic similarity: {score:.3f}",
            execution_time=execution_time
        )

    def validate_config(self) -> bool:
        return self.available and isinstance(self.threshold, (int, float))

# ==================== Performance Optimization ====================

class CacheManager:
    """Manage caching for expensive operations"""

    def __init__(self, cache_type: str = "memory"):
        self.cache_type = cache_type
        if cache_type == "memory":
            self.cache = {}
        elif cache_type == "redis":
            if not REDIS_AVAILABLE:
                raise ImportError("Redis library not available")
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            except Exception as e:
                raise ConnectionError(f"Could not connect to Redis: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if self.cache_type == "memory":
            return self.cache.get(key)
        elif self.cache_type == "redis":
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set cached value"""
        if self.cache_type == "memory":
            self.cache[key] = value
        elif self.cache_type == "redis":
            self.redis_client.setex(key, ttl, json.dumps(value, default=str))

    def clear(self):
        """Clear cache"""
        if self.cache_type == "memory":
            self.cache.clear()
        elif self.cache_type == "redis":
            self.redis_client.flushdb()

# ==================== Data Analysis and Visualization ====================

class ResultAnalyzer:
    """Analyze evaluation results and generate insights"""

    def __init__(self):
        self.cache_manager = CacheManager()

    def analyze_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Comprehensive analysis of evaluation results"""
        
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
        """Calculate overall statistics"""
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
        """Analyze results by metric type"""
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
        """Analyze performance trends over test cases"""
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
        """Detect outliers in scores and execution times"""
        outliers = {}
        
        # Score outliers (e.g., below 1st percentile)
        score_lower_bound = df['score'].quantile(0.01)
        outliers['low_score_test_cases'] = df[df['score'] < score_lower_bound][['test_case_id', 'metric_name', 'score']].to_dict(orient='records')
        
        # Latency outliers (e.g., above 99th percentile)
        latency_upper_bound = df['execution_time'].quantile(0.99)
        outliers['high_latency_test_cases'] = df[df['execution_time'] > latency_upper_bound][['test_case_id', 'metric_name', 'execution_time']].to_dict(orient='records')
        
        return outliers

    def _correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between metrics or other factors"""
        correlations = {}
        
        # Example: Correlation between score and execution time
        if 'score' in df.columns and 'execution_time' in df.columns:
            correlations['score_vs_execution_time'] = df['score'].corr(df['execution_time'])
        
        # Example: Correlation matrix for numerical metrics
        numerical_metrics = df.select_dtypes(include=[np.number])
        if not numerical_metrics.empty:
            correlations['metric_correlation_matrix'] = numerical_metrics.corr().to_dict()
            
        return correlations

# ==================== Configuration Management ====================

class ConfigManager:
    """Manages configuration loading and validation"""

    def __init__(self, config_path: Union[str, Path] = "config.json"):
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return json.load(f)
        return {}

    def get_evaluation_config(self) -> EvaluationConfig:
        return EvaluationConfig(**self._config.get("evaluation", {}))

    def save_config(self, config_data: Dict[str, Any]):
        with open(self.config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        self._config = config_data

    def update_config(self, key: str, value: Any):
        self._config[key] = value
        self.save_config(self._config)

# ==================== Security Management ====================

class SecurityManager:
    """Manages API key validation and user authentication (demo-grade)"""

    def validate_api_key(self, api_key: str) -> bool:
        # In a real application, this would involve a database lookup or JWT validation
        return api_key == "YOUR_SECURE_API_KEY"

    def get_current_user(self, api_key: str) -> Optional[str]:
        if self.validate_api_key(api_key):
            return "demo_user"
        return None

# ==================== Main Application Entry Point (Example) ====================

async def main():
    print("Starting comprehensive DeepEval example...")

    # 1. Configuration
    config_manager = ConfigManager()
    config = config_manager.get_evaluation_config()

    # 2. Initialize engine with monitoring
    engine = EvaluationEngine(config)
    metrics_collector = MetricsCollector() # Initialize a metrics collector

    # 3. Add metrics
    # Example: Add an accuracy metric
    accuracy_metric = AccuracyMetric(threshold=0.8)
    engine.add_metric(accuracy_metric)

    # Example: Add a relevance metric (requires LLM provider)
    if OPENAI_AVAILABLE:
        openai_provider = OpenAIProvider(api_key="YOUR_OPENAI_API_KEY")
        relevance_metric = RelevanceMetric(llm_provider=openai_provider, threshold=0.7)
        engine.add_metric(relevance_metric)
    else:
        print("OpenAI not available, skipping RelevanceMetric.")

    # Example: Add a coherence metric
    coherence_metric = CoherenceMetric(threshold=0.6)
    engine.add_metric(coherence_metric)

    # Example: Add a latency metric
    latency_metric = LatencyMetric(threshold=0.5) # Example: 0.5 seconds threshold
    engine.add_metric(latency_metric)

    # 4. Create a test suite
    test_cases_data = [
        {"input": {"question": "What is the capital of France?"}, "expected": {"answer": "Paris"}, "metadata": {"category": "Geography"}},
        {"input": {"question": "Explain quantum computing."}, "expected": {"answer": "Quantum computing uses quantum bits to process information."}, "metadata": {"category": "Science"}},
        {"input": {"question": "Tell me a joke."}, "expected": {"answer": "Why don't scientists trust atoms? Because they make up everything!"}, "metadata": {"category": "Humor"}},
        {"input": {"question": "What is 2+2?"}, "expected": {"answer": "4"}, "metadata": {"category": "Math"}, "tags": ["simple"]},
        {"input": {"question": "Who was the first person on the moon?"}, "expected": {"answer": "Neil Armstrong"}, "metadata": {"category": "History"}, "response_time": 0.4},
        {"input": {"question": "What is the fastest animal?"}, "expected": {"answer": "Cheetah"}, "metadata": {"category": "Biology"}, "response_time": 0.6},
        {"input": {"question": "What is the capital of Germany?"}, "expected": {"answer": "Berlin"}, "metadata": {"category": "Geography"}, "response_time": 0.3},
        {"input": {"question": "How does a black hole work?"}, "expected": {"answer": "A black hole is a region of spacetime where gravity is so strong that nothing—no particles or even electromagnetic radiation such as light—can escape from it."}, "metadata": {"category": "Science"}, "response_time": 0.8},
    ]
    test_suite = create_test_suite("My AI System Test Suite", test_cases_data)

    # 5. Define the system under test (SUT)
    # This function simulates your AI system's response
    # In a real scenario, this would call your actual AI model/service
    async def sut_wrapper(input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Simulate API call latency
        response_time = input_data.get("response_time", 0.1) # Default latency
        await asyncio.sleep(response_time)
        
        # Call your actual AI system
        actual_output = my_ai_system(input_data)
        actual_output["response_time"] = response_time # Add response time to output for LatencyMetric
        return actual_output

    # 6. Run evaluation
    print("Running evaluation...")
    evaluation_results = await engine.evaluate(test_suite, sut_wrapper, "Initial System Evaluation")
    print("Evaluation complete.")

    # 7. Generate and save report
    html_report = engine.generate_report(evaluation_results["results"], format="html")
    with open("deepeval_report.html", "w") as f:
        f.write(html_report)
    print("HTML report generated: deepeval_report.html")

    json_report = engine.generate_report(evaluation_results["results"], format="json")
    with open("deepeval_report.json", "w") as f:
        f.write(json_report)
    print("JSON report generated: deepeval_report.json")

    # 8. Analyze results
    analyzer = ResultAnalyzer()
    analysis_summary = analyzer.analyze_results(evaluation_results["results"])
    print("\nAnalysis Summary:")
    print(json.dumps(analysis_summary, indent=2, default=str))

    # 9. Example of API usage (if FastAPI is available)
    if FASTAPI_AVAILABLE:
        print("\nFastAPI endpoints are available. You can interact with them at:")
        print("  - /docs (Swagger UI)")
        print("  - /redoc (ReDoc)")
        print("  - /evaluate (POST to start an evaluation)")
        print("  - /evaluation/{run_id} (GET evaluation status)")
        print("  - /evaluation/{run_id}/report (GET evaluation report)")
        print("  - /catalog/metrics (GET available metrics)")
        print("  - /monitor/metrics (GET monitoring metrics)")
        print("  - /health (GET health check)")
        print("Run 'uvicorn deepeval_framework:app --reload' to start the API server.")

if __name__ == "__main__":
    asyncio.run(main())


