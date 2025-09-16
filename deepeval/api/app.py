"""
FastAPI application for the DeepEval framework.

This module provides REST API endpoints for running evaluations,
managing test suites, and accessing evaluation results.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

# Optional FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel as PydanticBaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create dummy classes for when FastAPI is not available
    class FastAPI:
        pass
    class HTTPException:
        pass
    class BackgroundTasks:
        pass
    class Depends:
        pass
    class PydanticBaseModel:
        pass
    class Field:
        pass
    class HTMLResponse:
        pass
    class JSONResponse:
        pass
    class CORSMiddleware:
        pass
    class TrustedHostMiddleware:
        pass

from ..models import TestCase, TestSuite, EvaluationResult
from ..engine import EvaluationEngine
from ..config import EvaluationConfig, get_evaluation_config
from ..database import get_database_manager, get_repository
from ..monitoring import get_metrics_collector, get_health_checker, get_alert_manager
from ..utils import create_test_suite, safe_json_dumps


# Request/Response Models
class TestCaseRequest(PydanticBaseModel):
    """Request model for test case creation."""
    id: Optional[str] = None
    input: Dict[str, Any]
    expected: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class TestSuiteRequest(PydanticBaseModel):
    """Request model for test suite creation."""
    name: str
    description: Optional[str] = None
    test_cases: List[TestCaseRequest]
    tags: Optional[List[str]] = None


class MetricConfigRequest(PydanticBaseModel):
    """Request model for metric configuration."""
    type: str
    threshold: Optional[float] = None
    llm_provider: Optional[Dict[str, Any]] = None
    k: Optional[int] = None  # For retrieval metrics


class EvaluationRequest(PydanticBaseModel):
    """Request model for evaluation runs."""
    test_suite: TestSuiteRequest
    metrics_config: List[MetricConfigRequest]
    system_endpoint: Optional[str] = None
    run_name: Optional[str] = None
    parallel_execution: bool = True
    timeout: int = 300


class EvaluationResponse(PydanticBaseModel):
    """Response model for evaluation runs."""
    run_id: str
    status: str
    message: str
    run_name: Optional[str] = None


class EvaluationStatusResponse(PydanticBaseModel):
    """Response model for evaluation status."""
    run_id: str
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    average_score: Optional[float] = None
    error_message: Optional[str] = None


class HealthResponse(PydanticBaseModel):
    """Response model for health checks."""
    status: str
    timestamp: str
    checks: Dict[str, Any]


class MetricsResponse(PydanticBaseModel):
    """Response model for metrics."""
    metrics: Dict[str, Any]


# Global instances
_global_engine: Optional[EvaluationEngine] = None
_global_config: Optional[EvaluationConfig] = None


def get_engine() -> EvaluationEngine:
    """Get the global evaluation engine."""
    global _global_engine
    if _global_engine is None:
        config = get_evaluation_config()
        _global_engine = EvaluationEngine(config)
    return _global_engine


def get_config() -> EvaluationConfig:
    """Get the global configuration."""
    global _global_config
    if _global_config is None:
        _global_config = get_evaluation_config()
    return _global_config


# Create FastAPI app
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="DeepEval API",
        description="Enterprise-grade AI Evaluation Framework API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )
else:
    app = None


# Helper functions
def _create_metric_from_config(config: MetricConfigRequest) -> Any:
    """Create metric instance from configuration."""
    from ..metrics.builtin import (
        AccuracyMetric, RelevanceMetric, CoherenceMetric, LatencyMetric,
        FluencyMetric, FactualConsistencyMetric, BiasDetectionMetric, ToxicityMetric
    )
    from ..metrics.retrieval import (
        RecallAtKMetric, MRRMetric, NDCGAtKMetric, PrecisionAtKMetric, F1AtKMetric
    )
    from ..providers.llm import LLMProviderFactory
    
    metric_type = config.type
    threshold = config.threshold
    llm_provider_config = config.llm_provider
    
    llm_provider = None
    if llm_provider_config:
        provider_name = llm_provider_config["name"]
        api_key = llm_provider_config["api_key"]
        model = llm_provider_config.get("model")
        llm_provider = LLMProviderFactory.create_provider(provider_name, api_key, model)
    
    if metric_type == "accuracy":
        return AccuracyMetric(threshold=threshold)
    elif metric_type == "coherence":
        return CoherenceMetric(threshold=threshold)
    elif metric_type == "relevance":
        if not llm_provider:
            raise ValueError("LLM provider required for RelevanceMetric")
        return RelevanceMetric(llm_provider=llm_provider, threshold=threshold)
    elif metric_type == "fluency":
        if not llm_provider:
            raise ValueError("LLM provider required for FluencyMetric")
        return FluencyMetric(llm_provider=llm_provider, threshold=threshold)
    elif metric_type == "factual_consistency":
        if not llm_provider:
            raise ValueError("LLM provider required for FactualConsistencyMetric")
        return FactualConsistencyMetric(llm_provider=llm_provider, threshold=threshold)
    elif metric_type == "bias_detection":
        if not llm_provider:
            raise ValueError("LLM provider required for BiasDetectionMetric")
        bias_types = llm_provider_config.get("bias_types") if llm_provider_config else None
        return BiasDetectionMetric(llm_provider=llm_provider, threshold=threshold, bias_types=bias_types)
    elif metric_type == "toxicity":
        if not llm_provider:
            raise ValueError("LLM provider required for ToxicityMetric")
        return ToxicityMetric(llm_provider=llm_provider, threshold=threshold)
    elif metric_type == "latency":
        return LatencyMetric(threshold=threshold)
    # Retrieval metrics
    elif metric_type == "recall_at_k":
        k = config.k or 10
        return RecallAtKMetric(k=k, threshold=threshold)
    elif metric_type == "mrr":
        return MRRMetric(threshold=threshold)
    elif metric_type == "ndcg_at_k":
        k = config.k or 10
        return NDCGAtKMetric(k=k, threshold=threshold)
    elif metric_type == "precision_at_k":
        k = config.k or 10
        return PrecisionAtKMetric(k=k, threshold=threshold)
    elif metric_type == "f1_at_k":
        k = config.k or 10
        return F1AtKMetric(k=k, threshold=threshold)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


async def _background_evaluation(engine: EvaluationEngine, request: EvaluationRequest, run_id: str):
    """Background task for running evaluations."""
    try:
        # Clear existing metrics
        engine.clear_metrics()
        
        # Add configured metrics
        for metric_config in request.metrics_config:
            metric = _create_metric_from_config(metric_config)
            engine.add_metric(metric)
        
        # Create test suite
        test_cases = []
        for tc_data in request.test_suite.test_cases:
            test_case = TestCase(
                id=tc_data.id or str(uuid.uuid4()),
                input_data=tc_data.input,
                expected_output=tc_data.expected,
                metadata=tc_data.metadata,
                tags=tc_data.tags
            )
            test_cases.append(test_case)
        
        test_suite = TestSuite(
            name=request.test_suite.name,
            description=request.test_suite.description,
            test_cases=test_cases,
            tags=request.test_suite.tags
        )
        
        # Mock system under test for now
        def system_under_test(input_data: Dict[str, Any]) -> Dict[str, Any]:
            # TODO: Implement actual system endpoint calling
            return {"answer": "Mock response", "response_time": 0.05}
        
        # Run evaluation
        await engine.evaluate(
            test_suite, 
            system_under_test, 
            request.run_name or f"API_Run_{run_id}"
        )
        
    except Exception as e:
        # Update evaluation status to failed
        repository = get_repository()
        if repository:
            repository.update_evaluation_run(run_id, {
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.now(timezone.utc)
            })
        
        # Log error
        logging.error(f"Background evaluation failed: {e}")


# API Endpoints
if FASTAPI_AVAILABLE:
    
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Root endpoint with API information."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>DeepEval API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
                .endpoint { margin: 10px 0; padding: 10px; border-left: 4px solid #007acc; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>DeepEval API</h1>
                <p>Enterprise-grade AI Evaluation Framework</p>
                <p>Version: 1.0.0</p>
            </div>
            
            <h2>Available Endpoints</h2>
            <div class="endpoint">
                <strong>POST /evaluate</strong> - Start an evaluation run
            </div>
            <div class="endpoint">
                <strong>GET /evaluation/{run_id}</strong> - Get evaluation status
            </div>
            <div class="endpoint">
                <strong>GET /evaluation/{run_id}/report</strong> - Get evaluation report
            </div>
            <div class="endpoint">
                <strong>GET /catalog/metrics</strong> - List available metrics
            </div>
            <div class="endpoint">
                <strong>GET /health</strong> - Health check
            </div>
            <div class="endpoint">
                <strong>GET /monitor/metrics</strong> - Get monitoring metrics
            </div>
            
            <h2>Documentation</h2>
            <p><a href="/docs">Swagger UI</a> | <a href="/redoc">ReDoc</a></p>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    @app.post("/evaluate", response_model=EvaluationResponse)
    async def run_evaluation(
        request: EvaluationRequest,
        background_tasks: BackgroundTasks,
        engine: EvaluationEngine = Depends(get_engine)
    ):
        """Start an evaluation run."""
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
                message="Evaluation started in background",
                run_name=request.run_name
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/evaluation/{run_id}", response_model=EvaluationStatusResponse)
    async def get_evaluation_status(run_id: str, engine: EvaluationEngine = Depends(get_engine)):
        """Get evaluation status."""
        repository = get_repository()
        if repository:
            eval_run = repository.get_evaluation_run(run_id)
            if eval_run:
                return EvaluationStatusResponse(
                    run_id=run_id,
                    status=eval_run["status"],
                    started_at=eval_run.get("started_at"),
                    completed_at=eval_run.get("completed_at"),
                    total_tests=eval_run.get("total_tests", 0),
                    passed_tests=eval_run.get("passed_tests", 0),
                    failed_tests=eval_run.get("failed_tests", 0),
                    average_score=eval_run.get("average_score"),
                    error_message=eval_run.get("error_message")
                )
        
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    
    @app.get("/evaluation/{run_id}/report")
    async def get_evaluation_report(
        run_id: str, 
        format: str = "json",
        engine: EvaluationEngine = Depends(get_engine)
    ):
        """Get evaluation report."""
        repository = get_repository()
        if repository:
            eval_run = repository.get_evaluation_run(run_id)
            if eval_run and eval_run.get("results"):
                results = [EvaluationResult(**r) for r in eval_run["results"]]
                report = engine.generate_report(results, format=format)
                
                if format == "html":
                    return HTMLResponse(content=report)
                else:
                    return {"report": report}
        
        raise HTTPException(status_code=404, detail="Report not found")
    
    @app.get("/catalog/metrics")
    async def list_available_metrics():
        """List available metrics."""
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
                    "name": "fluency",
                    "description": "Text fluency evaluation",
                    "parameters": ["threshold", "llm_provider"]
                },
                {
                    "name": "factual_consistency",
                    "description": "Factual consistency evaluation",
                    "parameters": ["threshold", "llm_provider"]
                },
                {
                    "name": "bias_detection",
                    "description": "Bias detection in responses", 
                    "parameters": ["threshold", "llm_provider", "bias_types"]
                },
                {
                    "name": "toxicity",
                    "description": "Toxicity detection",
                    "parameters": ["threshold", "llm_provider"]
                },
                {
                    "name": "latency",
                    "description": "Response latency measurement",
                    "parameters": ["threshold"]
                },
                {
                    "name": "recall_at_k",
                    "description": "Recall@K for retrieval evaluation",
                    "parameters": ["k", "threshold"]
                },
                {
                    "name": "mrr",
                    "description": "Mean Reciprocal Rank for retrieval evaluation",
                    "parameters": ["threshold"]
                },
                {
                    "name": "ndcg_at_k",
                    "description": "Normalized Discounted Cumulative Gain@K for retrieval evaluation",
                    "parameters": ["k", "threshold"]
                },
                {
                    "name": "precision_at_k",
                    "description": "Precision@K for retrieval evaluation",
                    "parameters": ["k", "threshold"]
                },
                {
                    "name": "f1_at_k",
                    "description": "F1@K for retrieval evaluation",
                    "parameters": ["k", "threshold"]
                }
            ]
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check(engine: EvaluationEngine = Depends(get_engine)):
        """Health check endpoint."""
        health_checker = get_health_checker(engine)
        health_data = await health_checker.check_health()
        return HealthResponse(**health_data)
    
    @app.get("/monitor/metrics", response_model=MetricsResponse)
    async def get_metrics():
        """Metrics endpoint for monitoring."""
        metrics_collector = get_metrics_collector()
        return MetricsResponse(metrics=metrics_collector.get_metrics())
    
    @app.get("/monitor/system")
    async def get_system_metrics():
        """Get system metrics."""
        from ..monitoring import get_system_monitor
        system_monitor = get_system_monitor()
        return {"system_metrics": system_monitor.get_current_metrics()}
    
    @app.get("/config")
    async def get_configuration():
        """Get current configuration."""
        config = get_config()
        return {"configuration": config.to_dict()}
    
    @app.post("/test-suites")
    async def create_test_suite_endpoint(request: TestSuiteRequest):
        """Create a new test suite."""
        try:
            # Convert request to TestSuite object
            test_cases = []
            for tc_data in request.test_cases:
                test_case = TestCase(
                    id=tc_data.id or str(uuid.uuid4()),
                    input_data=tc_data.input,
                    expected_output=tc_data.expected,
                    metadata=tc_data.metadata,
                    tags=tc_data.tags
                )
                test_cases.append(test_case)
            
            test_suite = TestSuite(
                name=request.name,
                description=request.description,
                test_cases=test_cases,
                tags=request.tags
            )
            
            # Save to database if available
            repository = get_repository()
            if repository:
                suite_id = repository.save_test_suite(test_suite.to_dict())
                return {"test_suite_id": suite_id, "message": "Test suite created successfully"}
            else:
                return {"message": "Test suite created (not saved to database)"}
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/test-suites/{suite_id}")
    async def get_test_suite(suite_id: str):
        """Get a test suite by ID."""
        repository = get_repository()
        if repository:
            test_suite = repository.get_test_suite(suite_id)
            if test_suite:
                return test_suite
        
        raise HTTPException(status_code=404, detail="Test suite not found")
    
    @app.get("/evaluations")
    async def list_evaluations(limit: int = 100, offset: int = 0):
        """List evaluation runs."""
        repository = get_repository()
        if repository:
            return repository.list_evaluation_runs(limit=limit, offset=offset)
        else:
            return {"message": "Database not configured"}


# Error handlers
if FASTAPI_AVAILABLE:
    
    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        return JSONResponse(
            status_code=404,
            content={"detail": "Resource not found"}
        )
    
    @app.exception_handler(500)
    async def internal_error_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )


# Startup and shutdown events
if FASTAPI_AVAILABLE:
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize the application on startup."""
        logging.info("DeepEval API starting up...")
        
        # Initialize configuration
        config = get_config()
        
        # Initialize database if configured
        if config.database_url:
            from ..database import initialize_database
            initialize_database(config.database_url)
            logging.info("Database initialized")
        
        # Start system monitoring
        from ..monitoring import get_system_monitor
        system_monitor = get_system_monitor()
        system_monitor.start_monitoring()
        logging.info("System monitoring started")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        logging.info("DeepEval API shutting down...")
        
        # Stop system monitoring
        from ..monitoring import get_system_monitor
        system_monitor = get_system_monitor()
        system_monitor.stop_monitoring()
        
        # Close database connections
        db_manager = get_database_manager()
        if db_manager:
            db_manager.close()
        
        logging.info("DeepEval API shutdown complete")
