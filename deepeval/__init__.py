"""
DeepEval - Enterprise-grade AI Evaluation Framework

A production-ready, scalable evaluation framework for AI systems with comprehensive
metrics, reporting, and enterprise features.

Key Features:
- Comprehensive evaluation metrics (accuracy, relevance, coherence, etc.)
- Multiple LLM provider support (OpenAI, Anthropic, Google)
- Database persistence and reporting
- REST API for remote evaluation
- Monitoring and observability
- Retrieval evaluation metrics (Recall@K, MRR, nDCG@K)
- Parallel execution and caching
- Enterprise security features

Example Usage:
    ```python
    from deepeval import EvaluationEngine, EvaluationConfig
    from deepeval.metrics import AccuracyMetric, RelevanceMetric
    from deepeval.providers import OpenAIProvider, LLMConfig
    
    # Configure evaluation
    config = EvaluationConfig()
    engine = EvaluationEngine(config)
    
    # Add metrics
    engine.add_metric(AccuracyMetric(threshold=0.8))
    
    # Create test suite
    test_suite = TestSuite(
        name="My Test Suite",
        test_cases=[
            TestCase(
                id="test_1",
                input_data={"question": "What is the capital of France?"},
                expected_output={"answer": "Paris"}
            )
        ]
    )
    
    # Run evaluation
    results = await engine.evaluate(test_suite, my_ai_system)
    ```
"""

__version__ = "1.0.0"
__author__ = "DeepEval Team"
__email__ = "team@deepeval.ai"

# Core imports
from .models import (
    TestCase,
    TestSuite,
    EvaluationResult,
    EvaluationSummary,
    EvaluationStatus,
    MetricType,
    RetrievalTestCase,
    RetrievalMetric
)

from .engine import (
    EvaluationEngine,
    EvaluationRunner
)

from .config import (
    EvaluationConfig,
    ConfigManager,
    get_evaluation_config,
    get_config_manager
)

from .database import (
    DatabaseManager,
    DatabaseRepository,
    initialize_database,
    get_database_manager,
    get_repository
)

# Metrics imports
from .metrics.base import (
    Metric,
    RetrievalMetric as BaseRetrievalMetric,
    LLMBasedMetric,
    CompositeMetric,
    ThresholdMetric,
    StatisticalMetric,
    MetricRegistry,
    get_metric_registry,
    register_metric
)

from .metrics.builtin import (
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

from .metrics.retrieval import (
    RecallAtKMetric,
    MRRMetric,
    NDCGAtKMetric,
    PrecisionAtKMetric,
    F1AtKMetric
)

# Provider imports
from .providers.llm import (
    LLMProvider,
    LLMConfig,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    MockProvider,
    LLMProviderFactory,
    LLMProviderManager
)

# Utility imports
from .utils import (
    create_test_suite,
    load_test_suite_from_file,
    save_test_suite_to_file,
    load_evaluation_results_from_file,
    save_evaluation_results_to_file,
    Timer,
    AsyncTimer,
    RateLimiter,
    BatchProcessor,
    DataValidator,
    ResultAnalyzer,
    ProgressTracker,
    generate_unique_id,
    safe_json_dumps,
    safe_json_loads
)

# Monitoring imports
from .monitoring import (
    MetricsCollector,
    SystemMonitor,
    HealthChecker,
    AlertManager,
    PerformanceProfiler,
    MonitoringDashboard,
    get_metrics_collector,
    get_system_monitor,
    get_health_checker,
    get_alert_manager
)

# New enterprise modules
from .cache import (
    CacheManager,
    get_cache_manager,
    set_cache_manager
)

from .analytics import (
    ResultAnalyzer,
    get_result_analyzer,
    set_result_analyzer
)

from .registry import (
    CustomMetricRegistry,
    get_custom_metric_registry,
    set_custom_metric_registry,
    register_custom_metric,
    create_custom_metric,
    list_custom_metrics
)

from .security import (
    SecurityManager,
    get_security_manager,
    set_security_manager,
    validate_api_key,
    get_current_user
)

# API imports (conditional)
try:
    from .api.app import app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    app = None

# Public API - what users should import
__all__ = [
    # Core classes
    "TestCase",
    "TestSuite", 
    "EvaluationResult",
    "EvaluationSummary",
    "EvaluationStatus",
    "MetricType",
    "RetrievalTestCase",
    "RetrievalMetric",
    
    # Engine
    "EvaluationEngine",
    "EvaluationRunner",
    
    # Configuration
    "EvaluationConfig",
    "ConfigManager",
    "get_evaluation_config",
    "get_config_manager",
    
    # Database
    "DatabaseManager",
    "DatabaseRepository",
    "initialize_database",
    "get_database_manager",
    "get_repository",
    
    # Metrics
    "Metric",
    "BaseRetrievalMetric",
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
    "F1AtKMetric",
    
    # LLM Providers
    "LLMProvider",
    "LLMConfig",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "MockProvider",
    "LLMProviderFactory",
    "LLMProviderManager",
    
    # Utilities
    "create_test_suite",
    "load_test_suite_from_file",
    "save_test_suite_to_file",
    "load_evaluation_results_from_file",
    "save_evaluation_results_to_file",
    "Timer",
    "AsyncTimer",
    "RateLimiter",
    "BatchProcessor",
    "DataValidator",
    "ResultAnalyzer",
    "ProgressTracker",
    "generate_unique_id",
    "safe_json_dumps",
    "safe_json_loads",
    
    # Monitoring
    "MetricsCollector",
    "SystemMonitor",
    "HealthChecker",
    "AlertManager",
    "PerformanceProfiler",
    "MonitoringDashboard",
    "get_metrics_collector",
    "get_system_monitor",
    "get_health_checker",
    "get_alert_manager",
    
    # Enterprise Features
    "CacheManager",
    "get_cache_manager",
    "set_cache_manager",
    "ResultAnalyzer",
    "get_result_analyzer",
    "set_result_analyzer",
    "CustomMetricRegistry",
    "get_custom_metric_registry",
    "set_custom_metric_registry",
    "register_custom_metric",
    "create_custom_metric",
    "list_custom_metrics",
    "SecurityManager",
    "get_security_manager",
    "set_security_manager",
    "validate_api_key",
    "get_current_user",
    
    # Version info
    "__version__",
    "__author__",
    "__email__",
]

# Add API app if available
if API_AVAILABLE:
    __all__.append("app")

# Package metadata
__package_name__ = "deepeval"
__description__ = "Enterprise-grade AI Evaluation Framework"
__url__ = "https://github.com/deepeval/deepeval"
__license__ = "MIT"
__keywords__ = ["ai", "evaluation", "metrics", "llm", "machine-learning", "nlp"]


def get_version() -> str:
    """Get the current version of DeepEval."""
    return __version__


def get_package_info() -> dict:
    """Get package information."""
    return {
        "name": __package_name__,
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "email": __email__,
        "url": __url__,
        "license": __license__,
        "keywords": __keywords__,
        "api_available": API_AVAILABLE
    }


def check_dependencies() -> dict:
    """Check availability of optional dependencies."""
    dependencies = {
        "core": {
            "pandas": True,
            "numpy": True,
            "pydantic": True,
            "sqlalchemy": True,
        },
        "optional": {
            "openai": False,
            "anthropic": False,
            "google-generativeai": False,
            "psutil": False,
            "redis": False,
            "fastapi": False,
            "sentence-transformers": False,
        }
    }
    
    # Check optional dependencies
    try:
        import openai
        dependencies["optional"]["openai"] = True
    except ImportError:
        pass
    
    try:
        import anthropic
        dependencies["optional"]["anthropic"] = True
    except ImportError:
        pass
    
    try:
        import google.generativeai
        dependencies["optional"]["google-generativeai"] = True
    except ImportError:
        pass
    
    try:
        import psutil
        dependencies["optional"]["psutil"] = True
    except ImportError:
        pass
    
    try:
        import redis
        dependencies["optional"]["redis"] = True
    except ImportError:
        pass
    
    try:
        import fastapi
        dependencies["optional"]["fastapi"] = True
    except ImportError:
        pass
    
    try:
        import sentence_transformers
        dependencies["optional"]["sentence-transformers"] = True
    except ImportError:
        pass
    
    return dependencies


def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup logging for the DeepEval framework."""
    import logging
    
    # Create logger
    logger = logging.getLogger("deepeval")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Initialize default configuration
def initialize_defaults():
    """Initialize default configuration and logging."""
    setup_logging()
    
    # Initialize default config manager
    config_manager = get_config_manager()
    config_manager.initialize_default_config()
    
    return config_manager


# Auto-initialize on import
_initialized = False

def _auto_initialize():
    """Auto-initialize the framework."""
    global _initialized
    if not _initialized:
        try:
            initialize_defaults()
            _initialized = True
        except Exception as e:
            # Don't fail on import if initialization fails
            import logging
            logging.getLogger(__name__).warning(f"Auto-initialization failed: {e}")

# Uncomment to enable auto-initialization
# _auto_initialize()
