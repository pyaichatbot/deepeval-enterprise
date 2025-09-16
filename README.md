# DeepEval - Enterprise AI Evaluation Framework

A production-ready, scalable evaluation framework for AI systems with comprehensive metrics, reporting, and enterprise features.

## 🏗️ Architecture Overview

The framework has been refactored into a clean, modular architecture following enterprise best practices:

```
deepeval/
├── __init__.py              # Main package exports
├── models.py                # Core data models and structures
├── config.py                # Configuration management
├── engine.py                # Evaluation engine core logic
├── database.py              # Database models and ORM
├── utils.py                 # Utility functions and helpers
├── monitoring.py            # Monitoring and observability
├── metrics/
│   ├── __init__.py
│   ├── base.py              # Base metric interfaces
│   ├── builtin.py           # Built-in metrics (accuracy, relevance, etc.)
│   └── retrieval.py         # Retrieval metrics (Recall@K, MRR, nDCG@K)
├── providers/
│   ├── __init__.py
│   └── llm.py               # LLM provider implementations
└── api/
    ├── __init__.py
    └── app.py               # FastAPI REST API
```

## 🚀 Key Features

### Core Evaluation Metrics
- **Accuracy**: Exact match evaluation
- **Relevance**: LLM-based relevance scoring
- **Coherence**: Text coherence analysis
- **Fluency**: Text fluency evaluation
- **Factual Consistency**: Factual accuracy checking
- **Bias Detection**: Bias identification in responses
- **Toxicity Detection**: Harmful content detection
- **Latency**: Response time measurement

### Retrieval Metrics (NEW!)
- **Recall@K**: Proportion of relevant docs retrieved in top K
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of first relevant result
- **nDCG@K**: Normalized Discounted Cumulative Gain at K
- **Precision@K**: Proportion of top K results that are relevant
- **F1@K**: Harmonic mean of Precision@K and Recall@K

### Enterprise Features
- **Database Persistence**: SQLAlchemy-based data storage
- **REST API**: FastAPI-based web service
- **Monitoring**: System metrics and health checks
- **Configuration Management**: Environment-based config
- **Parallel Execution**: Concurrent evaluation processing
- **Caching**: Redis and in-memory caching support
- **Security**: API key validation and rate limiting

## 📦 Installation

### Core Installation
```bash
pip install -r requirements.txt
```

### With Optional Dependencies
```bash
# All features
pip install -r requirements.txt[all]

# Specific features
pip install -r requirements.txt[openai]      # OpenAI support
pip install -r requirements.txt[anthropic]   # Anthropic support
pip install -r requirements.txt[api]         # FastAPI server
pip install -r requirements.txt[monitoring]  # System monitoring
```

## 🔧 Quick Start

### Basic Usage

```python
import asyncio
from deepeval import (
    EvaluationEngine, EvaluationConfig, TestSuite, TestCase,
    AccuracyMetric, CoherenceMetric, LatencyMetric
)

async def my_ai_system(input_data):
    # Your AI system implementation
    return {"answer": "Response", "response_time": 0.1}

async def main():
    # Configure evaluation
    config = EvaluationConfig()
    engine = EvaluationEngine(config)
    
    # Add metrics
    engine.add_metric(AccuracyMetric(threshold=0.8))
    engine.add_metric(CoherenceMetric(threshold=0.6))
    engine.add_metric(LatencyMetric(threshold=1.0))
    
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
    
    # Generate report
    report = engine.generate_report(results["results"], format="html")
    with open("report.html", "w") as f:
        f.write(report)

asyncio.run(main())
```

### Retrieval Evaluation

```python
from deepeval import (
    RecallAtKMetric, MRRMetric, NDCGAtKMetric,
    RetrievalTestCase, TestSuite
)

# Add retrieval metrics
engine.add_metric(RecallAtKMetric(k=10, threshold=0.6))
engine.add_metric(MRRMetric(threshold=0.5))
engine.add_metric(NDCGAtKMetric(k=10, threshold=0.7))

# Create retrieval test case
retrieval_test_case = RetrievalTestCase(
    id="retrieval_test_1",
    query="machine learning algorithms",
    relevant_documents=[
        {"id": "doc1", "content": "Introduction to ML", "relevance": 1.0},
        {"id": "doc2", "content": "Deep learning basics", "relevance": 0.8}
    ],
    retrieved_documents=[
        {"id": "doc1", "content": "Introduction to ML"},
        {"id": "doc3", "content": "Statistics fundamentals"},
        {"id": "doc2", "content": "Deep learning basics"}
    ]
)

test_suite = TestSuite(
    name="Retrieval Test Suite",
    test_cases=[retrieval_test_case]
)

# Run evaluation
results = await engine.evaluate(test_suite, my_retrieval_system)
```

## 🌐 REST API

Start the API server:

```bash
uvicorn deepeval.api.app:app --reload
```

### Key Endpoints

- `POST /evaluate` - Start evaluation run
- `GET /evaluation/{run_id}` - Get evaluation status
- `GET /evaluation/{run_id}/report` - Get evaluation report
- `GET /catalog/metrics` - List available metrics
- `GET /health` - Health check
- `GET /monitor/metrics` - System metrics

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔧 Configuration

### Environment Variables

```bash
# Database
export DEEPEVAL_DATABASE_URL="sqlite:///deepeval.db"

# Logging
export DEEPEVAL_LOG_LEVEL="INFO"
export DEEPEVAL_DEBUG="false"

# Execution
export DEEPEVAL_MAX_WORKERS="10"
export DEEPEVAL_TIMEOUT="300"
export DEEPEVAL_PARALLEL_EXECUTION="true"

# Security
export DEEPEVAL_ENABLE_SECURITY="true"
export DEEPEVAL_MAX_CONCURRENT_EVALUATIONS="5"
```

### Configuration File

Create `config.json`:

```json
{
  "evaluation": {
    "max_workers": 10,
    "timeout": 300,
    "parallel_execution": true,
    "save_results": true,
    "debug": false
  },
  "database": {
    "url": "sqlite:///deepeval.db",
    "echo": false,
    "pool_size": 10
  },
  "logging": {
    "level": "INFO",
    "file_path": "deepeval.log"
  }
}
```

## 📊 Monitoring

### Health Checks

```python
from deepeval.monitoring import get_health_checker

health_checker = get_health_checker(engine)
health_status = await health_checker.check_health()
print(health_status)
```

### Metrics Collection

```python
from deepeval.monitoring import get_metrics_collector

metrics_collector = get_metrics_collector()
metrics = metrics_collector.get_metrics()
print(metrics)
```

## 🧪 Testing

Run the example:

```bash
python example_usage.py
```

This will:
1. Create test suites for both standard and retrieval evaluation
2. Run evaluations with multiple metrics
3. Generate HTML and JSON reports
4. Demonstrate the complete workflow

## 🔒 Security Features

- API key validation
- Rate limiting
- CORS configuration
- Input sanitization
- Error message sanitization
- Secure credential management

## 📈 Performance

- Parallel execution with configurable workers
- Connection pooling for database operations
- Caching for expensive operations
- Streaming responses for large datasets
- Resource monitoring and alerting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

- Documentation: [Link to docs]
- Issues: [GitHub Issues]
- Discussions: [GitHub Discussions]

---

**Note**: This is a refactored version of the original monolithic `deepeval_framework.py` file, now organized into a proper enterprise-grade architecture with separation of concerns, comprehensive testing, and production-ready features.
