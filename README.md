# DeepEval - Enterprise AI Evaluation Framework

A production-ready, scalable evaluation framework for AI systems with comprehensive metrics, reporting, and enterprise features. Built for organizations with 30K+ users requiring enterprise-grade security, performance, and analytics.

## 🏗️ Architecture Overview

The framework has been completely refactored from the original monolithic design into a clean, modular architecture following enterprise best practices:

```
deepeval/
├── __init__.py              # Main package exports and public API
├── models.py                # Core data models and structures
├── config.py                # Configuration management
├── engine.py                # Evaluation engine core logic
├── database.py              # Database models and ORM
├── utils.py                 # Utility functions and helpers
├── monitoring.py            # Monitoring and observability
├── cache.py                 # Caching system (memory/Redis)
├── analytics.py             # Advanced result analysis
├── registry.py              # Custom metric registration
├── security.py              # Enterprise security features
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
- **Semantic Similarity**: Advanced semantic similarity using sentence transformers

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
- **Advanced Caching**: Redis and in-memory caching with CacheManager
- **Enterprise Security**: API key validation, rate limiting, and multi-tenant support
- **Advanced Analytics**: Comprehensive result analysis with ResultAnalyzer
- **Custom Metrics**: Dynamic metric registration with CustomMetricRegistry
- **Docker Support**: Production-ready containerization

## 📦 Installation

### Core Installation
```bash
pip install -r requirements.txt
```

### Docker Installation (Recommended for Production)
```bash
# Build the Docker image
docker build -t deepeval-enterprise .

# Run with Docker Compose
docker-compose up -d
```

### Optional Dependencies
The framework includes optional dependencies for advanced features:
- **sentence-transformers**: For SemanticSimilarityMetric
- **redis**: For distributed caching
- **All other dependencies**: Included in requirements.txt

## 🔧 Quick Start

### Basic Usage

```python
import asyncio
from deepeval import (
    EvaluationEngine, EvaluationConfig, TestSuite, TestCase,
    AccuracyMetric, CoherenceMetric, LatencyMetric, SemanticSimilarityMetric,
    CacheManager, ResultAnalyzer, SecurityManager
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
    engine.add_metric(SemanticSimilarityMetric(threshold=0.7))
    
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

### Enterprise Features Usage

```python
from deepeval import (
    CacheManager, ResultAnalyzer, SecurityManager, 
    CustomMetricRegistry, register_custom_metric
)

# Security Management
security_manager = SecurityManager()
api_key = security_manager.generate_api_key("user1", "user@company.com", ["user"])
user_info = security_manager.get_current_user(api_key)

# Advanced Caching
cache_manager = CacheManager(cache_type="memory")  # or "redis"
cache_manager.set("evaluation_results", results, ttl=3600)
cached_results = cache_manager.get("evaluation_results")

# Advanced Analytics
analyzer = ResultAnalyzer()
analysis = analyzer.analyze_results(results)
report = analyzer.generate_report(results, format="json")

# Custom Metrics
registry = CustomMetricRegistry()
registry.register("my_custom_metric", MyCustomMetric)
custom_metric = registry.create_metric("my_custom_metric", threshold=0.8)
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

### Run the Enterprise Test Suite

```bash
# Comprehensive enterprise functionality test
python test_enterprise_functionality.py
```

This will test:
1. **Security Manager**: API key generation and validation
2. **Cache Manager**: Memory and Redis caching
3. **Custom Metric Registry**: Dynamic metric registration
4. **Evaluation Engine**: Multi-metric evaluation
5. **Result Analyzer**: Advanced analytics and reporting
6. **Rate Limiting**: Enterprise security features

### Run Basic Examples

```bash
# Basic usage example
python example_usage.py
```

### Docker Testing

```bash
# Test in Docker environment
docker run --rm -v $(pwd):/app -w /app deepeval-enterprise python test_enterprise_functionality.py
```

## 🔒 Enterprise Security Features

- **API Key Management**: Secure generation, validation, and revocation
- **Rate Limiting**: Configurable request limits per user/API key
- **Multi-tenant Support**: User isolation and role-based access
- **Role-based Access Control**: Admin, user, and custom roles
- **CORS Configuration**: Secure cross-origin resource sharing
- **Input Sanitization**: Protection against injection attacks
- **Error Message Sanitization**: Secure error handling
- **Secure Credential Management**: Encrypted storage and transmission

## 📈 Enterprise Performance

- **Parallel Execution**: Configurable workers for concurrent processing
- **Advanced Caching**: Redis and in-memory caching with CacheManager
- **Connection Pooling**: Optimized database operations
- **Streaming Responses**: Efficient handling of large datasets
- **Resource Monitoring**: Real-time system metrics and alerting
- **Docker Optimization**: Production-ready containerization
- **Scalability**: Designed for 30K+ concurrent users

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

## 🏢 Enterprise Deployment

### Docker Production Setup

```bash
# Build production image
docker build -t deepeval-enterprise .

# Run with Docker Compose
docker-compose up -d

# Scale for high availability
docker-compose up -d --scale deepeval=3
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepeval-enterprise
spec:
  replicas: 3
  selector:
    matchLabels:
      app: deepeval-enterprise
  template:
    metadata:
      labels:
        app: deepeval-enterprise
    spec:
      containers:
      - name: deepeval
        image: deepeval-enterprise:latest
        ports:
        - containerPort: 8000
        env:
        - name: DEEPEVAL_DATABASE_URL
          value: "postgresql://user:pass@db:5432/deepeval"
        - name: DEEPEVAL_REDIS_URL
          value: "redis://redis:6379/0"
```

## 📊 Enterprise Analytics

The framework includes comprehensive analytics capabilities:

- **Performance Grading**: Automatic A-F grading system
- **Trend Analysis**: Performance tracking over time
- **Outlier Detection**: Identification of anomalous results
- **Correlation Analysis**: Metric relationship insights
- **Executive Reporting**: High-level business intelligence
- **Custom Dashboards**: Configurable monitoring views

---

**Note**: This is a complete enterprise refactor of the original monolithic `deepeval_framework.py` file. All missing classes have been integrated, including SemanticSimilarityMetric, CacheManager, ResultAnalyzer, CustomMetricRegistry, and SecurityManager. The framework is now production-ready for enterprise deployment with 30K+ users.
