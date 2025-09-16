# 🔌 DeepEval Connection Patterns

This document explains how DeepEval connects to your AI systems and the various integration patterns available.

## 📋 Table of Contents

- [System Under Test (SUT) Interface](#system-under-test-sut-interface)
- [Connection Patterns](#connection-patterns)
  - [Direct Function Call](#1-direct-function-call)
  - [API Endpoint](#2-api-endpoint)
  - [Database Integration](#3-database-integration)
  - [Microservice Architecture](#4-microservice-architecture)
- [Real-World Examples](#real-world-examples)
- [Configuration Options](#configuration-options)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## 🎯 System Under Test (SUT) Interface

DeepEval connects to your AI systems through a **System Under Test (SUT)** function. This function acts as a bridge between DeepEval and your actual AI implementation.

### SUT Function Signature

```python
def your_ai_system(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    System Under Test function that DeepEval will call.
    
    Args:
        input_data: Dictionary containing test case input data
        
    Returns:
        Dictionary containing the AI system's output
    """
    # Your AI system logic here
    return {
        "output": "AI system response",
        "metadata": {"processing_time": 0.5, "model_used": "gpt-4"}
    }
```

## 🔗 Connection Patterns

### 1. Direct Function Call

**Use Case**: Simple AI functions, local models, or when you have direct access to your AI logic.

```python
import asyncio
from deepeval import EvaluationEngine, TestSuite, TestCase, AccuracyMetric

# Your AI system as a direct function
def my_chatbot(input_data):
    user_message = input_data.get("message", "")
    
    # Your chatbot logic
    response = f"Bot response to: {user_message}"
    
    return {
        "response": response,
        "confidence": 0.85,
        "processing_time": 0.2
    }

async def evaluate_direct_function():
    engine = EvaluationEngine()
    engine.add_metric(AccuracyMetric(threshold=0.8))
    
    test_cases = [
        TestCase(
            id="test_1",
            input_data={"message": "Hello, how are you?"},
            expected_output={"response": "I'm doing well, thank you!"}
        )
    ]
    
    test_suite = TestSuite(name="Chatbot Test", test_cases=test_cases)
    
    # Direct function call - DeepEval calls your function directly
    results = await engine.evaluate(test_suite, my_chatbot)
    
    return results

# Run evaluation
asyncio.run(evaluate_direct_function())
```

**Advantages**:
- Simple and fast
- No network overhead
- Easy debugging
- Direct access to internal state

**Disadvantages**:
- Limited to local systems
- No load testing capabilities
- Harder to scale

### 2. API Endpoint

**Use Case**: REST APIs, microservices, or when your AI system is deployed as a web service.

```python
import asyncio
import aiohttp
from deepeval import EvaluationEngine, TestSuite, TestCase, LatencyMetric

class APIConnector:
    """Connector for API-based AI systems"""
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def call_ai_system(self, input_data):
        """Call your AI system via API"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        async with self.session.post(
            f"{self.base_url}/chat",
            json=input_data,
            headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"API call failed: {response.status}")

async def evaluate_api_system():
    engine = EvaluationEngine()
    engine.add_metric(LatencyMetric(threshold=2.0))  # Max 2 seconds
    
    test_cases = [
        TestCase(
            id="api_test_1",
            input_data={"message": "What is machine learning?"},
            expected_output={"response": "Machine learning is..."}
        )
    ]
    
    test_suite = TestSuite(name="API Test", test_cases=test_cases)
    
    # API-based evaluation
    async with APIConnector("https://your-ai-api.com", "your-api-key") as connector:
        results = await engine.evaluate(test_suite, connector.call_ai_system)
    
    return results

# Run evaluation
asyncio.run(evaluate_api_system())
```

**Advantages**:
- Tests real production endpoints
- Load testing capabilities
- Network latency included
- Easy to scale

**Disadvantages**:
- Network dependency
- Requires API to be running
- More complex error handling

### 3. Database Integration

**Use Case**: AI systems that store/retrieve data from databases, or when you need to evaluate data processing pipelines.

```python
import asyncio
import sqlite3
from deepeval import EvaluationEngine, TestSuite, TestCase, AccuracyMetric

class DatabaseConnector:
    """Connector for database-integrated AI systems"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def call_ai_system(self, input_data):
        """Call AI system that interacts with database"""
        query = input_data.get("query", "")
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Your AI system logic with database interaction
            cursor.execute("SELECT * FROM documents WHERE content LIKE ?", (f"%{query}%",))
            results = cursor.fetchall()
            
            # Process results with AI
            ai_response = f"Found {len(results)} relevant documents for: {query}"
            
            return {
                "response": ai_response,
                "documents_found": len(results),
                "query": query
            }
            
        finally:
            conn.close()

async def evaluate_database_system():
    engine = EvaluationEngine()
    engine.add_metric(AccuracyMetric(threshold=0.7))
    
    test_cases = [
        TestCase(
            id="db_test_1",
            input_data={"query": "machine learning"},
            expected_output={"documents_found": 5}
        )
    ]
    
    test_suite = TestSuite(name="Database Test", test_cases=test_cases)
    
    # Database-based evaluation
    connector = DatabaseConnector("knowledge_base.db")
    results = await engine.evaluate(test_suite, connector.call_ai_system)
    
    return results

# Run evaluation
asyncio.run(evaluate_database_system())
```

**Advantages**:
- Tests data processing pipelines
- Real database interactions
- Data consistency validation
- Complex query testing

**Disadvantages**:
- Database setup required
- Data dependency
- Slower execution

### 4. Microservice Architecture

**Use Case**: Complex AI systems with multiple services, or when you need to test service orchestration.

```python
import asyncio
import aiohttp
from deepeval import EvaluationEngine, TestSuite, TestCase, LatencyMetric

class MicroserviceConnector:
    """Connector for microservice-based AI systems"""
    
    def __init__(self, services_config: dict):
        self.services = services_config
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def call_ai_system(self, input_data):
        """Call AI system with multiple microservices"""
        user_input = input_data.get("message", "")
        
        # Step 1: Preprocessing service
        preprocess_response = await self._call_service(
            "preprocessing", 
            {"text": user_input}
        )
        
        # Step 2: Intent classification service
        intent_response = await self._call_service(
            "intent_classification",
            {"processed_text": preprocess_response["processed_text"]}
        )
        
        # Step 3: Response generation service
        response = await self._call_service(
            "response_generation",
            {
                "intent": intent_response["intent"],
                "original_text": user_input
            }
        )
        
        return {
            "response": response["generated_response"],
            "intent": intent_response["intent"],
            "confidence": response["confidence"],
            "processing_steps": 3
        }
    
    async def _call_service(self, service_name: str, data: dict):
        """Call individual microservice"""
        service_url = self.services[service_name]["url"]
        
        async with self.session.post(
            f"{service_url}/process",
            json=data,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Service {service_name} failed: {response.status}")

async def evaluate_microservice_system():
    engine = EvaluationEngine()
    engine.add_metric(LatencyMetric(threshold=5.0))  # Max 5 seconds for complex pipeline
    
    test_cases = [
        TestCase(
            id="microservice_test_1",
            input_data={"message": "I need help with my order"},
            expected_output={"intent": "customer_support"}
        )
    ]
    
    test_suite = TestSuite(name="Microservice Test", test_cases=test_cases)
    
    # Microservice-based evaluation
    services_config = {
        "preprocessing": {"url": "http://preprocessing-service:8001"},
        "intent_classification": {"url": "http://intent-service:8002"},
        "response_generation": {"url": "http://response-service:8003"}
    }
    
    async with MicroserviceConnector(services_config) as connector:
        results = await engine.evaluate(test_suite, connector.call_ai_system)
    
    return results

# Run evaluation
asyncio.run(evaluate_microservice_system())
```

**Advantages**:
- Tests complex architectures
- Service orchestration validation
- Real-world complexity
- Scalability testing

**Disadvantages**:
- Complex setup
- Multiple dependencies
- Harder to debug
- Network latency accumulation

## 🌍 Real-World Examples

### Example 1: OpenAI GPT Integration

```python
import openai
from deepeval import EvaluationEngine, TestSuite, TestCase, CoherenceMetric

class OpenAIConnector:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def call_ai_system(self, input_data):
        user_message = input_data.get("message", "")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ]
        )
        
        return {
            "response": response.choices[0].message.content,
            "model": self.model,
            "usage": response.usage.dict()
        }

# Usage
connector = OpenAIConnector("your-openai-api-key")
results = await engine.evaluate(test_suite, connector.call_ai_system)
```

### Example 2: Custom ML Model Integration

```python
import torch
from transformers import pipeline
from deepeval import EvaluationEngine, TestSuite, TestCase, AccuracyMetric

class CustomMLConnector:
    def __init__(self, model_path: str):
        self.model = pipeline(
            "text-classification",
            model=model_path,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def call_ai_system(self, input_data):
        text = input_data.get("text", "")
        
        result = self.model(text)
        
        return {
            "prediction": result[0]["label"],
            "confidence": result[0]["score"],
            "model": "custom-classifier"
        }

# Usage
connector = CustomMLConnector("./models/my-classifier")
results = await engine.evaluate(test_suite, connector.call_ai_system)
```

### Example 3: Retrieval-Augmented Generation (RAG)

```python
from deepeval import EvaluationEngine, TestSuite, RetrievalTestCase, RecallAtKMetric

class RAGConnector:
    def __init__(self, vector_store, llm_provider):
        self.vector_store = vector_store
        self.llm_provider = llm_provider
    
    def call_ai_system(self, input_data):
        query = input_data.get("query", "")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.vector_store.search(query, k=5)
        
        # Step 2: Generate answer using retrieved context
        context = " ".join([doc["content"] for doc in retrieved_docs])
        answer = self.llm_provider.generate_answer(query, context)
        
        return {
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "context_used": len(retrieved_docs)
        }

# Usage
rag_connector = RAGConnector(vector_store, llm_provider)
results = await engine.evaluate(test_suite, rag_connector.call_ai_system)
```

## ⚙️ Configuration Options

### Timeout Configuration

```python
from deepeval import EvaluationConfig

# Configure timeouts for different connection types
config = EvaluationConfig(
    timeout_seconds=30,        # Overall evaluation timeout
    retry_attempts=3,          # Number of retry attempts
    retry_delay=1.0,          # Delay between retries
    max_concurrent_evaluations=5  # Concurrent evaluation limit
)

engine = EvaluationEngine(config)
```

### Error Handling Configuration

```python
class RobustConnector:
    def __init__(self, max_retries=3, timeout=30):
        self.max_retries = max_retries
        self.timeout = timeout
    
    async def call_ai_system(self, input_data):
        for attempt in range(self.max_retries):
            try:
                # Your AI system call
                result = await self._call_with_timeout(input_data)
                return result
                
            except asyncio.TimeoutError:
                if attempt == self.max_retries - 1:
                    return {"error": "timeout", "attempts": attempt + 1}
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {"error": str(e), "attempts": attempt + 1}
                await asyncio.sleep(1)
    
    async def _call_with_timeout(self, input_data):
        # Your actual AI system call with timeout
        return await asyncio.wait_for(
            self._actual_ai_call(input_data),
            timeout=self.timeout
        )
```

## 🚨 Error Handling

### Common Error Scenarios

1. **Network Timeouts**
```python
try:
    result = await connector.call_ai_system(input_data)
except asyncio.TimeoutError:
    return {"error": "timeout", "status": "failed"}
```

2. **API Rate Limiting**
```python
try:
    result = await connector.call_ai_system(input_data)
except aiohttp.ClientResponseError as e:
    if e.status == 429:  # Rate limited
        await asyncio.sleep(1)  # Wait and retry
        result = await connector.call_ai_system(input_data)
```

3. **Service Unavailable**
```python
try:
    result = await connector.call_ai_system(input_data)
except aiohttp.ClientConnectorError:
    return {"error": "service_unavailable", "status": "failed"}
```

## 🏆 Best Practices

### 1. Connection Design

- **Use async/await** for better performance
- **Implement proper error handling** with retries
- **Add timeout configurations** to prevent hanging
- **Use connection pooling** for API-based systems
- **Implement circuit breakers** for microservices

### 2. Testing Strategy

- **Test with realistic data** that matches production
- **Include edge cases** and error scenarios
- **Monitor performance metrics** during evaluation
- **Use different connection patterns** for different test types
- **Implement health checks** before running evaluations

### 3. Production Considerations

- **Use environment variables** for configuration
- **Implement proper logging** for debugging
- **Add monitoring and alerting** for failures
- **Use staging environments** for testing
- **Implement graceful degradation** for service failures

### 4. Security

- **Secure API keys** and credentials
- **Use HTTPS** for all external connections
- **Implement authentication** where needed
- **Validate input data** before processing
- **Log security events** for audit trails

## 📊 Monitoring and Observability

```python
import logging
from deepeval import EvaluationEngine, TestSuite, TestCase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoredConnector:
    def __init__(self, base_connector):
        self.base_connector = base_connector
    
    async def call_ai_system(self, input_data):
        start_time = asyncio.get_event_loop().time()
        
        try:
            result = await self.base_connector.call_ai_system(input_data)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"AI system call successful: {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"AI system call failed after {execution_time:.2f}s: {str(e)}")
            raise

# Usage
monitored_connector = MonitoredConnector(your_ai_connector)
results = await engine.evaluate(test_suite, monitored_connector.call_ai_system)
```

This comprehensive guide covers all the major connection patterns for integrating DeepEval with your AI systems. Choose the pattern that best fits your architecture and requirements!
