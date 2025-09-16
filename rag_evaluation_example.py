#!/usr/bin/env python3
"""
Comprehensive RAG System Evaluation with DeepEval

This script demonstrates how to evaluate:
1. PDF Extraction Tool
2. Code Search Tool  
3. RAG Pipeline (End-to-End)
4. Document Retrieval
5. Answer Generation

Replace the mock implementations with your actual tools.
"""

import asyncio
import os
from typing import Dict, Any, List
from pathlib import Path

# Import DeepEval components
from deepeval import (
    EvaluationEngine, EvaluationConfig, TestSuite, TestCase, RetrievalTestCase,
    AccuracyMetric, CoherenceMetric, LatencyMetric,
    RecallAtKMetric, MRRMetric, NDCGAtKMetric, PrecisionAtKMetric, F1AtKMetric,
    RelevanceMetric, FluencyMetric, FactualConsistencyMetric
)
from deepeval.providers import OpenAIProvider, LLMConfig


# ============================================================================
# FASTAPI DOCKER SERVICE CONNECTORS (Replace with your actual endpoints)
# ============================================================================

import aiohttp
import asyncio
from typing import Optional

class FastAPIConnector:
    """Base connector for FastAPI services running in Docker"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to FastAPI endpoint"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._get_headers()
        
        try:
            async with self.session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"API call failed: {response.status} - {error_text}")
        except asyncio.TimeoutError:
            raise Exception(f"Request timeout after {self.timeout} seconds")
        except aiohttp.ClientConnectorError as e:
            raise Exception(f"Connection error: {str(e)}")

class PDFExtractorAPI:
    """FastAPI connector for PDF extraction service"""
    
    def __init__(self, base_url: str = "http://localhost:8001", api_key: Optional[str] = None):
        self.connector = FastAPIConnector(base_url, api_key)
    
    async def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text from PDF via FastAPI endpoint"""
        async with self.connector as conn:
            data = {"pdf_path": pdf_path}
            result = await conn._make_request("/extract", data)
            return result

class CodeSearchAPI:
    """FastAPI connector for code search service"""
    
    def __init__(self, base_url: str = "http://localhost:8002", api_key: Optional[str] = None):
        self.connector = FastAPIConnector(base_url, api_key)
    
    async def search_code(self, query: str, repo_path: Optional[str] = None) -> Dict[str, Any]:
        """Search for code via FastAPI endpoint"""
        async with self.connector as conn:
            data = {"query": query}
            if repo_path:
                data["repo_path"] = repo_path
            result = await conn._make_request("/search", data)
            return result

class DocumentRetrieverAPI:
    """FastAPI connector for document retrieval service"""
    
    def __init__(self, base_url: str = "http://localhost:8003", api_key: Optional[str] = None):
        self.connector = FastAPIConnector(base_url, api_key)
    
    async def retrieve_documents(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Retrieve documents via FastAPI endpoint"""
        async with self.connector as conn:
            data = {"query": query, "k": k}
            result = await conn._make_request("/retrieve", data)
            return result

class RAGPipelineAPI:
    """FastAPI connector for complete RAG pipeline service"""
    
    def __init__(self, base_url: str = "http://localhost:8004", api_key: Optional[str] = None):
        self.connector = FastAPIConnector(base_url, api_key)
    
    async def rag_pipeline(self, query: str) -> Dict[str, Any]:
        """Run complete RAG pipeline via FastAPI endpoint"""
        async with self.connector as conn:
            data = {"query": query}
            result = await conn._make_request("/rag", data)
            return result

# ============================================================================
# CONFIGURATION FOR YOUR DOCKER SERVICES
# ============================================================================

# Update these URLs to match your actual Docker service endpoints
SERVICE_CONFIG = {
    "pdf_extractor": {
        "base_url": "http://localhost:8001",  # Your PDF extraction service
        "api_key": None  # Add your API key if required
    },
    "code_search": {
        "base_url": "http://localhost:8002",  # Your code search service
        "api_key": None  # Add your API key if required
    },
    "document_retriever": {
        "base_url": "http://localhost:8003",  # Your document retrieval service
        "api_key": None  # Add your API key if required
    },
    "rag_pipeline": {
        "base_url": "http://localhost:8004",  # Your complete RAG pipeline service
        "api_key": None  # Add your API key if required
    }
}

# Alternative: Use Docker service names if running in Docker Compose
DOCKER_COMPOSE_CONFIG = {
    "pdf_extractor": {
        "base_url": "http://pdf-extractor:8001",  # Docker service name
        "api_key": None
    },
    "code_search": {
        "base_url": "http://code-search:8002",  # Docker service name
        "api_key": None
    },
    "document_retriever": {
        "base_url": "http://document-retriever:8003",  # Docker service name
        "api_key": None
    },
    "rag_pipeline": {
        "base_url": "http://rag-pipeline:8004",  # Docker service name
        "api_key": None
    }
}


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

async def evaluate_pdf_extraction():
    """Evaluate PDF extraction tool via FastAPI"""
    print("📄 Evaluating PDF Extraction Tool (FastAPI)...")
    
    engine = EvaluationEngine(EvaluationConfig())
    
    # Add metrics for PDF extraction
    engine.add_metric(AccuracyMetric(threshold=0.8))  # Text extraction accuracy
    engine.add_metric(LatencyMetric(threshold=5.0))   # Max 5 seconds per PDF (includes network latency)
    
    # Create test cases for PDF extraction
    test_cases = [
        TestCase(
            id="pdf_extract_1",
            input_data={"pdf_path": "sample_document.pdf"},
            expected_output={
                "text": "This is extracted text from the PDF document...",
                "metadata": {"pages": 10, "word_count": 1500}
            }
        ),
        TestCase(
            id="pdf_extract_2", 
            input_data={"pdf_path": "technical_manual.pdf"},
            expected_output={
                "text": "Technical documentation content...",
                "metadata": {"pages": 25, "word_count": 5000}
            }
        )
    ]
    
    test_suite = TestSuite(name="PDF Extraction Test", test_cases=test_cases)
    
    # FastAPI PDF extraction system
    async def pdf_extraction_system(input_data):
        pdf_path = input_data.get("pdf_path", "")
        config = SERVICE_CONFIG["pdf_extractor"]
        extractor = PDFExtractorAPI(config["base_url"], config["api_key"])
        result = await extractor.extract_text(pdf_path)
        return result
    
    try:
        results = await engine.evaluate(test_suite, pdf_extraction_system, "PDF Extraction Evaluation")
        
        print(f"✅ PDF Extraction Results:")
        print(f"   - Pass Rate: {results['summary']['pass_rate']:.2%}")
        print(f"   - Average Score: {results['summary']['average_score']:.3f}")
        
        return results
    except Exception as e:
        print(f"❌ PDF Extraction failed: {str(e)}")
        print("   Make sure your PDF extraction service is running on the configured port")
        return None

async def evaluate_code_search():
    """Evaluate code search tool via FastAPI"""
    print("\n🔍 Evaluating Code Search Tool (FastAPI)...")
    
    engine = EvaluationEngine(EvaluationConfig())
    
    # Add retrieval metrics for code search
    engine.add_metric(RecallAtKMetric(k=5, threshold=0.6))  # Find 60% of relevant code
    engine.add_metric(MRRMetric(threshold=0.5))             # Good ranking
    engine.add_metric(PrecisionAtKMetric(k=5, threshold=0.7))  # 70% precision
    engine.add_metric(LatencyMetric(threshold=3.0))         # Max 3 seconds search (includes network latency)
    
    # Create retrieval test cases for code search
    test_cases = [
        RetrievalTestCase(
            id="code_search_1",
            query="neural network training function",
            relevant_documents=[
                {"id": "neural_network.py", "content": "def train_model(data, epochs=100):", "relevance": 1.0},
                {"id": "model_utils.py", "content": "def initialize_weights():", "relevance": 0.8}
            ],
            retrieved_documents=[
                {"id": "neural_network.py", "content": "def train_model(data, epochs=100):"},
                {"id": "data_loader.py", "content": "def load_dataset():"},
                {"id": "model_utils.py", "content": "def initialize_weights():"},
                {"id": "config.py", "content": "MODEL_CONFIG = {}"},
                {"id": "trainer.py", "content": "class ModelTrainer:"}
            ]
        ),
        RetrievalTestCase(
            id="code_search_2",
            query="data preprocessing pipeline",
            relevant_documents=[
                {"id": "preprocessing.py", "content": "def preprocess_data(raw_data):", "relevance": 1.0},
                {"id": "data_utils.py", "content": "def clean_data(data):", "relevance": 0.9}
            ],
            retrieved_documents=[
                {"id": "preprocessing.py", "content": "def preprocess_data(raw_data):"},
                {"id": "data_utils.py", "content": "def clean_data(data):"},
                {"id": "validation.py", "content": "def validate_data(data):"},
                {"id": "transform.py", "content": "def transform_features(data):"},
                {"id": "loader.py", "content": "def load_raw_data():"}
            ]
        )
    ]
    
    test_suite = TestSuite(name="Code Search Test", test_cases=test_cases)
    
    # FastAPI code search system
    async def code_search_system(input_data):
        query = input_data.get("query", "")
        config = SERVICE_CONFIG["code_search"]
        search_tool = CodeSearchAPI(config["base_url"], config["api_key"])
        result = await search_tool.search_code(query)
        
        # Convert to DeepEval format
        return {
            "retrieved_documents": [
                {
                    "id": doc["file_path"],
                    "content": doc["code_snippet"],
                    "relevance": doc["relevance_score"]
                }
                for doc in result["results"]
            ]
        }
    
    try:
        results = await engine.evaluate(test_suite, code_search_system, "Code Search Evaluation")
        
        print(f"✅ Code Search Results:")
        print(f"   - Pass Rate: {results['summary']['pass_rate']:.2%}")
        print(f"   - Average Score: {results['summary']['average_score']:.3f}")
        
        return results
    except Exception as e:
        print(f"❌ Code Search failed: {str(e)}")
        print("   Make sure your code search service is running on the configured port")
        return None

async def evaluate_document_retrieval():
    """Evaluate document retrieval component via FastAPI"""
    print("\n📚 Evaluating Document Retrieval (FastAPI)...")
    
    engine = EvaluationEngine(EvaluationConfig())
    
    # Add comprehensive retrieval metrics
    engine.add_metric(RecallAtKMetric(k=5, threshold=0.7))   # 70% recall
    engine.add_metric(MRRMetric(threshold=0.6))              # Good ranking
    engine.add_metric(NDCGAtKMetric(k=5, threshold=0.8))     # High quality ranking
    engine.add_metric(PrecisionAtKMetric(k=5, threshold=0.8)) # 80% precision
    engine.add_metric(F1AtKMetric(k=5, threshold=0.75))      # Balanced F1 score
    
    # Create comprehensive retrieval test cases
    test_cases = [
        RetrievalTestCase(
            id="retrieval_1",
            query="machine learning algorithms",
            relevant_documents=[
                {"id": "doc1", "content": "Machine learning algorithms overview", "relevance": 1.0},
                {"id": "doc2", "content": "Deep learning neural networks", "relevance": 0.8},
                {"id": "doc3", "content": "Supervised learning methods", "relevance": 0.9}
            ],
            retrieved_documents=[
                {"id": "doc1", "content": "Machine learning algorithms overview"},
                {"id": "doc3", "content": "Supervised learning methods"},
                {"id": "doc2", "content": "Deep learning neural networks"},
                {"id": "doc4", "content": "Computer vision applications"},
                {"id": "doc5", "content": "Data preprocessing techniques"}
            ]
        ),
        RetrievalTestCase(
            id="retrieval_2",
            query="natural language processing",
            relevant_documents=[
                {"id": "doc3", "content": "Natural language processing techniques", "relevance": 1.0},
                {"id": "doc6", "content": "Text analysis methods", "relevance": 0.7}
            ],
            retrieved_documents=[
                {"id": "doc3", "content": "Natural language processing techniques"},
                {"id": "doc6", "content": "Text analysis methods"},
                {"id": "doc1", "content": "Machine learning algorithms overview"},
                {"id": "doc7", "content": "Speech recognition systems"},
                {"id": "doc8", "content": "Information extraction"}
            ]
        )
    ]
    
    test_suite = TestSuite(name="Document Retrieval Test", test_cases=test_cases)
    
    # FastAPI document retrieval system
    async def document_retrieval_system(input_data):
        query = input_data.get("query", "")
        config = SERVICE_CONFIG["document_retriever"]
        retriever = DocumentRetrieverAPI(config["base_url"], config["api_key"])
        result = await retriever.retrieve_documents(query, k=5)
        return result
    
    try:
        results = await engine.evaluate(test_suite, document_retrieval_system, "Document Retrieval Evaluation")
        
        print(f"✅ Document Retrieval Results:")
        print(f"   - Pass Rate: {results['summary']['pass_rate']:.2%}")
        print(f"   - Average Score: {results['summary']['average_score']:.3f}")
        
        return results
    except Exception as e:
        print(f"❌ Document Retrieval failed: {str(e)}")
        print("   Make sure your document retrieval service is running on the configured port")
        return None

async def evaluate_rag_pipeline():
    """Evaluate complete RAG pipeline via FastAPI"""
    print("\n🤖 Evaluating Complete RAG Pipeline (FastAPI)...")
    
    engine = EvaluationEngine(EvaluationConfig())
    
    # Add comprehensive metrics for RAG evaluation
    engine.add_metric(AccuracyMetric(threshold=0.7))           # Answer accuracy
    engine.add_metric(CoherenceMetric(threshold=0.8))          # Answer coherence
    engine.add_metric(LatencyMetric(threshold=10.0))           # Max 10 seconds total (includes network latency)
    engine.add_metric(RecallAtKMetric(k=3, threshold=0.6))     # Document retrieval
    engine.add_metric(MRRMetric(threshold=0.5))                # Retrieval ranking
    
    # Create test cases for complete RAG pipeline
    test_cases = [
        TestCase(
            id="rag_pipeline_1",
            input_data={"query": "What is machine learning?"},
            expected_output={
                "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
                "context_used": 3
            }
        ),
        TestCase(
            id="rag_pipeline_2",
            input_data={"query": "How do neural networks work?"},
            expected_output={
                "answer": "Neural networks are computing systems inspired by biological neural networks that can learn to perform tasks by considering examples.",
                "context_used": 2
            }
        )
    ]
    
    test_suite = TestSuite(name="RAG Pipeline Test", test_cases=test_cases)
    
    # FastAPI RAG system
    async def rag_system(input_data):
        query = input_data.get("query", "")
        config = SERVICE_CONFIG["rag_pipeline"]
        rag_api = RAGPipelineAPI(config["base_url"], config["api_key"])
        result = await rag_api.rag_pipeline(query)
        return result
    
    try:
        results = await engine.evaluate(test_suite, rag_system, "RAG Pipeline Evaluation")
        
        print(f"✅ RAG Pipeline Results:")
        print(f"   - Pass Rate: {results['summary']['pass_rate']:.2%}")
        print(f"   - Average Score: {results['summary']['average_score']:.3f}")
        
        return results
    except Exception as e:
        print(f"❌ RAG Pipeline failed: {str(e)}")
        print("   Make sure your RAG pipeline service is running on the configured port")
        return None

async def evaluate_with_llm_metrics():
    """Evaluate with LLM-based metrics (requires OpenAI API key)"""
    print("\n🧠 Evaluating with LLM-based Metrics...")
    
    # Check if OpenAI is available
    try:
        from deepeval.providers import OpenAIProvider, LLMConfig
        
        # You'll need to set your OpenAI API key
        # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
        
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  OpenAI API key not found. Skipping LLM-based evaluation.")
            return None
        
        engine = EvaluationEngine(EvaluationConfig())
        
        # Create LLM provider
        llm_config = LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo"
        )
        llm_provider = OpenAIProvider(llm_config)
        
        # Add LLM-based metrics
        engine.add_metric(RelevanceMetric(llm_provider=llm_provider, threshold=0.7))
        engine.add_metric(FluencyMetric(llm_provider=llm_provider, threshold=0.8))
        engine.add_metric(FactualConsistencyMetric(llm_provider=llm_provider, threshold=0.8))
        
        # Create test cases for LLM evaluation
        test_cases = [
            TestCase(
                id="llm_eval_1",
                input_data={
                    "query": "What is machine learning?",
                    "context": "Machine learning is a subset of AI that enables computers to learn from data."
                },
                expected_output={
                    "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
                }
            )
        ]
        
        test_suite = TestSuite(name="LLM-based Evaluation", test_cases=test_cases)
        
        # Mock RAG system with context
        def rag_with_context(input_data):
            query = input_data.get("query", "")
            context = input_data.get("context", "")
            
            # Simulate RAG response
            return {
                "answer": f"Based on the context: {context}, {query} is a field that involves...",
                "context_used": context
            }
        
        results = await engine.evaluate(test_suite, rag_with_context, "LLM-based RAG Evaluation")
        
        print(f"✅ LLM-based Evaluation Results:")
        print(f"   - Pass Rate: {results['summary']['pass_rate']:.2%}")
        print(f"   - Average Score: {results['summary']['average_score']:.3f}")
        
        return results
        
    except ImportError:
        print("⚠️  OpenAI library not available. Install with: pip install openai")
        return None

async def main():
    """Run comprehensive RAG system evaluation"""
    print("🚀 Comprehensive RAG System Evaluation")
    print("=" * 60)
    
    all_results = {}
    
    # 1. Evaluate PDF Extraction
    pdf_results = await evaluate_pdf_extraction()
    all_results["pdf_extraction"] = pdf_results
    
    # 2. Evaluate Code Search
    code_results = await evaluate_code_search()
    all_results["code_search"] = code_results
    
    # 3. Evaluate Document Retrieval
    retrieval_results = await evaluate_document_retrieval()
    all_results["document_retrieval"] = retrieval_results
    
    # 4. Evaluate Complete RAG Pipeline
    rag_results = await evaluate_rag_pipeline()
    all_results["rag_pipeline"] = rag_results
    
    # 5. Evaluate with LLM-based metrics (optional)
    llm_results = await evaluate_with_llm_metrics()
    if llm_results:
        all_results["llm_evaluation"] = llm_results
    
    # Generate comprehensive reports
    print("\n📊 Generating Comprehensive Reports...")
    
    # Individual component reports
    for component, results in all_results.items():
        if results:
            report = results["results"]
            html_report = EvaluationEngine(EvaluationConfig()).generate_report(report, format="html")
            
            filename = f"rag_evaluation_{component}.html"
            with open(filename, "w") as f:
                f.write(html_report)
            print(f"✅ {component.title()} report saved: {filename}")
    
    # Overall summary
    print("\n📋 Overall Evaluation Summary:")
    print("-" * 40)
    
    for component, results in all_results.items():
        if results:
            summary = results["summary"]
            print(f"{component.replace('_', ' ').title()}:")
            print(f"  - Pass Rate: {summary['pass_rate']:.2%}")
            print(f"  - Average Score: {summary['average_score']:.3f}")
            print(f"  - Total Tests: {summary['total_evaluations']}")
            print()
    
    print("🎉 RAG System Evaluation Complete!")
    print("\nNext Steps:")
    print("1. Update SERVICE_CONFIG with your actual Docker service URLs")
    print("2. Ensure all your FastAPI services are running in Docker")
    print("3. Add more test cases specific to your domain")
    print("4. Set up continuous evaluation in your CI/CD pipeline")
    print("5. Monitor performance over time")
    print("\nDocker Service Configuration:")
    print("- PDF Extractor: http://localhost:8001/extract")
    print("- Code Search: http://localhost:8002/search")
    print("- Document Retriever: http://localhost:8003/retrieve")
    print("- RAG Pipeline: http://localhost:8004/rag")

if __name__ == "__main__":
    asyncio.run(main())
