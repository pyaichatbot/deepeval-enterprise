# 🔍 DeepEval for RAG, Code Search & PDF Extraction Tools

This comprehensive guide shows you how to evaluate RAG-based systems, code search tools, and PDF extraction tools using DeepEval.

## 📋 Table of Contents

- [Overview](#overview)
- [RAG System Evaluation](#rag-system-evaluation)
- [PDF Extraction Tool Evaluation](#pdf-extraction-tool-evaluation)
- [Code Search Tool Evaluation](#code-search-tool-evaluation)
- [Complete RAG Pipeline Evaluation](#complete-rag-pipeline-evaluation)
- [Advanced Evaluation Patterns](#advanced-evaluation-patterns)
- [Real-World Examples](#real-world-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

DeepEval provides specialized metrics and evaluation patterns for:

- **RAG Systems**: End-to-end retrieval-augmented generation evaluation
- **PDF Extraction**: Document processing and text extraction quality
- **Code Search**: Semantic code search and retrieval accuracy
- **Document Retrieval**: Vector search and ranking evaluation
- **Answer Generation**: LLM response quality and accuracy

## 🤖 RAG System Evaluation

### Architecture Overview

```
RAG System Components:
├── Document Ingestion (PDF Extraction)
├── Vector Store (Embeddings)
├── Retrieval Engine (Similarity Search)
├── Context Processing
└── Answer Generation (LLM)
```

### Complete RAG Evaluation Setup

```python
import asyncio
import os
from typing import Dict, Any, List
from pathlib import Path

# Import DeepEval components
from deepeval import (
    EvaluationEngine, EvaluationConfig, TestSuite, TestCase, RetrievalTestCase,
    AccuracyMetric, CoherenceMetric, LatencyMetric, RelevanceMetric,
    RecallAtKMetric, MRRMetric, NDCGAtKMetric, PrecisionAtKMetric, F1AtKMetric,
    FluencyMetric, FactualConsistencyMetric
)
from deepeval.providers import OpenAIProvider, LLMConfig

# ============================================================================
# RAG SYSTEM COMPONENTS (Replace with your actual implementations)
# ============================================================================

class PDFExtractor:
    """Your PDF extraction tool"""
    
    def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text from PDF"""
        # Your actual PDF extraction logic here
        return {
            "text": "Extracted text content from PDF...",
            "metadata": {
                "pages": 10,
                "word_count": 1500,
                "extraction_time": 0.5,
                "sections": [
                    {"title": "Introduction", "content": "..."},
                    {"title": "Methods", "content": "..."}
                ]
            }
        }

class VectorStore:
    """Your vector store implementation"""
    
    def __init__(self):
        self.documents = [
            {"id": "doc1", "content": "Machine learning algorithms overview", "embedding": [0.1, 0.2, ...]},
            {"id": "doc2", "content": "Deep learning neural networks", "embedding": [0.3, 0.4, ...]},
            {"id": "doc3", "content": "Natural language processing techniques", "embedding": [0.5, 0.6, ...]}
        ]
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        # Your actual vector search logic here
        query_embedding = self._get_query_embedding(query)
        
        # Simulate similarity search
        scored_docs = []
        for doc in self.documents:
            similarity = self._cosine_similarity(query_embedding, doc["embedding"])
            scored_docs.append({
                "id": doc["id"],
                "content": doc["content"],
                "similarity_score": similarity
            })
        
        # Sort by similarity and return top k
        scored_docs.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_docs[:k]
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query"""
        # Your actual embedding logic here
        return [0.1, 0.2, 0.3, 0.4, 0.5]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity"""
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class RAGSystem:
    """Complete RAG system"""
    
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.vector_store = VectorStore()
        # Add your LLM provider here
        # self.llm_provider = OpenAIProvider(LLMConfig(api_key="your-key", model="gpt-4"))
    
    def rag_pipeline(self, query: str) -> Dict[str, Any]:
        """Complete RAG pipeline"""
        start_time = asyncio.get_event_loop().time()
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.vector_store.search(query, k=3)
        
        # Step 2: Generate answer using retrieved context
        context = " ".join([doc["content"] for doc in retrieved_docs])
        answer = self._generate_answer(query, context)
        
        total_time = asyncio.get_event_loop().time() - start_time
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "context_used": len(retrieved_docs),
            "total_time": total_time
        }
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using context"""
        # Your actual LLM call here
        # For demo purposes, return a mock answer
        if "machine learning" in query.lower():
            return "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
        elif "neural network" in query.lower():
            return "Neural networks are computing systems inspired by biological neural networks that can learn to perform tasks by considering examples."
        else:
            return f"Based on the context: {context[:100]}..., I can provide information about your query."

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

async def evaluate_pdf_extraction():
    """Evaluate PDF extraction tool"""
    print("📄 Evaluating PDF Extraction Tool...")
    
    engine = EvaluationEngine(EvaluationConfig())
    
    # Add metrics for PDF extraction
    engine.add_metric(AccuracyMetric(threshold=0.8))  # Text extraction accuracy
    engine.add_metric(LatencyMetric(threshold=2.0))   # Max 2 seconds per PDF
    engine.add_metric(CoherenceMetric(threshold=0.7)) # Extracted text coherence
    
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
    
    # Your PDF extraction system
    def pdf_extraction_system(input_data):
        pdf_path = input_data.get("pdf_path", "")
        extractor = PDFExtractor()
        result = extractor.extract_text(pdf_path)
        return result
    
    results = await engine.evaluate(test_suite, pdf_extraction_system, "PDF Extraction Evaluation")
    
    print(f"✅ PDF Extraction Results:")
    print(f"   - Pass Rate: {results['summary']['pass_rate']:.2%}")
    print(f"   - Average Score: {results['summary']['average_score']:.3f}")
    
    return results

async def evaluate_document_retrieval():
    """Evaluate document retrieval component"""
    print("\n📚 Evaluating Document Retrieval...")
    
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
    
    # Your document retrieval system
    def document_retrieval_system(input_data):
        query = input_data.get("query", "")
        vector_store = VectorStore()
        retrieved_docs = vector_store.search(query, k=5)
        return {"retrieved_documents": retrieved_docs}
    
    results = await engine.evaluate(test_suite, document_retrieval_system, "Document Retrieval Evaluation")
    
    print(f"✅ Document Retrieval Results:")
    print(f"   - Pass Rate: {results['summary']['pass_rate']:.2%}")
    print(f"   - Average Score: {results['summary']['average_score']:.3f}")
    
    return results

async def evaluate_rag_pipeline():
    """Evaluate complete RAG pipeline"""
    print("\n🤖 Evaluating Complete RAG Pipeline...")
    
    engine = EvaluationEngine(EvaluationConfig())
    
    # Add comprehensive metrics for RAG evaluation
    engine.add_metric(AccuracyMetric(threshold=0.7))           # Answer accuracy
    engine.add_metric(CoherenceMetric(threshold=0.8))          # Answer coherence
    engine.add_metric(LatencyMetric(threshold=3.0))            # Max 3 seconds total
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
    
    # Your RAG system
    def rag_system(input_data):
        query = input_data.get("query", "")
        rag = RAGSystem()
        result = rag.rag_pipeline(query)
        return result
    
    results = await engine.evaluate(test_suite, rag_system, "RAG Pipeline Evaluation")
    
    print(f"✅ RAG Pipeline Results:")
    print(f"   - Pass Rate: {results['summary']['pass_rate']:.2%}")
    print(f"   - Average Score: {results['summary']['average_score']:.3f}")
    
    return results

async def evaluate_with_llm_metrics():
    """Evaluate with LLM-based metrics (requires OpenAI API key)"""
    print("\n🧠 Evaluating with LLM-based Metrics...")
    
    # Check if OpenAI is available
    try:
        from deepeval.providers import OpenAIProvider, LLMConfig
        
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
        
        # Your RAG system with context
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
    
    # 2. Evaluate Document Retrieval
    retrieval_results = await evaluate_document_retrieval()
    all_results["document_retrieval"] = retrieval_results
    
    # 3. Evaluate Complete RAG Pipeline
    rag_results = await evaluate_rag_pipeline()
    all_results["rag_pipeline"] = rag_results
    
    # 4. Evaluate with LLM-based metrics (optional)
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
    print("1. Replace mock implementations with your actual tools")
    print("2. Add more test cases specific to your domain")
    print("3. Set up continuous evaluation in your CI/CD pipeline")
    print("4. Monitor performance over time")

if __name__ == "__main__":
    asyncio.run(main())
```

## 📄 PDF Extraction Tool Evaluation

### Specialized PDF Metrics

```python
from deepeval import (
    AccuracyMetric, LatencyMetric, CoherenceMetric,
    TestCase, TestSuite, EvaluationEngine
)

class PDFExtractionEvaluator:
    """Specialized evaluator for PDF extraction tools"""
    
    def __init__(self):
        self.engine = EvaluationEngine()
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup metrics specific to PDF extraction"""
        self.engine.add_metric(AccuracyMetric(threshold=0.85))    # Text accuracy
        self.engine.add_metric(LatencyMetric(threshold=3.0))      # Processing time
        self.engine.add_metric(CoherenceMetric(threshold=0.8))    # Text coherence
    
    async def evaluate_pdf_extraction(self, pdf_extractor, test_cases):
        """Evaluate PDF extraction tool"""
        
        def pdf_system(input_data):
            pdf_path = input_data.get("pdf_path", "")
            return pdf_extractor.extract_text(pdf_path)
        
        test_suite = TestSuite(name="PDF Extraction", test_cases=test_cases)
        results = await self.engine.evaluate(test_suite, pdf_system)
        
        return results

# Usage example
async def evaluate_my_pdf_tool():
    evaluator = PDFExtractionEvaluator()
    
    # Your PDF extraction tool
    class MyPDFExtractor:
        def extract_text(self, pdf_path):
            # Your actual PDF extraction logic
            return {
                "text": "Extracted text...",
                "metadata": {"pages": 10, "word_count": 1500}
            }
    
    # Test cases
    test_cases = [
        TestCase(
            id="pdf_1",
            input_data={"pdf_path": "document1.pdf"},
            expected_output={"word_count": 1500}
        ),
        TestCase(
            id="pdf_2",
            input_data={"pdf_path": "document2.pdf"},
            expected_output={"word_count": 3000}
        )
    ]
    
    results = await evaluator.evaluate_pdf_extraction(MyPDFExtractor(), test_cases)
    return results
```

### PDF-Specific Test Cases

```python
# Comprehensive PDF test cases
pdf_test_cases = [
    # Text extraction accuracy
    TestCase(
        id="text_accuracy_1",
        input_data={"pdf_path": "technical_document.pdf"},
        expected_output={
            "text": "Expected extracted text content...",
            "metadata": {"pages": 15, "word_count": 2500}
        }
    ),
    
    # Multi-page document
    TestCase(
        id="multipage_1",
        input_data={"pdf_path": "research_paper.pdf"},
        expected_output={
            "metadata": {"pages": 25, "sections": 5}
        }
    ),
    
    # Complex formatting
    TestCase(
        id="complex_format_1",
        input_data={"pdf_path": "formatted_document.pdf"},
        expected_output={
            "text": "Text with tables, images, and complex formatting...",
            "metadata": {"has_tables": True, "has_images": True}
        }
    ),
    
    # OCR quality (for scanned PDFs)
    TestCase(
        id="ocr_quality_1",
        input_data={"pdf_path": "scanned_document.pdf"},
        expected_output={
            "text": "OCR extracted text...",
            "metadata": {"ocr_confidence": 0.95}
        }
    )
]
```

## 🔍 Code Search Tool Evaluation

### Code Search Metrics

```python
from deepeval import (
    RecallAtKMetric, MRRMetric, NDCGAtKMetric,
    PrecisionAtKMetric, F1AtKMetric, RetrievalTestCase
)

class CodeSearchEvaluator:
    """Specialized evaluator for code search tools"""
    
    def __init__(self):
        self.engine = EvaluationEngine()
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup metrics specific to code search"""
        self.engine.add_metric(RecallAtKMetric(k=10, threshold=0.6))    # Find 60% of relevant code
        self.engine.add_metric(MRRMetric(threshold=0.5))                # Good ranking
        self.engine.add_metric(NDCGAtKMetric(k=10, threshold=0.7))      # High quality ranking
        self.engine.add_metric(PrecisionAtKMetric(k=10, threshold=0.7)) # 70% precision
        self.engine.add_metric(F1AtKMetric(k=10, threshold=0.65))       # Balanced F1
    
    async def evaluate_code_search(self, code_search_tool, test_cases):
        """Evaluate code search tool"""
        
        def code_search_system(input_data):
            query = input_data.get("query", "")
            results = code_search_tool.search_code(query)
            
            # Convert to DeepEval format
            return {
                "retrieved_documents": [
                    {
                        "id": result["file_path"],
                        "content": result["code_snippet"],
                        "relevance": result["relevance_score"]
                    }
                    for result in results["results"]
                ]
            }
        
        test_suite = TestSuite(name="Code Search", test_cases=test_cases)
        results = await self.engine.evaluate(test_suite, code_search_system)
        
        return results

# Usage example
async def evaluate_my_code_search():
    evaluator = CodeSearchEvaluator()
    
    # Your code search tool
    class MyCodeSearchTool:
        def search_code(self, query):
            # Your actual code search logic
            return {
                "results": [
                    {
                        "file_path": "src/models/neural_network.py",
                        "code_snippet": "def train_model(data, epochs=100):...",
                        "relevance_score": 0.95
                    }
                ]
            }
    
    # Test cases
    test_cases = [
        RetrievalTestCase(
            id="code_search_1",
            query="neural network training function",
            relevant_documents=[
                {"id": "neural_network.py", "content": "def train_model(data, epochs=100):", "relevance": 1.0}
            ],
            retrieved_documents=[
                {"id": "neural_network.py", "content": "def train_model(data, epochs=100):"},
                {"id": "data_loader.py", "content": "def load_dataset():"}
            ]
        )
    ]
    
    results = await evaluator.evaluate_code_search(MyCodeSearchTool(), test_cases)
    return results
```

### Code Search Test Cases

```python
# Comprehensive code search test cases
code_search_test_cases = [
    # Function search
    RetrievalTestCase(
        id="function_search_1",
        query="machine learning training function",
        relevant_documents=[
            {"id": "train.py", "content": "def train_model(data, epochs=100):", "relevance": 1.0},
            {"id": "model.py", "content": "def fit_model(X, y):", "relevance": 0.8}
        ],
        retrieved_documents=[
            {"id": "train.py", "content": "def train_model(data, epochs=100):"},
            {"id": "model.py", "content": "def fit_model(X, y):"},
            {"id": "utils.py", "content": "def preprocess_data(data):"},
            {"id": "config.py", "content": "MODEL_CONFIG = {}"}
        ]
    ),
    
    # Class search
    RetrievalTestCase(
        id="class_search_1",
        query="neural network class implementation",
        relevant_documents=[
            {"id": "neural_network.py", "content": "class NeuralNetwork:", "relevance": 1.0},
            {"id": "models.py", "content": "class MLP:", "relevance": 0.7}
        ],
        retrieved_documents=[
            {"id": "neural_network.py", "content": "class NeuralNetwork:"},
            {"id": "models.py", "content": "class MLP:"},
            {"id": "layers.py", "content": "class DenseLayer:"},
            {"id": "activations.py", "content": "def relu(x):"}
        ]
    ),
    
    # Algorithm search
    RetrievalTestCase(
        id="algorithm_search_1",
        query="gradient descent optimization",
        relevant_documents=[
            {"id": "optimizers.py", "content": "def gradient_descent(loss, params):", "relevance": 1.0},
            {"id": "training.py", "content": "def optimize_model(model, data):", "relevance": 0.8}
        ],
        retrieved_documents=[
            {"id": "optimizers.py", "content": "def gradient_descent(loss, params):"},
            {"id": "training.py", "content": "def optimize_model(model, data):"},
            {"id": "losses.py", "content": "def mse_loss(pred, target):"},
            {"id": "metrics.py", "content": "def accuracy_score(y_true, y_pred):"}
        ]
    )
]
```

## 🔄 Complete RAG Pipeline Evaluation

### End-to-End RAG Evaluation

```python
class RAGPipelineEvaluator:
    """Complete RAG pipeline evaluator"""
    
    def __init__(self):
        self.engine = EvaluationEngine()
        self._setup_comprehensive_metrics()
    
    def _setup_comprehensive_metrics(self):
        """Setup comprehensive metrics for RAG evaluation"""
        
        # Retrieval metrics
        self.engine.add_metric(RecallAtKMetric(k=5, threshold=0.7))
        self.engine.add_metric(MRRMetric(threshold=0.6))
        self.engine.add_metric(NDCGAtKMetric(k=5, threshold=0.8))
        
        # Answer quality metrics
        self.engine.add_metric(AccuracyMetric(threshold=0.7))
        self.engine.add_metric(CoherenceMetric(threshold=0.8))
        self.engine.add_metric(LatencyMetric(threshold=5.0))
        
        # Optional: LLM-based metrics (if OpenAI available)
        try:
            if os.getenv("OPENAI_API_KEY"):
                llm_config = LLMConfig(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model="gpt-3.5-turbo"
                )
                llm_provider = OpenAIProvider(llm_config)
                
                self.engine.add_metric(RelevanceMetric(llm_provider=llm_provider, threshold=0.7))
                self.engine.add_metric(FluencyMetric(llm_provider=llm_provider, threshold=0.8))
                self.engine.add_metric(FactualConsistencyMetric(llm_provider=llm_provider, threshold=0.8))
        except ImportError:
            print("⚠️  OpenAI not available, skipping LLM-based metrics")
    
    async def evaluate_rag_pipeline(self, rag_system, test_cases):
        """Evaluate complete RAG pipeline"""
        
        def rag_system_wrapper(input_data):
            query = input_data.get("query", "")
            return rag_system.rag_pipeline(query)
        
        test_suite = TestSuite(name="RAG Pipeline", test_cases=test_cases)
        results = await self.engine.evaluate(test_suite, rag_system_wrapper)
        
        return results

# Usage example
async def evaluate_complete_rag():
    evaluator = RAGPipelineEvaluator()
    
    # Your complete RAG system
    rag_system = RAGSystem()  # From the main example above
    
    # Comprehensive test cases
    test_cases = [
        TestCase(
            id="rag_end_to_end_1",
            input_data={"query": "What is machine learning?"},
            expected_output={
                "answer": "Machine learning is a subset of artificial intelligence...",
                "context_used": 3
            }
        ),
        TestCase(
            id="rag_end_to_end_2",
            input_data={"query": "How do neural networks work?"},
            expected_output={
                "answer": "Neural networks are computing systems...",
                "context_used": 2
            }
        )
    ]
    
    results = await evaluator.evaluate_rag_pipeline(rag_system, test_cases)
    return results
```

## 🚀 Advanced Evaluation Patterns

### 1. Multi-Stage Evaluation

```python
async def multi_stage_evaluation():
    """Evaluate RAG system in multiple stages"""
    
    # Stage 1: Document Ingestion
    print("Stage 1: Document Ingestion")
    ingestion_results = await evaluate_pdf_extraction()
    
    # Stage 2: Vector Store Performance
    print("Stage 2: Vector Store Performance")
    vector_results = await evaluate_document_retrieval()
    
    # Stage 3: End-to-End RAG
    print("Stage 3: End-to-End RAG")
    rag_results = await evaluate_rag_pipeline()
    
    # Stage 4: LLM-based Quality Assessment
    print("Stage 4: LLM-based Quality Assessment")
    llm_results = await evaluate_with_llm_metrics()
    
    return {
        "ingestion": ingestion_results,
        "vector_store": vector_results,
        "rag_pipeline": rag_results,
        "llm_quality": llm_results
    }
```

### 2. A/B Testing Framework

```python
class RAGABTester:
    """A/B testing framework for RAG systems"""
    
    def __init__(self, system_a, system_b):
        self.system_a = system_a
        self.system_b = system_b
        self.engine = EvaluationEngine()
    
    async def compare_systems(self, test_cases):
        """Compare two RAG systems"""
        
        # Test System A
        results_a = await self.engine.evaluate(
            TestSuite(name="System A", test_cases=test_cases),
            self.system_a.rag_pipeline
        )
        
        # Test System B
        results_b = await self.engine.evaluate(
            TestSuite(name="System B", test_cases=test_cases),
            self.system_b.rag_pipeline
        )
        
        # Compare results
        comparison = {
            "system_a": {
                "pass_rate": results_a["summary"]["pass_rate"],
                "average_score": results_a["summary"]["average_score"]
            },
            "system_b": {
                "pass_rate": results_b["summary"]["pass_rate"],
                "average_score": results_b["summary"]["average_score"]
            },
            "winner": "A" if results_a["summary"]["average_score"] > results_b["summary"]["average_score"] else "B"
        }
        
        return comparison

# Usage
async def ab_test_rag_systems():
    system_a = RAGSystem()  # Your current system
    system_b = RAGSystem()  # Your improved system
    
    ab_tester = RAGABTester(system_a, system_b)
    comparison = await ab_tester.compare_systems(test_cases)
    
    print(f"System A Score: {comparison['system_a']['average_score']:.3f}")
    print(f"System B Score: {comparison['system_b']['average_score']:.3f}")
    print(f"Winner: System {comparison['winner']}")
```

### 3. Continuous Evaluation

```python
import schedule
import time

class ContinuousRAGEvaluator:
    """Continuous evaluation for RAG systems"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.engine = EvaluationEngine()
        self.results_history = []
    
    async def run_evaluation(self):
        """Run periodic evaluation"""
        print(f"Running evaluation at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Your test cases
        test_cases = [
            TestCase(
                id="continuous_1",
                input_data={"query": "What is machine learning?"},
                expected_output={"answer": "Machine learning is..."}
            )
        ]
        
        results = await self.engine.evaluate(
            TestSuite(name="Continuous Evaluation", test_cases=test_cases),
            self.rag_system.rag_pipeline
        )
        
        # Store results
        self.results_history.append({
            "timestamp": time.time(),
            "results": results
        })
        
        # Alert if performance drops
        if results["summary"]["average_score"] < 0.7:
            print("⚠️  Performance alert: Score below threshold!")
        
        return results
    
    def start_continuous_evaluation(self):
        """Start continuous evaluation"""
        # Run every hour
        schedule.every().hour.do(lambda: asyncio.run(self.run_evaluation()))
        
        while True:
            schedule.run_pending()
            time.sleep(60)

# Usage
async def start_continuous_monitoring():
    rag_system = RAGSystem()
    evaluator = ContinuousRAGEvaluator(rag_system)
    
    # Run initial evaluation
    await evaluator.run_evaluation()
    
    # Start continuous monitoring
    evaluator.start_continuous_evaluation()
```

## 🌍 Real-World Examples

### Example 1: Legal Document RAG System

```python
class LegalDocumentRAG:
    """RAG system for legal documents"""
    
    def __init__(self):
        self.vector_store = LegalVectorStore()
        self.llm_provider = LegalLLMProvider()
    
    def rag_pipeline(self, query):
        # Retrieve relevant legal documents
        legal_docs = self.vector_store.search(query, k=5)
        
        # Generate legal advice
        context = " ".join([doc["content"] for doc in legal_docs])
        advice = self.llm_provider.generate_legal_advice(query, context)
        
        return {
            "query": query,
            "advice": advice,
            "legal_documents": legal_docs,
            "disclaimer": "This is not legal advice"
        }

# Evaluation for legal RAG
async def evaluate_legal_rag():
    legal_rag = LegalDocumentRAG()
    
    test_cases = [
        TestCase(
            id="legal_1",
            input_data={"query": "What are the requirements for a valid contract?"},
            expected_output={"advice": "A valid contract requires..."}
        )
    ]
    
    engine = EvaluationEngine()
    engine.add_metric(AccuracyMetric(threshold=0.8))
    engine.add_metric(CoherenceMetric(threshold=0.9))
    
    results = await engine.evaluate(
        TestSuite(name="Legal RAG", test_cases=test_cases),
        legal_rag.rag_pipeline
    )
    
    return results
```

### Example 2: Code Documentation RAG

```python
class CodeDocumentationRAG:
    """RAG system for code documentation"""
    
    def __init__(self):
        self.code_search = CodeSearchTool()
        self.doc_retriever = DocumentationRetriever()
        self.llm_provider = CodeLLMProvider()
    
    def rag_pipeline(self, query):
        # Search for relevant code
        code_results = self.code_search.search_code(query)
        
        # Retrieve documentation
        doc_results = self.doc_retriever.retrieve_docs(query)
        
        # Generate explanation
        context = {
            "code": code_results,
            "documentation": doc_results
        }
        explanation = self.llm_provider.explain_code(query, context)
        
        return {
            "query": query,
            "explanation": explanation,
            "code_examples": code_results,
            "documentation": doc_results
        }

# Evaluation for code documentation RAG
async def evaluate_code_doc_rag():
    code_doc_rag = CodeDocumentationRAG()
    
    test_cases = [
        TestCase(
            id="code_doc_1",
            input_data={"query": "How do I implement a neural network?"},
            expected_output={"explanation": "To implement a neural network..."}
        )
    ]
    
    engine = EvaluationEngine()
    engine.add_metric(RecallAtKMetric(k=5, threshold=0.7))
    engine.add_metric(AccuracyMetric(threshold=0.8))
    
    results = await engine.evaluate(
        TestSuite(name="Code Documentation RAG", test_cases=test_cases),
        code_doc_rag.rag_pipeline
    )
    
    return results
```

## 🏆 Best Practices

### 1. Test Case Design

```python
# Good test case design
good_test_cases = [
    TestCase(
        id="comprehensive_test_1",
        input_data={
            "query": "What is machine learning?",
            "context": "Technical documentation",
            "user_type": "beginner"
        },
        expected_output={
            "answer": "Machine learning is a subset of AI...",
            "complexity": "beginner_friendly",
            "sources": 3
        },
        metadata={
            "domain": "AI/ML",
            "difficulty": "easy",
            "expected_time": 2.0
        }
    )
]

# Bad test case design
bad_test_cases = [
    TestCase(
        id="vague_test_1",
        input_data={"query": "help"},  # Too vague
        expected_output={"answer": "good"}  # Too subjective
    )
]
```

### 2. Metric Selection

```python
# Choose metrics based on your use case
def setup_metrics_for_use_case(use_case):
    engine = EvaluationEngine()
    
    if use_case == "retrieval_focused":
        # Focus on retrieval quality
        engine.add_metric(RecallAtKMetric(k=5, threshold=0.7))
        engine.add_metric(MRRMetric(threshold=0.6))
        engine.add_metric(NDCGAtKMetric(k=5, threshold=0.8))
        
    elif use_case == "answer_quality_focused":
        # Focus on answer quality
        engine.add_metric(AccuracyMetric(threshold=0.8))
        engine.add_metric(CoherenceMetric(threshold=0.9))
        engine.add_metric(RelevanceMetric(threshold=0.8))
        
    elif use_case == "performance_focused":
        # Focus on performance
        engine.add_metric(LatencyMetric(threshold=2.0))
        engine.add_metric(AccuracyMetric(threshold=0.7))
        
    elif use_case == "comprehensive":
        # All metrics
        engine.add_metric(RecallAtKMetric(k=5, threshold=0.7))
        engine.add_metric(MRRMetric(threshold=0.6))
        engine.add_metric(AccuracyMetric(threshold=0.8))
        engine.add_metric(CoherenceMetric(threshold=0.8))
        engine.add_metric(LatencyMetric(threshold=3.0))
    
    return engine
```

### 3. Error Handling

```python
class RobustRAGEvaluator:
    """Robust RAG evaluator with error handling"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.engine = EvaluationEngine()
    
    async def evaluate_with_retry(self, test_suite, max_retries=3):
        """Evaluate with retry logic"""
        
        for attempt in range(max_retries):
            try:
                results = await self.engine.evaluate(
                    test_suite,
                    self.rag_system.rag_pipeline
                )
                return results
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == max_retries - 1:
                    return {
                        "error": str(e),
                        "status": "failed",
                        "attempts": max_retries
                    }
                
                # Wait before retry
                await asyncio.sleep(2 ** attempt)
        
        return {"error": "Max retries exceeded", "status": "failed"}
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. Low Retrieval Scores

```python
# Problem: Low Recall@K scores
# Solution: Improve vector search or add more relevant documents

def improve_retrieval_scores():
    # Check document quality
    documents = vector_store.get_all_documents()
    for doc in documents:
        if len(doc["content"]) < 100:  # Too short
            print(f"Document {doc['id']} is too short")
    
    # Check embedding quality
    embeddings = vector_store.get_embeddings()
    for i, emb in enumerate(embeddings):
        if len(emb) != expected_dimension:
            print(f"Embedding {i} has wrong dimension")
    
    # Add more relevant documents
    vector_store.add_documents(new_relevant_documents)
```

#### 2. High Latency

```python
# Problem: High latency in RAG pipeline
# Solution: Optimize each component

def optimize_rag_latency():
    # Optimize vector search
    vector_store.set_index_type("faiss")  # Use FAISS for faster search
    
    # Optimize LLM calls
    llm_provider.set_model("gpt-3.5-turbo")  # Use faster model
    
    # Optimize context length
    rag_system.set_max_context_length(1000)  # Limit context
    
    # Use caching
    rag_system.enable_caching()
```

#### 3. Poor Answer Quality

```python
# Problem: Poor answer quality
# Solution: Improve context and prompt engineering

def improve_answer_quality():
    # Improve context selection
    def better_context_selection(query, docs):
        # Use re-ranking
        reranked_docs = rerank_documents(query, docs)
        return reranked_docs[:3]  # Use top 3 most relevant
    
    # Improve prompt engineering
    def better_prompt(query, context):
        return f"""
        Based on the following context, answer the question accurately and concisely.
        
        Context: {context}
        
        Question: {query}
        
        Answer:
        """
    
    # Add answer validation
    def validate_answer(answer, context):
        # Check if answer is supported by context
        return check_factual_consistency(answer, context)
```

This comprehensive guide provides everything you need to evaluate your RAG, code search, and PDF extraction tools using DeepEval. Start with the basic examples and gradually implement more advanced patterns as your system evolves!
