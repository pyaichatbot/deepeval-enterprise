#!/usr/bin/env python3
"""
Example usage of the DeepEval framework.

This script demonstrates how to use the refactored DeepEval framework
with both standard metrics and retrieval metrics.
"""

import asyncio
import os
from typing import Dict, Any

# Import the refactored DeepEval framework
from deepeval import (
    EvaluationEngine, EvaluationConfig, TestSuite, TestCase,
    AccuracyMetric, CoherenceMetric, LatencyMetric,
    RecallAtKMetric, MRRMetric, NDCGAtKMetric,
    create_test_suite
)


async def my_ai_system(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Mock AI system for demonstration."""
    question = input_data.get("question", "")
    
    # Simple mock responses
    if "France" in question:
        return {"answer": "Paris", "response_time": 0.1}
    elif "quantum" in question:
        return {"answer": "Quantum computing uses quantum bits to process information.", "response_time": 0.2}
    elif "retrieval" in question.lower():
        # Mock retrieval system response
        return {
            "retrieved_documents": [
                {"id": "doc1", "content": "Document about retrieval systems"},
                {"id": "doc2", "content": "Information retrieval techniques"},
                {"id": "doc3", "content": "Search algorithms and ranking"}
            ],
            "response_time": 0.15
        }
    else:
        return {"answer": "I don't know", "response_time": 0.05}


async def main():
    """Main example function."""
    print("🚀 DeepEval Framework Example")
    print("=" * 50)
    
    # 1. Configuration
    config = EvaluationConfig(
        max_workers=5,
        timeout=30,
        parallel_execution=True,
        debug=True
    )
    
    # 2. Initialize engine
    engine = EvaluationEngine(config)
    print("✅ Evaluation engine initialized")
    
    # 3. Add standard metrics
    engine.add_metric(AccuracyMetric(threshold=0.8))
    engine.add_metric(CoherenceMetric(threshold=0.6))
    engine.add_metric(LatencyMetric(threshold=1.0))
    print("✅ Standard metrics added")
    
    # 4. Create test suite for standard evaluation
    test_cases_data = [
        {
            "input": {"question": "What is the capital of France?"},
            "expected": {"answer": "Paris"},
            "metadata": {"category": "Geography"}
        },
        {
            "input": {"question": "Explain quantum computing."},
            "expected": {"answer": "Quantum computing uses quantum bits to process information."},
            "metadata": {"category": "Science"}
        },
        {
            "input": {"question": "What is 2+2?"},
            "expected": {"answer": "4"},
            "metadata": {"category": "Math"}
        }
    ]
    
    test_suite = create_test_suite("Standard Test Suite", test_cases_data)
    print("✅ Standard test suite created")
    
    # 5. Run standard evaluation
    print("\n📊 Running standard evaluation...")
    results = await engine.evaluate(test_suite, my_ai_system, "Standard Evaluation")
    
    print(f"✅ Standard evaluation completed!")
    print(f"   - Run ID: {results['run_id']}")
    print(f"   - Status: {results['status']}")
    print(f"   - Total tests: {results['summary']['total_evaluations']}")
    print(f"   - Pass rate: {results['summary']['pass_rate']:.2%}")
    print(f"   - Average score: {results['summary']['average_score']:.3f}")
    
    # 6. Add retrieval metrics
    engine.clear_metrics()
    engine.add_metric(RecallAtKMetric(k=5, threshold=0.6))
    engine.add_metric(MRRMetric(threshold=0.5))
    engine.add_metric(NDCGAtKMetric(k=5, threshold=0.7))
    print("\n✅ Retrieval metrics added")
    
    # 7. Create retrieval test suite
    retrieval_test_cases_data = [
        {
            "input": {"query": "retrieval systems"},
            "expected": {
                "relevant_documents": [
                    {"id": "doc1", "content": "Document about retrieval systems", "relevance": 1.0},
                    {"id": "doc2", "content": "Information retrieval techniques", "relevance": 0.8}
                ]
            },
            "metadata": {"category": "Retrieval"}
        },
        {
            "input": {"query": "search algorithms"},
            "expected": {
                "relevant_documents": [
                    {"id": "doc3", "content": "Search algorithms and ranking", "relevance": 1.0},
                    {"id": "doc1", "content": "Document about retrieval systems", "relevance": 0.6}
                ]
            },
            "metadata": {"category": "Retrieval"}
        }
    ]
    
    retrieval_test_suite = create_test_suite("Retrieval Test Suite", retrieval_test_cases_data)
    print("✅ Retrieval test suite created")
    
    # 8. Run retrieval evaluation
    print("\n🔍 Running retrieval evaluation...")
    retrieval_results = await engine.evaluate(
        retrieval_test_suite, 
        my_ai_system, 
        "Retrieval Evaluation"
    )
    
    print(f"✅ Retrieval evaluation completed!")
    print(f"   - Run ID: {retrieval_results['run_id']}")
    print(f"   - Status: {retrieval_results['status']}")
    print(f"   - Total tests: {retrieval_results['summary']['total_evaluations']}")
    print(f"   - Pass rate: {retrieval_results['summary']['pass_rate']:.2%}")
    print(f"   - Average score: {retrieval_results['summary']['average_score']:.3f}")
    
    # 9. Generate reports
    print("\n📄 Generating reports...")
    
    # JSON report
    json_report = engine.generate_report(results["results"], format="json")
    with open("deepeval_report.json", "w") as f:
        f.write(json_report)
    print("✅ JSON report saved: deepeval_report.json")
    
    # HTML report
    html_report = engine.generate_report(results["results"], format="html")
    with open("deepeval_report.html", "w") as f:
        f.write(html_report)
    print("✅ HTML report saved: deepeval_report.html")
    
    # Retrieval report
    retrieval_json_report = engine.generate_report(retrieval_results["results"], format="json")
    with open("retrieval_report.json", "w") as f:
        f.write(retrieval_json_report)
    print("✅ Retrieval JSON report saved: retrieval_report.json")
    
    # 10. Show detailed results
    print("\n📋 Detailed Results:")
    print("-" * 30)
    
    for result in results["results"]:
        print(f"Test: {result.test_case_id}")
        print(f"  Metric: {result.metric_name}")
        print(f"  Score: {result.score:.3f}")
        print(f"  Passed: {'✅' if result.passed else '❌'}")
        print(f"  Explanation: {result.explanation}")
        print()
    
    print("\n🎉 Example completed successfully!")
    print("\nTo run the API server:")
    print("  uvicorn deepeval.api.app:app --reload")
    print("\nTo view the API documentation:")
    print("  http://localhost:8000/docs")


if __name__ == "__main__":
    asyncio.run(main())
