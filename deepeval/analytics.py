"""
Advanced analytics and result analysis for the DeepEval framework.

This module provides comprehensive analysis capabilities for evaluation results,
including statistical analysis, trend detection, and business intelligence features.
"""

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from .models import EvaluationResult
from .cache import get_cache_manager

logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """Analyze evaluation results and generate insights."""

    def __init__(self, cache_manager=None):
        """
        Initialize result analyzer.
        
        Args:
            cache_manager: Optional cache manager for performance optimization
        """
        self.cache_manager = cache_manager or get_cache_manager()
        self.logger = logging.getLogger(f"{__name__}.ResultAnalyzer")

    def analyze_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Comprehensive analysis of evaluation results.
        
        Args:
            results: List of evaluation results to analyze
            
        Returns:
            Dictionary containing comprehensive analysis
        """
        if not results:
            return {"error": "No results to analyze"}
        
        try:
            # Convert results to DataFrame for analysis
            df = pd.DataFrame([asdict(r) for r in results])
            
            analysis = {
                "overall_statistics": self._calculate_overall_stats(df),
                "metric_analysis": self._analyze_by_metric(df),
                "performance_trends": self._analyze_performance_trends(df),
                "outlier_detection": self._detect_outliers(df),
                "correlation_analysis": self._correlation_analysis(df),
                "summary": self._generate_summary(df)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing results: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def _calculate_overall_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall statistics."""
        try:
            return {
                "total_evaluations": len(df),
                "pass_rate": float(df['passed'].mean()) if 'passed' in df.columns else 0.0,
                "average_score": float(df['score'].mean()) if 'score' in df.columns else 0.0,
                "score_std": float(df['score'].std()) if 'score' in df.columns else 0.0,
                "score_distribution": {
                    "min": float(df['score'].min()) if 'score' in df.columns else 0.0,
                    "q25": float(df['score'].quantile(0.25)) if 'score' in df.columns else 0.0,
                    "median": float(df['score'].median()) if 'score' in df.columns else 0.0,
                    "q75": float(df['score'].quantile(0.75)) if 'score' in df.columns else 0.0,
                    "max": float(df['score'].max()) if 'score' in df.columns else 0.0
                },
                "execution_time_stats": {
                    "mean": float(df['execution_time'].mean()) if 'execution_time' in df.columns else 0.0,
                    "std": float(df['execution_time'].std()) if 'execution_time' in df.columns else 0.0,
                    "min": float(df['execution_time'].min()) if 'execution_time' in df.columns else 0.0,
                    "max": float(df['execution_time'].max()) if 'execution_time' in df.columns else 0.0
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating overall stats: {e}")
            return {"error": str(e)}

    def _analyze_by_metric(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze results by metric type."""
        try:
            metric_analysis = {}
            
            if 'metric_name' not in df.columns:
                return metric_analysis
            
            for metric in df['metric_name'].unique():
                metric_data = df[df['metric_name'] == metric]
                metric_analysis[metric] = {
                    "count": len(metric_data),
                    "pass_rate": float(metric_data['passed'].mean()) if 'passed' in metric_data.columns else 0.0,
                    "average_score": float(metric_data['score'].mean()) if 'score' in metric_data.columns else 0.0,
                    "score_std": float(metric_data['score'].std()) if 'score' in metric_data.columns else 0.0,
                    "average_execution_time": float(metric_data['execution_time'].mean()) if 'execution_time' in metric_data.columns else 0.0
                }
            
            return metric_analysis
        except Exception as e:
            self.logger.error(f"Error analyzing by metric: {e}")
            return {"error": str(e)}

    def _analyze_performance_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance trends over test cases."""
        try:
            if len(df) < 2:
                return {"scores_over_time": [], "latency_over_time": []}
            
            # Sort by test_case_id to see trends
            df_sorted = df.sort_values('test_case_id')
            
            # Calculate rolling averages if we have enough data
            if len(df_sorted) > 5:
                df_sorted['rolling_avg_score'] = df_sorted['score'].rolling(window=5, min_periods=1).mean()
                df_sorted['rolling_avg_latency'] = df_sorted['execution_time'].rolling(window=5, min_periods=1).mean()
            
            # Prepare trend data
            trend_data = {
                "scores_over_time": [],
                "latency_over_time": []
            }
            
            if 'rolling_avg_score' in df_sorted.columns:
                trend_data["scores_over_time"] = df_sorted[['test_case_id', 'score', 'rolling_avg_score']].to_dict(orient='records')
            else:
                trend_data["scores_over_time"] = df_sorted[['test_case_id', 'score']].to_dict(orient='records')
            
            if 'rolling_avg_latency' in df_sorted.columns:
                trend_data["latency_over_time"] = df_sorted[['test_case_id', 'execution_time', 'rolling_avg_latency']].to_dict(orient='records')
            else:
                trend_data["latency_over_time"] = df_sorted[['test_case_id', 'execution_time']].to_dict(orient='records')
            
            return trend_data
        except Exception as e:
            self.logger.error(f"Error analyzing performance trends: {e}")
            return {"error": str(e)}

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in scores and execution times."""
        try:
            outliers = {}
            
            # Score outliers (below 1st percentile)
            if 'score' in df.columns and len(df) > 0:
                score_lower_bound = df['score'].quantile(0.01)
                low_score_cases = df[df['score'] < score_lower_bound]
                outliers['low_score_test_cases'] = low_score_cases[['test_case_id', 'metric_name', 'score']].to_dict(orient='records')
            
            # Latency outliers (above 99th percentile)
            if 'execution_time' in df.columns and len(df) > 0:
                latency_upper_bound = df['execution_time'].quantile(0.99)
                high_latency_cases = df[df['execution_time'] > latency_upper_bound]
                outliers['high_latency_test_cases'] = high_latency_cases[['test_case_id', 'metric_name', 'execution_time']].to_dict(orient='records')
            
            return outliers
        except Exception as e:
            self.logger.error(f"Error detecting outliers: {e}")
            return {"error": str(e)}

    def _correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between metrics or other factors."""
        try:
            correlations = {}
            
            # Correlation between score and execution time
            if 'score' in df.columns and 'execution_time' in df.columns:
                corr_value = df['score'].corr(df['execution_time'])
                correlations['score_vs_execution_time'] = float(corr_value) if not pd.isna(corr_value) else 0.0
            
            # Correlation matrix for numerical metrics
            numerical_metrics = df.select_dtypes(include=[np.number])
            if not numerical_metrics.empty and len(numerical_metrics.columns) > 1:
                corr_matrix = numerical_metrics.corr()
                correlations['metric_correlation_matrix'] = corr_matrix.to_dict()
            
            return correlations
        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {e}")
            return {"error": str(e)}

    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate executive summary of results."""
        try:
            summary = {
                "total_tests": len(df),
                "unique_metrics": len(df['metric_name'].unique()) if 'metric_name' in df.columns else 0,
                "overall_pass_rate": float(df['passed'].mean()) if 'passed' in df.columns else 0.0,
                "average_score": float(df['score'].mean()) if 'score' in df.columns else 0.0,
                "performance_grade": self._calculate_performance_grade(df)
            }
            
            return summary
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return {"error": str(e)}

    def _calculate_performance_grade(self, df: pd.DataFrame) -> str:
        """Calculate overall performance grade."""
        try:
            if 'score' not in df.columns or len(df) == 0:
                return "N/A"
            
            avg_score = df['score'].mean()
            pass_rate = df['passed'].mean() if 'passed' in df.columns else 0.0
            
            # Simple grading system
            if avg_score >= 0.9 and pass_rate >= 0.9:
                return "A"
            elif avg_score >= 0.8 and pass_rate >= 0.8:
                return "B"
            elif avg_score >= 0.7 and pass_rate >= 0.7:
                return "C"
            elif avg_score >= 0.6 and pass_rate >= 0.6:
                return "D"
            else:
                return "F"
        except Exception as e:
            self.logger.error(f"Error calculating performance grade: {e}")
            return "Error"

    def generate_report(self, results: List[EvaluationResult], format: str = "json") -> str:
        """
        Generate formatted report from analysis.
        
        Args:
            results: List of evaluation results
            format: Report format ("json" or "summary")
            
        Returns:
            Formatted report string
        """
        try:
            analysis = self.analyze_results(results)
            
            if format == "json":
                import json
                return json.dumps(analysis, indent=2, default=str)
            elif format == "summary":
                return self._format_summary_report(analysis)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return f"Report generation failed: {str(e)}"

    def _format_summary_report(self, analysis: Dict[str, Any]) -> str:
        """Format analysis as human-readable summary."""
        try:
            summary = analysis.get("summary", {})
            overall_stats = analysis.get("overall_statistics", {})
            
            report = f"""
DeepEval Analysis Report
=======================

Overall Performance:
- Total Tests: {summary.get('total_tests', 0)}
- Pass Rate: {summary.get('overall_pass_rate', 0):.1%}
- Average Score: {summary.get('average_score', 0):.3f}
- Performance Grade: {summary.get('performance_grade', 'N/A')}

Statistics:
- Total Evaluations: {overall_stats.get('total_evaluations', 0)}
- Score Range: {overall_stats.get('score_distribution', {}).get('min', 0):.3f} - {overall_stats.get('score_distribution', {}).get('max', 0):.3f}
- Average Execution Time: {overall_stats.get('execution_time_stats', {}).get('mean', 0):.3f}s

Metric Breakdown:
"""
            
            metric_analysis = analysis.get("metric_analysis", {})
            for metric_name, metric_data in metric_analysis.items():
                report += f"- {metric_name}: {metric_data.get('pass_rate', 0):.1%} pass rate, {metric_data.get('average_score', 0):.3f} avg score\n"
            
            return report.strip()
            
        except Exception as e:
            self.logger.error(f"Error formatting summary report: {e}")
            return f"Summary formatting failed: {str(e)}"


# Global analyzer instance
_analyzer: Optional[ResultAnalyzer] = None


def get_result_analyzer() -> ResultAnalyzer:
    """Get global result analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = ResultAnalyzer()
    return _analyzer


def set_result_analyzer(analyzer: ResultAnalyzer):
    """Set global result analyzer instance."""
    global _analyzer
    _analyzer = analyzer
