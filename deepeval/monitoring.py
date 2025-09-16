"""
Monitoring and observability for the DeepEval framework.

This module provides comprehensive monitoring, metrics collection,
health checks, and observability features.
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json

# Optional integrations
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


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    timestamp: datetime


@dataclass
class EvaluationMetrics:
    """Evaluation-specific metrics."""
    total_evaluations: int
    successful_evaluations: int
    failed_evaluations: int
    average_execution_time: float
    active_evaluations: int
    metrics_used: Dict[str, int]
    timestamp: datetime


class MetricsCollector:
    """Collect and expose metrics for monitoring."""

    def __init__(self, max_history: int = 1000):
        self.metrics = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "average_execution_time": 0.0,
            "active_evaluations": 0,
            "metrics_used": defaultdict(int)
        }
        self.lock = threading.Lock()
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        self.logger = logging.getLogger(__name__)

    def increment_evaluations(self):
        """Increment total evaluations counter."""
        with self.lock:
            self.metrics["total_evaluations"] += 1

    def increment_successful(self):
        """Increment successful evaluations counter."""
        with self.lock:
            self.metrics["successful_evaluations"] += 1

    def increment_failed(self):
        """Increment failed evaluations counter."""
        with self.lock:
            self.metrics["failed_evaluations"] += 1

    def update_execution_time(self, execution_time: float):
        """Update average execution time."""
        with self.lock:
            current_avg = self.metrics["average_execution_time"]
            total = self.metrics["total_evaluations"]
            # Running average
            self.metrics["average_execution_time"] = ((current_avg * (total - 1)) + execution_time) / total

    def set_active_evaluations(self, count: int):
        """Set number of active evaluations."""
        with self.lock:
            self.metrics["active_evaluations"] = count

    def record_metric_usage(self, metric_name: str):
        """Record usage of a specific metric."""
        with self.lock:
            self.metrics["metrics_used"][metric_name] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self.lock:
            metrics_copy = self.metrics.copy()
            metrics_copy["metrics_used"] = dict(metrics_copy["metrics_used"])
            metrics_copy["timestamp"] = datetime.now(timezone.utc).isoformat()
            return metrics_copy

    def get_metrics_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get metrics history."""
        with self.lock:
            history_list = list(self.history)
            if limit:
                history_list = history_list[-limit:]
            return history_list

    def record_snapshot(self):
        """Record current metrics as a snapshot."""
        with self.lock:
            snapshot = self.get_metrics()
            self.history.append(snapshot)

    def get_success_rate(self) -> float:
        """Calculate success rate."""
        with self.lock:
            total = self.metrics["total_evaluations"]
            if total == 0:
                return 0.0
            return self.metrics["successful_evaluations"] / total

    def get_failure_rate(self) -> float:
        """Calculate failure rate."""
        with self.lock:
            total = self.metrics["total_evaluations"]
            if total == 0:
                return 0.0
            return self.metrics["failed_evaluations"] / total


class SystemMonitor:
    """Monitor system resources and health."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_history = deque(maxlen=1000)

    def start_monitoring(self, interval: float = 30.0):
        """Start system monitoring in background thread."""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("System monitoring started")

    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("System monitoring stopped")

    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self.get_system_metrics()
                self.metrics_history.append(metrics)
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
            
            time.sleep(interval)

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        if not PSUTIL_AVAILABLE:
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_gb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0,
                timestamp=datetime.now(timezone.utc)
            )

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)

            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_gb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0,
                timestamp=datetime.now(timezone.utc)
            )

    def get_metrics_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get system metrics history."""
        history_list = list(self.metrics_history)
        if limit:
            history_list = history_list[-limit:]
        return [asdict(metrics) for metrics in history_list]

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics as dictionary."""
        metrics = self.get_system_metrics()
        return asdict(metrics)


class HealthChecker:
    """Health check for the evaluation system."""

    def __init__(self, engine=None, cache_manager=None):
        self.engine = engine
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)

    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {}
        }

        # Database connectivity
        try:
            if self.engine and hasattr(self.engine, 'db_manager') and self.engine.db_manager:
                with self.engine.db_manager.get_session() as session:
                    session.execute("SELECT 1")
                health_status["checks"]["database"] = "healthy"
            else:
                health_status["checks"]["database"] = "not_configured"
        except Exception as e:
            health_status["checks"]["database"] = f"unhealthy: {str(e)}"
            health_status["status"] = "unhealthy"

        # System resources
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

                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                health_status["checks"]["cpu"] = {
                    "usage_percent": cpu_percent
                }
            else:
                health_status["checks"]["system_resources"] = "psutil_not_available"
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["checks"]["system_resources"] = f"error: {str(e)}"
            health_status["status"] = "degraded"

        # Cache status
        try:
            if self.cache_manager:
                if hasattr(self.cache_manager, 'cache_type'):
                    if self.cache_manager.cache_type == 'redis':
                        if REDIS_AVAILABLE:
                            self.cache_manager.redis_client.ping()
                            health_status["checks"]["redis_cache"] = "healthy"
                        else:
                            health_status["checks"]["redis_cache"] = "redis_not_available"
                            health_status["status"] = "degraded"
                    else:
                        health_status["checks"]["cache"] = "memory_cache_healthy"
                else:
                    health_status["checks"]["cache"] = "not_configured"
            else:
                health_status["checks"]["cache"] = "not_configured"
        except Exception as e:
            health_status["checks"]["cache"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"

        # Evaluation engine status
        try:
            if self.engine:
                metrics_count = len(self.engine.get_metrics())
                health_status["checks"]["evaluation_engine"] = {
                    "status": "healthy",
                    "metrics_loaded": metrics_count
                }
            else:
                health_status["checks"]["evaluation_engine"] = "not_configured"
        except Exception as e:
            health_status["checks"]["evaluation_engine"] = f"unhealthy: {str(e)}"
            health_status["status"] = "unhealthy"

        return health_status

    def check_dependencies(self) -> Dict[str, Any]:
        """Check availability of optional dependencies."""
        dependencies = {
            "psutil": PSUTIL_AVAILABLE,
            "redis": REDIS_AVAILABLE,
            "pandas": True,  # Core dependency
            "numpy": True,   # Core dependency
        }

        # Check for optional LLM providers
        try:
            import openai
            dependencies["openai"] = True
        except ImportError:
            dependencies["openai"] = False

        try:
            import anthropic
            dependencies["anthropic"] = True
        except ImportError:
            dependencies["anthropic"] = False

        try:
            import google.generativeai
            dependencies["google_generativeai"] = True
        except ImportError:
            dependencies["google_generativeai"] = False

        return dependencies


class AlertManager:
    """Manage alerts and notifications."""

    def __init__(self):
        self.alerts = []
        self.alert_callbacks = []
        self.logger = logging.getLogger(__name__)

    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a callback function for alerts."""
        self.alert_callbacks.append(callback)

    def trigger_alert(self, alert_type: str, message: str, severity: str = "warning", metadata: Optional[Dict[str, Any]] = None):
        """Trigger an alert."""
        alert = {
            "id": f"alert_{int(time.time())}",
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }

        self.alerts.append(alert)
        self.logger.warning(f"Alert triggered: {alert_type} - {message}")

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")

    def get_alerts(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        alerts = self.alerts
        if limit:
            alerts = alerts[-limit:]
        return alerts

    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()


class PerformanceProfiler:
    """Profile performance of evaluation operations."""

    def __init__(self):
        self.profiles = {}
        self.logger = logging.getLogger(__name__)

    def start_profile(self, name: str):
        """Start profiling an operation."""
        self.profiles[name] = {
            "start_time": time.time(),
            "end_time": None,
            "duration": None
        }

    def end_profile(self, name: str):
        """End profiling an operation."""
        if name in self.profiles:
            end_time = time.time()
            start_time = self.profiles[name]["start_time"]
            self.profiles[name]["end_time"] = end_time
            self.profiles[name]["duration"] = end_time - start_time

    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """Get profile data for an operation."""
        return self.profiles.get(name)

    def get_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all profile data."""
        return self.profiles.copy()

    def clear_profiles(self):
        """Clear all profile data."""
        self.profiles.clear()


class MonitoringDashboard:
    """Dashboard for monitoring data visualization."""

    def __init__(self, metrics_collector: MetricsCollector, system_monitor: SystemMonitor):
        self.metrics_collector = metrics_collector
        self.system_monitor = system_monitor
        self.logger = logging.getLogger(__name__)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "evaluation_metrics": self.metrics_collector.get_metrics(),
            "system_metrics": self.system_monitor.get_current_metrics(),
            "system_history": self.system_monitor.get_metrics_history(limit=100),
            "evaluation_history": self.metrics_collector.get_metrics_history(limit=100),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        data = self.get_dashboard_data()
        
        if format == "json":
            return json.dumps(data, indent=2, default=str)
        elif format == "csv":
            # Convert to CSV format (simplified)
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write headers
            writer.writerow(["metric", "value", "timestamp"])
            
            # Write evaluation metrics
            eval_metrics = data["evaluation_metrics"]
            for key, value in eval_metrics.items():
                if key != "timestamp":
                    writer.writerow([f"evaluation.{key}", value, eval_metrics.get("timestamp", "")])
            
            # Write system metrics
            sys_metrics = data["system_metrics"]
            for key, value in sys_metrics.items():
                if key != "timestamp":
                    writer.writerow([f"system.{key}", value, sys_metrics.get("timestamp", "")])
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global monitoring instances
_metrics_collector: Optional[MetricsCollector] = None
_system_monitor: Optional[SystemMonitor] = None
_health_checker: Optional[HealthChecker] = None
_alert_manager: Optional[AlertManager] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_system_monitor() -> SystemMonitor:
    """Get the global system monitor."""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    return _system_monitor


def get_health_checker(engine=None, cache_manager=None) -> HealthChecker:
    """Get the global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker(engine, cache_manager)
    return _health_checker


def get_alert_manager() -> AlertManager:
    """Get the global alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
