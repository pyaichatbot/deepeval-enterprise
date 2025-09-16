"""
Custom metric registry for the DeepEval framework.

This module provides a registry system for custom metrics, allowing users
to register and manage their own evaluation metrics.
"""

import logging
from typing import Any, Dict, List, Type, Optional

from .models import Metric

logger = logging.getLogger(__name__)


class CustomMetricRegistry:
    """Registry for custom metrics."""

    def __init__(self):
        """Initialize the custom metric registry."""
        self.metrics: Dict[str, Type[Metric]] = {}
        self.logger = logging.getLogger(f"{__name__}.CustomMetricRegistry")

    def register(self, name: str, metric_class: Type[Metric]) -> None:
        """
        Register a custom metric.
        
        Args:
            name: Name to register the metric under
            metric_class: Metric class that inherits from Metric
            
        Raises:
            ValueError: If metric_class doesn't inherit from Metric
        """
        if not issubclass(metric_class, Metric):
            raise ValueError("Metric must inherit from Metric base class")
        
        self.metrics[name] = metric_class
        self.logger.info(f"Registered custom metric: {name}")

    def unregister(self, name: str) -> None:
        """
        Unregister a custom metric.
        
        Args:
            name: Name of the metric to unregister
        """
        if name in self.metrics:
            del self.metrics[name]
            self.logger.info(f"Unregistered custom metric: {name}")
        else:
            self.logger.warning(f"Attempted to unregister non-existent metric: {name}")

    def create_metric(self, name: str, **kwargs) -> Metric:
        """
        Create instance of registered metric.
        
        Args:
            name: Name of the registered metric
            **kwargs: Arguments to pass to metric constructor
            
        Returns:
            Instance of the metric
            
        Raises:
            ValueError: If metric is not registered
        """
        if name not in self.metrics:
            raise ValueError(f"Metric {name} not registered. Available metrics: {list(self.metrics.keys())}")
        
        try:
            return self.metrics[name](**kwargs)
        except Exception as e:
            self.logger.error(f"Error creating metric {name}: {e}")
            raise

    def list_metrics(self) -> List[str]:
        """
        List all registered metrics.
        
        Returns:
            List of registered metric names
        """
        return list(self.metrics.keys())

    def get_metric_class(self, name: str) -> Optional[Type[Metric]]:
        """
        Get the metric class for a registered metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            Metric class or None if not found
        """
        return self.metrics.get(name)

    def is_registered(self, name: str) -> bool:
        """
        Check if a metric is registered.
        
        Args:
            name: Name of the metric to check
            
        Returns:
            True if metric is registered, False otherwise
        """
        return name in self.metrics

    def get_metric_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            Dictionary with metric information or None if not found
        """
        if name not in self.metrics:
            return None
        
        metric_class = self.metrics[name]
        return {
            "name": name,
            "class": metric_class.__name__,
            "module": metric_class.__module__,
            "docstring": metric_class.__doc__,
            "base_classes": [cls.__name__ for cls in metric_class.__bases__]
        }

    def list_metric_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered metrics.
        
        Returns:
            Dictionary mapping metric names to their information
        """
        return {name: self.get_metric_info(name) for name in self.metrics.keys()}

    def clear(self) -> None:
        """Clear all registered metrics."""
        self.metrics.clear()
        self.logger.info("Cleared all custom metrics")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        return {
            "total_metrics": len(self.metrics),
            "metric_names": list(self.metrics.keys()),
            "metric_classes": [cls.__name__ for cls in self.metrics.values()]
        }


# Global registry instance
_registry: Optional[CustomMetricRegistry] = None


def get_custom_metric_registry() -> CustomMetricRegistry:
    """Get global custom metric registry instance."""
    global _registry
    if _registry is None:
        _registry = CustomMetricRegistry()
    return _registry


def set_custom_metric_registry(registry: CustomMetricRegistry):
    """Set global custom metric registry instance."""
    global _registry
    _registry = registry


def register_custom_metric(name: str, metric_class: Type[Metric]) -> None:
    """
    Register a custom metric in the global registry.
    
    Args:
        name: Name to register the metric under
        metric_class: Metric class that inherits from Metric
    """
    registry = get_custom_metric_registry()
    registry.register(name, metric_class)


def create_custom_metric(name: str, **kwargs) -> Metric:
    """
    Create instance of registered custom metric from global registry.
    
    Args:
        name: Name of the registered metric
        **kwargs: Arguments to pass to metric constructor
        
    Returns:
        Instance of the metric
    """
    registry = get_custom_metric_registry()
    return registry.create_metric(name, **kwargs)


def list_custom_metrics() -> List[str]:
    """
    List all registered custom metrics from global registry.
    
    Returns:
        List of registered metric names
    """
    registry = get_custom_metric_registry()
    return registry.list_metrics()
