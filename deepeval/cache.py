"""
Cache management for the DeepEval framework.

This module provides caching capabilities for expensive operations,
supporting both in-memory and Redis-based caching for enterprise scalability.
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Optional Redis support
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CacheManager:
    """Manage caching for expensive operations."""

    def __init__(self, cache_type: str = "memory", redis_url: Optional[str] = None):
        """
        Initialize cache manager.
        
        Args:
            cache_type: Type of cache ("memory" or "redis")
            redis_url: Redis connection URL (required for redis cache_type)
        """
        self.cache_type = cache_type
        self.logger = logging.getLogger(f"{__name__}.CacheManager")
        
        if cache_type == "memory":
            self.cache = {}
            self.logger.info("Initialized memory cache")
        elif cache_type == "redis":
            if not REDIS_AVAILABLE:
                raise ImportError("Redis library not available. Install with: pip install redis")
            
            try:
                if redis_url:
                    self.redis_client = redis.from_url(redis_url)
                else:
                    self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                
                # Test connection
                self.redis_client.ping()
                self.logger.info("Initialized Redis cache")
            except Exception as e:
                raise ConnectionError(f"Could not connect to Redis: {e}")
        else:
            raise ValueError(f"Unsupported cache type: {cache_type}")

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            if self.cache_type == "memory":
                return self.cache.get(key)
            elif self.cache_type == "redis":
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
        except Exception as e:
            self.logger.error(f"Error getting cache key {key}: {e}")
            return None
        
        return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        """
        Set cached value.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (for Redis)
        """
        try:
            if self.cache_type == "memory":
                self.cache[key] = value
            elif self.cache_type == "redis":
                self.redis_client.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            self.logger.error(f"Error setting cache key {key}: {e}")

    def delete(self, key: str):
        """
        Delete cached value.
        
        Args:
            key: Cache key to delete
        """
        try:
            if self.cache_type == "memory":
                self.cache.pop(key, None)
            elif self.cache_type == "redis":
                self.redis_client.delete(key)
        except Exception as e:
            self.logger.error(f"Error deleting cache key {key}: {e}")

    def clear(self):
        """Clear all cached values."""
        try:
            if self.cache_type == "memory":
                self.cache.clear()
                self.logger.info("Cleared memory cache")
            elif self.cache_type == "redis":
                self.redis_client.flushdb()
                self.logger.info("Cleared Redis cache")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            if self.cache_type == "memory":
                return key in self.cache
            elif self.cache_type == "redis":
                return bool(self.redis_client.exists(key))
        except Exception as e:
            self.logger.error(f"Error checking cache key {key}: {e}")
            return False
        
        return False

    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "cache_type": self.cache_type,
            "redis_available": REDIS_AVAILABLE
        }
        
        try:
            if self.cache_type == "memory":
                stats["memory_cache_size"] = len(self.cache)
            elif self.cache_type == "redis":
                info = self.redis_client.info()
                stats.update({
                    "redis_used_memory": info.get("used_memory_human", "unknown"),
                    "redis_connected_clients": info.get("connected_clients", 0),
                    "redis_total_commands_processed": info.get("total_commands_processed", 0)
                })
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            stats["error"] = str(e)
        
        return stats


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def set_cache_manager(cache_manager: CacheManager):
    """Set global cache manager instance."""
    global _cache_manager
    _cache_manager = cache_manager
