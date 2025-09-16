"""
Security management for the DeepEval framework.

This module provides enterprise-grade security features including API key validation,
user authentication, and access control for multi-tenant environments.
"""

import hashlib
import hmac
import logging
import secrets
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SecurityManager:
    """Manages API key validation and user authentication."""

    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize security manager.
        
        Args:
            secret_key: Secret key for API key generation and validation
        """
        self.secret_key = secret_key or self._generate_secret_key()
        self.logger = logging.getLogger(f"{__name__}.SecurityManager")
        
        # In-memory storage for demo purposes
        # In production, this would be a database
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.users: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
        # Initialize with default admin user
        self._initialize_default_users()

    def _generate_secret_key(self) -> str:
        """Generate a secure secret key."""
        return secrets.token_urlsafe(32)

    def _initialize_default_users(self):
        """Initialize default users for demo purposes."""
        # Default admin user
        admin_key = self.generate_api_key("admin", "admin@deepeval.ai", ["admin", "user"])
        self.logger.info(f"Initialized default admin user with API key: {admin_key}")

    def generate_api_key(self, user_id: str, email: str, roles: List[str]) -> str:
        """
        Generate a new API key for a user.
        
        Args:
            user_id: Unique user identifier
            email: User email address
            roles: List of user roles
            
        Returns:
            Generated API key
        """
        # Generate API key
        timestamp = str(int(time.time()))
        data = f"{user_id}:{email}:{timestamp}"
        api_key = hmac.new(
            self.secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Store user and API key information
        self.users[user_id] = {
            "user_id": user_id,
            "email": email,
            "roles": roles,
            "created_at": timestamp,
            "active": True
        }
        
        self.api_keys[api_key] = {
            "user_id": user_id,
            "email": email,
            "roles": roles,
            "created_at": timestamp,
            "last_used": None,
            "usage_count": 0,
            "active": True
        }
        
        self.logger.info(f"Generated API key for user {user_id}")
        return api_key

    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not api_key or api_key not in self.api_keys:
            return False
        
        key_info = self.api_keys[api_key]
        if not key_info.get("active", False):
            return False
        
        # Update usage statistics
        key_info["last_used"] = str(int(time.time()))
        key_info["usage_count"] = key_info.get("usage_count", 0) + 1
        
        return True

    def get_current_user(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Get current user information from API key.
        
        Args:
            api_key: API key to lookup
            
        Returns:
            User information dictionary or None if invalid
        """
        if not self.validate_api_key(api_key):
            return None
        
        key_info = self.api_keys[api_key]
        user_id = key_info["user_id"]
        
        if user_id in self.users:
            user_info = self.users[user_id].copy()
            user_info["api_key"] = api_key
            return user_info
        
        return None

    def get_user_roles(self, api_key: str) -> List[str]:
        """
        Get user roles from API key.
        
        Args:
            api_key: API key to lookup
            
        Returns:
            List of user roles
        """
        user_info = self.get_current_user(api_key)
        if user_info:
            return user_info.get("roles", [])
        return []

    def has_role(self, api_key: str, role: str) -> bool:
        """
        Check if user has a specific role.
        
        Args:
            api_key: API key to check
            role: Role to check for
            
        Returns:
            True if user has the role, False otherwise
        """
        roles = self.get_user_roles(api_key)
        return role in roles

    def is_admin(self, api_key: str) -> bool:
        """
        Check if user is an admin.
        
        Args:
            api_key: API key to check
            
        Returns:
            True if user is admin, False otherwise
        """
        return self.has_role(api_key, "admin")

    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if revoked successfully, False otherwise
        """
        if api_key in self.api_keys:
            self.api_keys[api_key]["active"] = False
            self.logger.info(f"Revoked API key for user {self.api_keys[api_key]['user_id']}")
            return True
        return False

    def list_api_keys(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List API keys.
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            List of API key information
        """
        keys = []
        for api_key, info in self.api_keys.items():
            if user_id is None or info["user_id"] == user_id:
                key_info = info.copy()
                key_info["api_key"] = api_key
                keys.append(key_info)
        return keys

    def check_rate_limit(self, api_key: str, limit: int = 100, window: int = 3600) -> bool:
        """
        Check if user is within rate limits.
        
        Args:
            api_key: API key to check
            limit: Maximum requests per window
            window: Time window in seconds
            
        Returns:
            True if within limits, False if rate limited
        """
        current_time = int(time.time())
        window_start = current_time - window
        
        # Initialize rate limit tracking
        if api_key not in self.rate_limits:
            self.rate_limits[api_key] = {
                "requests": [],
                "blocked_until": None
            }
        
        rate_info = self.rate_limits[api_key]
        
        # Check if currently blocked
        if rate_info["blocked_until"] and current_time < rate_info["blocked_until"]:
            return False
        
        # Clean old requests
        rate_info["requests"] = [
            req_time for req_time in rate_info["requests"] 
            if req_time > window_start
        ]
        
        # Check if within limits
        if len(rate_info["requests"]) >= limit:
            # Block for the remainder of the window
            rate_info["blocked_until"] = current_time + window
            self.logger.warning(f"Rate limited API key: {api_key}")
            return False
        
        # Record this request
        rate_info["requests"].append(current_time)
        return True

    def get_rate_limit_status(self, api_key: str) -> Dict[str, Any]:
        """
        Get rate limit status for an API key.
        
        Args:
            api_key: API key to check
            
        Returns:
            Rate limit status information
        """
        if api_key not in self.rate_limits:
            return {
                "current_requests": 0,
                "blocked": False,
                "blocked_until": None
            }
        
        rate_info = self.rate_limits[api_key]
        current_time = int(time.time())
        
        return {
            "current_requests": len(rate_info["requests"]),
            "blocked": rate_info["blocked_until"] and current_time < rate_info["blocked_until"],
            "blocked_until": rate_info["blocked_until"]
        }

    def get_security_stats(self) -> Dict[str, Any]:
        """
        Get security statistics.
        
        Returns:
            Dictionary with security statistics
        """
        active_keys = sum(1 for info in self.api_keys.values() if info.get("active", False))
        total_users = len(self.users)
        
        return {
            "total_api_keys": len(self.api_keys),
            "active_api_keys": active_keys,
            "total_users": total_users,
            "rate_limited_keys": len([
                key for key, info in self.rate_limits.items()
                if info.get("blocked_until") and int(time.time()) < info["blocked_until"]
            ])
        }


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


def set_security_manager(security_manager: SecurityManager):
    """Set global security manager instance."""
    global _security_manager
    _security_manager = security_manager


def validate_api_key(api_key: str) -> bool:
    """
    Validate an API key using the global security manager.
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid, False otherwise
    """
    security_manager = get_security_manager()
    return security_manager.validate_api_key(api_key)


def get_current_user(api_key: str) -> Optional[Dict[str, Any]]:
    """
    Get current user information using the global security manager.
    
    Args:
        api_key: API key to lookup
        
    Returns:
        User information dictionary or None if invalid
    """
    security_manager = get_security_manager()
    return security_manager.get_current_user(api_key)
