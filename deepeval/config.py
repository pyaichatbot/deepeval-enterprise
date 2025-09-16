"""
Configuration management for the DeepEval framework.

This module handles loading, validation, and management of configuration
settings for the evaluation framework.
"""

import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field, validator


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20
    pool_recycle: int = 3600
    pool_pre_ping: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    cache_type: str = "memory"  # memory, redis
    redis_url: Optional[str] = None
    max_size: int = 1000
    ttl: int = 3600  # seconds


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    api_key_validation: bool = True
    rate_limiting: bool = True
    max_requests_per_minute: int = 100
    allowed_origins: List[str] = None
    enable_cors: bool = True

    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["*"]


class EvaluationConfig(BaseModel):
    """Configuration for evaluation runs."""
    
    # Execution settings
    max_workers: int = Field(default=10, ge=1, le=100)
    timeout: int = Field(default=300, ge=1, le=3600)
    retry_attempts: int = Field(default=3, ge=0, le=10)
    parallel_execution: bool = True
    
    # Database settings
    database_url: Optional[str] = None
    save_results: bool = True
    
    # Logging settings
    debug: bool = False
    log_level: str = Field(default="INFO")
    
    # Cache settings
    enable_cache: bool = True
    cache_ttl: int = Field(default=3600, ge=0)
    
    # Security settings
    enable_security: bool = True
    max_concurrent_evaluations: int = Field(default=5, ge=1, le=50)
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationConfig':
        """Create config from dictionary."""
        return cls(**data)


class ConfigManager:
    """Manages configuration loading and validation."""

    def __init__(self, config_path: Union[str, Path] = "config.json"):
        self.config_path = Path(config_path)
        self._config_data = {}
        self.logger = logging.getLogger(__name__)
        self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment."""
        config_data = {}
        
        # Load from file if exists
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    config_data = json.load(f)
                self.logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                self.logger.error(f"Failed to load config file: {e}")
        
        # Override with environment variables
        config_data.update(self._load_from_env())
        
        self._config_data = config_data
        return config_data

    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Map environment variables to config keys
        env_mappings = {
            'DEEPEVAL_DATABASE_URL': 'database_url',
            'DEEPEVAL_DEBUG': 'debug',
            'DEEPEVAL_LOG_LEVEL': 'log_level',
            'DEEPEVAL_MAX_WORKERS': 'max_workers',
            'DEEPEVAL_TIMEOUT': 'timeout',
            'DEEPEVAL_PARALLEL_EXECUTION': 'parallel_execution',
            'DEEPEVAL_SAVE_RESULTS': 'save_results',
            'DEEPEVAL_ENABLE_CACHE': 'enable_cache',
            'DEEPEVAL_CACHE_TTL': 'cache_ttl',
            'DEEPEVAL_ENABLE_SECURITY': 'enable_security',
            'DEEPEVAL_MAX_CONCURRENT_EVALUATIONS': 'max_concurrent_evaluations',
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if config_key in ['debug', 'parallel_execution', 'save_results', 'enable_cache', 'enable_security']:
                    env_config[config_key] = value.lower() in ('true', '1', 'yes', 'on')
                elif config_key in ['max_workers', 'timeout', 'cache_ttl', 'max_concurrent_evaluations']:
                    try:
                        env_config[config_key] = int(value)
                    except ValueError:
                        self.logger.warning(f"Invalid integer value for {env_var}: {value}")
                else:
                    env_config[config_key] = value
        
        return env_config

    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration."""
        eval_config_data = self._config_data.get("evaluation", {})
        return EvaluationConfig(**eval_config_data)

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        db_config_data = self._config_data.get("database", {})
        return DatabaseConfig(**db_config_data)

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        log_config_data = self._config_data.get("logging", {})
        return LoggingConfig(**log_config_data)

    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration."""
        cache_config_data = self._config_data.get("cache", {})
        return CacheConfig(**cache_config_data)

    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        security_config_data = self._config_data.get("security", {})
        return SecurityConfig(**security_config_data)

    def save_config(self, config_data: Dict[str, Any]):
        """Save configuration to file."""
        try:
            with open(self.config_path, "w") as f:
                json.dump(config_data, f, indent=2)
            self._config_data = config_data
            self.logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            raise

    def update_config(self, key: str, value: Any):
        """Update a specific configuration value."""
        self._config_data[key] = value
        self.save_config(self._config_data)

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a specific configuration value."""
        return self._config_data.get(key, default)

    def validate_config(self) -> List[str]:
        """Validate the current configuration and return any issues."""
        issues = []
        
        try:
            # Validate evaluation config
            self.get_evaluation_config()
        except Exception as e:
            issues.append(f"Evaluation config validation failed: {e}")
        
        try:
            # Validate database config if URL is provided
            db_config = self.get_database_config()
            if db_config.url and not db_config.url.startswith(('sqlite:///', 'postgresql://', 'mysql://')):
                issues.append("Invalid database URL format")
        except Exception as e:
            issues.append(f"Database config validation failed: {e}")
        
        try:
            # Validate logging config
            log_config = self.get_logging_config()
            if log_config.level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                issues.append("Invalid log level")
        except Exception as e:
            issues.append(f"Logging config validation failed: {e}")
        
        return issues

    def create_default_config(self) -> Dict[str, Any]:
        """Create a default configuration."""
        return {
            "evaluation": {
                "max_workers": 10,
                "timeout": 300,
                "retry_attempts": 3,
                "parallel_execution": True,
                "save_results": True,
                "debug": False,
                "log_level": "INFO",
                "enable_cache": True,
                "cache_ttl": 3600,
                "enable_security": True,
                "max_concurrent_evaluations": 5
            },
            "database": {
                "url": "sqlite:///deepeval.db",
                "echo": False,
                "pool_size": 10,
                "max_overflow": 20,
                "pool_recycle": 3600,
                "pool_pre_ping": True
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_path": None,
                "max_file_size": 10485760,
                "backup_count": 5
            },
            "cache": {
                "cache_type": "memory",
                "redis_url": None,
                "max_size": 1000,
                "ttl": 3600
            },
            "security": {
                "api_key_validation": True,
                "rate_limiting": True,
                "max_requests_per_minute": 100,
                "allowed_origins": ["*"],
                "enable_cors": True
            }
        }

    def initialize_default_config(self):
        """Initialize with default configuration if no config exists."""
        if not self.config_path.exists():
            default_config = self.create_default_config()
            self.save_config(default_config)
            self.logger.info("Initialized with default configuration")


class EnvironmentConfig:
    """Configuration loaded from environment variables only."""

    @staticmethod
    def get_evaluation_config() -> EvaluationConfig:
        """Get evaluation config from environment variables."""
        return EvaluationConfig(
            max_workers=int(os.getenv('DEEPEVAL_MAX_WORKERS', '10')),
            timeout=int(os.getenv('DEEPEVAL_TIMEOUT', '300')),
            retry_attempts=int(os.getenv('DEEPEVAL_RETRY_ATTEMPTS', '3')),
            parallel_execution=os.getenv('DEEPEVAL_PARALLEL_EXECUTION', 'true').lower() == 'true',
            database_url=os.getenv('DEEPEVAL_DATABASE_URL'),
            save_results=os.getenv('DEEPEVAL_SAVE_RESULTS', 'true').lower() == 'true',
            debug=os.getenv('DEEPEVAL_DEBUG', 'false').lower() == 'true',
            log_level=os.getenv('DEEPEVAL_LOG_LEVEL', 'INFO'),
            enable_cache=os.getenv('DEEPEVAL_ENABLE_CACHE', 'true').lower() == 'true',
            cache_ttl=int(os.getenv('DEEPEVAL_CACHE_TTL', '3600')),
            enable_security=os.getenv('DEEPEVAL_ENABLE_SECURITY', 'true').lower() == 'true',
            max_concurrent_evaluations=int(os.getenv('DEEPEVAL_MAX_CONCURRENT_EVALUATIONS', '5'))
        )


# Global configuration manager
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Union[str, Path] = "config.json") -> ConfigManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_evaluation_config() -> EvaluationConfig:
    """Get evaluation configuration from the global config manager."""
    return get_config_manager().get_evaluation_config()


def get_database_config() -> DatabaseConfig:
    """Get database configuration from the global config manager."""
    return get_config_manager().get_database_config()


def get_logging_config() -> LoggingConfig:
    """Get logging configuration from the global config manager."""
    return get_config_manager().get_logging_config()


def get_cache_config() -> CacheConfig:
    """Get cache configuration from the global config manager."""
    return get_config_manager().get_cache_config()


def get_security_config() -> SecurityConfig:
    """Get security configuration from the global config manager."""
    return get_config_manager().get_security_config()
