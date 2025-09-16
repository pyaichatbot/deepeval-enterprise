"""
Database models and ORM setup for the DeepEval framework.

This module contains all database-related functionality including models,
session management, and database utilities.
"""

import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

from sqlalchemy import (
    create_engine, 
    Column, 
    String, 
    Float, 
    DateTime, 
    Text, 
    Integer, 
    Boolean,
    ForeignKey,
    Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine
from contextlib import contextmanager
import logging

# Optional PostgreSQL support
try:
    from sqlalchemy.dialects.postgresql import UUID, JSONB
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

Base = declarative_base()


class EvaluationRun(Base):
    """Database model for evaluation runs."""
    __tablename__ = "evaluation_runs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False)
    test_suite_id = Column(String(255), nullable=False)
    configuration = Column(Text)  # JSON serialized
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    total_tests = Column(Integer, default=0)
    passed_tests = Column(Integer, default=0)
    failed_tests = Column(Integer, default=0)
    average_score = Column(Float)
    results = Column(Text)  # JSON serialized
    error_message = Column(Text)
    created_by = Column(String(255))
    
    # Relationships
    metric_results = relationship("MetricResult", back_populates="evaluation_run", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_evaluation_runs_status', 'status'),
        Index('idx_evaluation_runs_test_suite', 'test_suite_id'),
        Index('idx_evaluation_runs_started_at', 'started_at'),
    )


class MetricResult(Base):
    """Database model for individual metric results."""
    __tablename__ = "metric_results"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    evaluation_run_id = Column(String(36), ForeignKey('evaluation_runs.id'), nullable=False)
    test_case_id = Column(String(255), nullable=False)
    metric_name = Column(String(100), nullable=False)
    score = Column(Float, nullable=False)
    passed = Column(Boolean, nullable=False)
    threshold = Column(Float)
    explanation = Column(Text)
    metric_metadata = Column(Text)  # JSON serialized
    execution_time = Column(Float)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    evaluation_run = relationship("EvaluationRun", back_populates="metric_results")
    
    # Indexes
    __table_args__ = (
        Index('idx_metric_results_run_id', 'evaluation_run_id'),
        Index('idx_metric_results_metric_name', 'metric_name'),
        Index('idx_metric_results_test_case', 'test_case_id'),
        Index('idx_metric_results_created_at', 'created_at'),
    )


class TestSuite(Base):
    """Database model for test suites."""
    __tablename__ = "test_suites"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    test_cases = Column(Text)  # JSON serialized
    tags = Column(Text)  # JSON serialized
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    created_by = Column(String(255))
    version = Column(String(50), default="1.0")
    
    # Indexes
    __table_args__ = (
        Index('idx_test_suites_name', 'name'),
        Index('idx_test_suites_created_at', 'created_at'),
    )


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, database_url: str, echo: bool = False):
        self.database_url = database_url
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self.logger = logging.getLogger(__name__)
        self._setup_database(echo)
    
    def _setup_database(self, echo: bool = False):
        """Setup database engine and session factory."""
        try:
            # Configure connection pooling
            engine_kwargs = {
                "echo": echo,
                "poolclass": QueuePool,
                "pool_size": 10,
                "max_overflow": 20,
                "pool_pre_ping": True,
                "pool_recycle": 3600,  # Recycle connections every hour
            }
            
            # Add PostgreSQL-specific settings if available
            if POSTGRES_AVAILABLE and self.database_url.startswith("postgresql"):
                engine_kwargs["json_serializer"] = json.dumps
                engine_kwargs["json_deserializer"] = json.loads
            
            self.engine = create_engine(self.database_url, **engine_kwargs)
            self.SessionLocal = sessionmaker(bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(self.engine)
            
            self.logger.info(f"Database setup completed for: {self.database_url}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup database: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """Get a database session with proper cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_session_sync(self) -> Session:
        """Get a synchronous database session."""
        return self.SessionLocal()
    
    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Database connections closed")


class DatabaseRepository:
    """Repository pattern for database operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    def save_evaluation_run(self, eval_run_data: Dict[str, Any]) -> str:
        """Save evaluation run to database."""
        with self.db_manager.get_session() as session:
            eval_run = EvaluationRun(
                id=eval_run_data.get("id"),
                name=eval_run_data["name"],
                status=eval_run_data["status"],
                test_suite_id=eval_run_data["test_suite_id"],
                configuration=json.dumps(eval_run_data.get("configuration", {})),
                started_at=eval_run_data.get("started_at"),
                completed_at=eval_run_data.get("completed_at"),
                total_tests=eval_run_data.get("total_tests", 0),
                passed_tests=eval_run_data.get("passed_tests", 0),
                failed_tests=eval_run_data.get("failed_tests", 0),
                average_score=eval_run_data.get("average_score"),
                results=json.dumps(eval_run_data.get("results", []), default=str),
                error_message=eval_run_data.get("error_message"),
                created_by=eval_run_data.get("created_by")
            )
            session.add(eval_run)
            session.flush()  # Get the ID
            return eval_run.id
    
    def update_evaluation_run(self, run_id: str, updates: Dict[str, Any]) -> bool:
        """Update evaluation run."""
        with self.db_manager.get_session() as session:
            eval_run = session.query(EvaluationRun).filter_by(id=run_id).first()
            if not eval_run:
                return False
            
            for key, value in updates.items():
                if hasattr(eval_run, key):
                    if key in ["configuration", "results"] and isinstance(value, (dict, list)):
                        setattr(eval_run, key, json.dumps(value, default=str))
                    else:
                        setattr(eval_run, key, value)
            
            return True
    
    def get_evaluation_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get evaluation run by ID."""
        with self.db_manager.get_session() as session:
            eval_run = session.query(EvaluationRun).filter_by(id=run_id).first()
            if not eval_run:
                return None
            
            return {
                "id": eval_run.id,
                "name": eval_run.name,
                "status": eval_run.status,
                "test_suite_id": eval_run.test_suite_id,
                "configuration": json.loads(eval_run.configuration) if eval_run.configuration else {},
                "started_at": eval_run.started_at,
                "completed_at": eval_run.completed_at,
                "total_tests": eval_run.total_tests,
                "passed_tests": eval_run.passed_tests,
                "failed_tests": eval_run.failed_tests,
                "average_score": eval_run.average_score,
                "results": json.loads(eval_run.results) if eval_run.results else [],
                "error_message": eval_run.error_message,
                "created_by": eval_run.created_by
            }
    
    def save_metric_result(self, result_data: Dict[str, Any]) -> str:
        """Save metric result to database."""
        with self.db_manager.get_session() as session:
            metric_result = MetricResult(
                evaluation_run_id=result_data["evaluation_run_id"],
                test_case_id=result_data["test_case_id"],
                metric_name=result_data["metric_name"],
                score=result_data["score"],
                passed=result_data["passed"],
                threshold=result_data.get("threshold"),
                explanation=result_data.get("explanation"),
                metric_metadata=json.dumps(result_data.get("metadata", {})),
                execution_time=result_data.get("execution_time")
            )
            session.add(metric_result)
            session.flush()
            return metric_result.id
    
    def get_metric_results(self, evaluation_run_id: str) -> List[Dict[str, Any]]:
        """Get all metric results for an evaluation run."""
        with self.db_manager.get_session() as session:
            results = session.query(MetricResult).filter_by(
                evaluation_run_id=evaluation_run_id
            ).all()
            
            return [
                {
                    "id": result.id,
                    "evaluation_run_id": result.evaluation_run_id,
                    "test_case_id": result.test_case_id,
                    "metric_name": result.metric_name,
                    "score": result.score,
                    "passed": result.passed,
                    "threshold": result.threshold,
                    "explanation": result.explanation,
                    "metadata": json.loads(result.metric_metadata) if result.metric_metadata else {},
                    "execution_time": result.execution_time,
                    "created_at": result.created_at
                }
                for result in results
            ]
    
    def save_test_suite(self, test_suite_data: Dict[str, Any]) -> str:
        """Save test suite to database."""
        with self.db_manager.get_session() as session:
            test_suite = TestSuite(
                id=test_suite_data.get("id"),
                name=test_suite_data["name"],
                description=test_suite_data.get("description"),
                test_cases=json.dumps(test_suite_data.get("test_cases", [])),
                tags=json.dumps(test_suite_data.get("tags", [])),
                created_by=test_suite_data.get("created_by"),
                version=test_suite_data.get("version", "1.0")
            )
            session.add(test_suite)
            session.flush()
            return test_suite.id
    
    def get_test_suite(self, suite_id: str) -> Optional[Dict[str, Any]]:
        """Get test suite by ID."""
        with self.db_manager.get_session() as session:
            test_suite = session.query(TestSuite).filter_by(id=suite_id).first()
            if not test_suite:
                return None
            
            return {
                "id": test_suite.id,
                "name": test_suite.name,
                "description": test_suite.description,
                "test_cases": json.loads(test_suite.test_cases) if test_suite.test_cases else [],
                "tags": json.loads(test_suite.tags) if test_suite.tags else [],
                "created_at": test_suite.created_at,
                "created_by": test_suite.created_by,
                "version": test_suite.version
            }
    
    def list_evaluation_runs(
        self, 
        limit: int = 100, 
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List evaluation runs with pagination."""
        with self.db_manager.get_session() as session:
            query = session.query(EvaluationRun)
            
            if status:
                query = query.filter_by(status=status)
            
            runs = query.order_by(EvaluationRun.started_at.desc()).offset(offset).limit(limit).all()
            
            return [
                {
                    "id": run.id,
                    "name": run.name,
                    "status": run.status,
                    "test_suite_id": run.test_suite_id,
                    "started_at": run.started_at,
                    "completed_at": run.completed_at,
                    "total_tests": run.total_tests,
                    "passed_tests": run.passed_tests,
                    "failed_tests": run.failed_tests,
                    "average_score": run.average_score,
                    "created_by": run.created_by
                }
                for run in runs
            ]
    
    def cleanup_old_runs(self, days_old: int = 30) -> int:
        """Clean up old evaluation runs."""
        cutoff_date = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=days_old)
        
        with self.db_manager.get_session() as session:
            old_runs = session.query(EvaluationRun).filter(
                EvaluationRun.started_at < cutoff_date
            ).all()
            
            count = len(old_runs)
            for run in old_runs:
                session.delete(run)
            
            return count


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def initialize_database(database_url: str, echo: bool = False) -> DatabaseManager:
    """Initialize the global database manager."""
    global _db_manager
    _db_manager = DatabaseManager(database_url, echo)
    return _db_manager


def get_database_manager() -> Optional[DatabaseManager]:
    """Get the global database manager."""
    return _db_manager


def get_repository() -> Optional[DatabaseRepository]:
    """Get a database repository instance."""
    if _db_manager:
        return DatabaseRepository(_db_manager)
    return None
