"""
Repository pattern implementation for database operations.

Provides a clean data access layer with transaction management,
batch operations, and error handling.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from contextlib import contextmanager
from sqlalchemy import create_engine, and_, desc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import logging

from .models import (
    Base,
    ResearchSession,
    Fact,
    Connection,
    RiskFlag,
    SearchQuery,
    SessionStatus,
    FactCategory,
    RiskSeverity,
    RiskCategory,
)

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass


class ResearchRepository:
    """
    Repository for managing research data persistence.

    Implements the repository pattern to abstract database operations
    and provide a clean interface for the application.
    """

    def __init__(self, database_url: str, echo: bool = False):
        """
        Initialize the repository with database connection.

        Args:
            database_url: SQLAlchemy database URL
            echo: Whether to echo SQL statements (for debugging)
        """
        self.engine = create_engine(database_url, echo=echo, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        logger.info(f"Database repository initialized with URL: {database_url}")

    def create_tables(self) -> None:
        """Create all database tables."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create tables: {e}")
            raise DatabaseError(f"Table creation failed: {e}")

    def drop_tables(self) -> None:
        """
        Drop all database tables.

        Warning: This will delete all data!
        """
        try:
            Base.metadata.drop_all(self.engine)
            logger.warning("All database tables dropped")
        except SQLAlchemyError as e:
            logger.error(f"Failed to drop tables: {e}")
            raise DatabaseError(f"Table drop failed: {e}")

    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions with automatic commit/rollback.

        Yields:
            Session: SQLAlchemy session

        Example:
            with repo.get_session() as session:
                session.add(obj)
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error, rolling back: {e}")
            raise
        finally:
            session.close()

    # ==================== Research Session Operations ====================

    def create_session(
        self,
        session_id: str,
        target_name: str,
        research_depth: int = 5,
    ) -> ResearchSession:
        """
        Create a new research session.

        Args:
            session_id: Unique session identifier
            target_name: Name of research target
            research_depth: Maximum search iterations

        Returns:
            ResearchSession: Created session object

        Raises:
            DatabaseError: If session creation fails
        """
        try:
            with self.get_session() as session:
                research_session = ResearchSession(
                    session_id=session_id,
                    target_name=target_name,
                    research_depth=research_depth,
                    status=SessionStatus.PENDING,
                    created_at=datetime.now(),
                )
                session.add(research_session)
                session.flush()
                session.refresh(research_session)

                logger.info(f"Created research session: {session_id} for target: {target_name}")
                return research_session

        except SQLAlchemyError as e:
            logger.error(f"Failed to create session: {e}")
            raise DatabaseError(f"Session creation failed: {e}")

    def get_research_session(self, session_id: str) -> Optional[ResearchSession]:
        """
        Get a research session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Optional[ResearchSession]: Session object or None if not found
        """
        try:
            with self.SessionLocal() as session:
                return session.query(ResearchSession).filter_by(session_id=session_id).first()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None

    def update_session_status(
        self,
        session_id: str,
        status: SessionStatus,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Update the status of a research session.

        Args:
            session_id: Session identifier
            status: New status
            error_message: Error message if status is FAILED

        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            with self.get_session() as session:
                research_session = session.query(ResearchSession).filter_by(
                    session_id=session_id
                ).first()

                if not research_session:
                    logger.warning(f"Session not found: {session_id}")
                    return False

                research_session.status = status
                research_session.updated_at = datetime.now()

                if status == SessionStatus.IN_PROGRESS and not research_session.started_at:
                    research_session.started_at = datetime.now()
                elif status == SessionStatus.COMPLETED:
                    research_session.completed_at = datetime.now()

                if error_message:
                    research_session.error_message = error_message

                logger.info(f"Updated session {session_id} status to {status.value}")
                return True

        except SQLAlchemyError as e:
            logger.error(f"Failed to update session status: {e}")
            return False

    def update_session_counts(self, session_id: str) -> bool:
        """
        Update aggregate counts for a session.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if update successful
        """
        try:
            with self.get_session() as session:
                research_session = session.query(ResearchSession).filter_by(
                    session_id=session_id
                ).first()

                if not research_session:
                    return False

                research_session.total_facts = session.query(Fact).filter_by(
                    session_id=session_id
                ).count()
                research_session.total_connections = session.query(Connection).filter_by(
                    session_id=session_id
                ).count()
                research_session.total_risks = session.query(RiskFlag).filter_by(
                    session_id=session_id
                ).count()
                research_session.total_searches = session.query(SearchQuery).filter_by(
                    session_id=session_id
                ).count()

                return True

        except SQLAlchemyError as e:
            logger.error(f"Failed to update session counts: {e}")
            return False

    def save_final_report(
        self,
        session_id: str,
        report: str,
        connection_graph: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save the final research report.

        Args:
            session_id: Session identifier
            report: Final report text
            connection_graph: Connection graph data

        Returns:
            bool: True if save successful
        """
        try:
            with self.get_session() as session:
                research_session = session.query(ResearchSession).filter_by(
                    session_id=session_id
                ).first()

                if not research_session:
                    return False

                research_session.final_report = report
                if connection_graph:
                    research_session.connection_graph = connection_graph
                research_session.updated_at = datetime.now()

                logger.info(f"Saved final report for session {session_id}")
                return True

        except SQLAlchemyError as e:
            logger.error(f"Failed to save final report: {e}")
            return False

    def update_session_checkpoint(
        self,
        session_id: str,
        node_name: str,
        state: Dict[str, Any]
    ) -> bool:
        """
        Save a checkpoint after a node execution.

        Args:
            session_id: Session identifier
            node_name: Name of the node that just completed
            state: Current workflow state

        Returns:
            bool: True if checkpoint saved successfully
        """
        try:
            with self.get_session() as session:
                research_session = session.query(ResearchSession).filter_by(
                    session_id=session_id
                ).first()

                if not research_session:
                    logger.warning(f"Session not found for checkpoint: {session_id}")
                    return False

                # Update status to in_progress if not already
                if research_session.status == SessionStatus.PENDING:
                    research_session.status = SessionStatus.IN_PROGRESS
                    research_session.started_at = datetime.now()

                research_session.updated_at = datetime.now()

                logger.debug(f"Checkpoint saved for session {session_id} after {node_name}")
                return True

        except SQLAlchemyError as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def complete_session(self, session_id: str) -> bool:
        """
        Mark a research session as completed.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if session marked as completed
        """
        return self.update_session_status(session_id, SessionStatus.COMPLETED)

    def fail_session(self, session_id: str, error_message: str) -> bool:
        """
        Mark a research session as failed with error message.

        Args:
            session_id: Session identifier
            error_message: Error description

        Returns:
            bool: True if session marked as failed
        """
        return self.update_session_status(session_id, SessionStatus.FAILED, error_message)

    # ==================== Fact Operations ====================

    def save_fact(
        self,
        session_id: str,
        content: str,
        source_url: str,
        category: FactCategory,
        confidence_score: float,
        extraction_method: Optional[str] = None,
        raw_context: Optional[str] = None,
    ) -> Optional[int]:
        """
        Save a discovered fact.

        Args:
            session_id: Session identifier
            content: Fact content
            source_url: Source URL
            category: Fact category
            confidence_score: Confidence score (0.0-1.0)
            extraction_method: Method used to extract fact
            raw_context: Original context where fact was found

        Returns:
            Optional[int]: ID of created fact or None if failed
        """
        try:
            with self.get_session() as session:
                fact = Fact(
                    session_id=session_id,
                    content=content,
                    source_url=source_url,
                    category=category,
                    confidence_score=confidence_score,
                    extraction_method=extraction_method,
                    raw_context=raw_context,
                    extracted_date=datetime.now(),
                )
                session.add(fact)
                session.flush()
                fact_id = fact.id

                logger.debug(f"Saved fact for session {session_id}: {content[:50]}...")
                return fact_id

        except SQLAlchemyError as e:
            logger.error(f"Failed to save fact: {e}")
            return None

    def save_facts_batch(self, facts: List[Dict[str, Any]]) -> int:
        """
        Save multiple facts in a batch operation.

        Args:
            facts: List of fact dictionaries with required fields

        Returns:
            int: Number of facts saved successfully
        """
        try:
            with self.get_session() as session:
                fact_objects = [
                    Fact(
                        session_id=f["session_id"],
                        content=f["content"],
                        source_url=f["source_url"],
                        category=f["category"],
                        confidence_score=f["confidence_score"],
                        extraction_method=f.get("extraction_method"),
                        raw_context=f.get("raw_context"),
                        extracted_date=datetime.now(),
                    )
                    for f in facts
                ]
                session.bulk_save_objects(fact_objects)

                logger.info(f"Batch saved {len(facts)} facts")
                return len(facts)

        except SQLAlchemyError as e:
            logger.error(f"Failed to batch save facts: {e}")
            return 0

    def get_facts_by_session(
        self,
        session_id: str,
        category: Optional[FactCategory] = None,
        min_confidence: Optional[float] = None,
    ) -> List[Fact]:
        """
        Get facts for a session with optional filtering.

        Args:
            session_id: Session identifier
            category: Optional category filter
            min_confidence: Optional minimum confidence filter

        Returns:
            List[Fact]: List of matching facts
        """
        try:
            with self.SessionLocal() as session:
                query = session.query(Fact).filter_by(session_id=session_id)

                if category:
                    query = query.filter_by(category=category)
                if min_confidence is not None:
                    query = query.filter(Fact.confidence_score >= min_confidence)

                return query.order_by(desc(Fact.confidence_score)).all()

        except SQLAlchemyError as e:
            logger.error(f"Failed to get facts: {e}")
            return []

    # ==================== Connection Operations ====================

    def save_connection(
        self,
        session_id: str,
        entity_a: str,
        entity_b: str,
        relationship_type: str,
        evidence: List[str],
        confidence_score: float,
    ) -> Optional[Connection]:
        """
        Save a discovered connection.

        Args:
            session_id: Session identifier
            entity_a: First entity
            entity_b: Second entity
            relationship_type: Type of relationship
            evidence: List of evidence URLs/descriptions
            confidence_score: Confidence score (0.0-1.0)

        Returns:
            Optional[Connection]: Created connection or None if failed
        """
        try:
            with self.get_session() as session:
                connection = Connection(
                    session_id=session_id,
                    entity_a=entity_a,
                    entity_b=entity_b,
                    relationship_type=relationship_type,
                    evidence=evidence,
                    confidence_score=confidence_score,
                )
                session.add(connection)
                session.flush()
                session.refresh(connection)

                logger.debug(f"Saved connection: {entity_a} -> {entity_b}")
                return connection

        except SQLAlchemyError as e:
            logger.error(f"Failed to save connection: {e}")
            return None

    def save_connections_batch(self, connections: List[Dict[str, Any]]) -> int:
        """
        Save multiple connections in a batch operation.

        Args:
            connections: List of connection dictionaries with required fields

        Returns:
            int: Number of connections saved successfully
        """
        try:
            with self.get_session() as session:
                connection_objects = [
                    Connection(
                        session_id=c["session_id"],
                        entity_a=c["entity_a"],
                        entity_b=c["entity_b"],
                        relationship_type=c["relationship_type"],
                        evidence=c["evidence"],
                        confidence_score=c["confidence"],
                    )
                    for c in connections
                ]
                session.bulk_save_objects(connection_objects)

                logger.info(f"Batch saved {len(connections)} connections")
                return len(connections)

        except SQLAlchemyError as e:
            logger.error(f"Failed to batch save connections: {e}")
            return 0

    def get_connections_by_session(self, session_id: str) -> List[Connection]:
        """
        Get all connections for a session.

        Args:
            session_id: Session identifier

        Returns:
            List[Connection]: List of connections
        """
        try:
            with self.SessionLocal() as session:
                return session.query(Connection).filter_by(
                    session_id=session_id
                ).order_by(desc(Connection.confidence_score)).all()

        except SQLAlchemyError as e:
            logger.error(f"Failed to get connections: {e}")
            return []

    # ==================== Risk Flag Operations ====================

    def save_risk_flag(
        self,
        session_id: str,
        severity: RiskSeverity,
        category: RiskCategory,
        description: str,
        evidence: List[str],
        confidence_score: float,
        mitigation_notes: Optional[str] = None,
    ) -> Optional[RiskFlag]:
        """
        Save a risk flag.

        Args:
            session_id: Session identifier
            severity: Risk severity level
            category: Risk category
            description: Risk description
            evidence: List of evidence URLs/descriptions
            confidence_score: Confidence score (0.0-1.0)
            mitigation_notes: Optional mitigation notes

        Returns:
            Optional[RiskFlag]: Created risk flag or None if failed
        """
        try:
            with self.get_session() as session:
                risk = RiskFlag(
                    session_id=session_id,
                    severity=severity,
                    category=category,
                    description=description,
                    evidence=evidence,
                    confidence_score=confidence_score,
                    mitigation_notes=mitigation_notes,
                )
                session.add(risk)
                session.flush()
                session.refresh(risk)

                logger.debug(f"Saved risk flag: {severity.value} - {description[:50]}...")
                return risk

        except SQLAlchemyError as e:
            logger.error(f"Failed to save risk flag: {e}")
            return None

    def save_risk_flags_batch(self, risks: List[Dict[str, Any]]) -> int:
        """
        Save multiple risk flags in a batch operation.

        Args:
            risks: List of risk dictionaries with required fields

        Returns:
            int: Number of risk flags saved successfully
        """
        try:
            with self.get_session() as session:
                risk_objects = [
                    RiskFlag(
                        session_id=r["session_id"],
                        severity=r["severity"],
                        category=r["category"],
                        description=r["description"],
                        evidence=r["evidence"],
                        confidence_score=r["confidence"],
                        mitigation_notes=r.get("recommended_follow_up"),
                    )
                    for r in risks
                ]
                session.bulk_save_objects(risk_objects)

                logger.info(f"Batch saved {len(risks)} risk flags")
                return len(risks)

        except SQLAlchemyError as e:
            logger.error(f"Failed to batch save risk flags: {e}")
            return 0

    def get_risks_by_session(
        self,
        session_id: str,
        min_severity: Optional[RiskSeverity] = None,
    ) -> List[RiskFlag]:
        """
        Get risk flags for a session.

        Args:
            session_id: Session identifier
            min_severity: Optional minimum severity filter

        Returns:
            List[RiskFlag]: List of risk flags
        """
        try:
            with self.SessionLocal() as session:
                query = session.query(RiskFlag).filter_by(session_id=session_id)

                # Order by severity (critical > high > medium > low)
                severity_order = {
                    RiskSeverity.CRITICAL: 0,
                    RiskSeverity.HIGH: 1,
                    RiskSeverity.MEDIUM: 2,
                    RiskSeverity.LOW: 3,
                }

                risks = query.all()
                return sorted(risks, key=lambda r: severity_order.get(r.severity, 99))

        except SQLAlchemyError as e:
            logger.error(f"Failed to get risks: {e}")
            return []

    # ==================== Search Query Operations ====================

    def save_search_query(
        self,
        session_id: str,
        query: str,
        iteration: int,
        search_engine: str = "serpapi",
        results_count: int = 0,
        relevance_score: Optional[float] = None,
        facts_extracted: int = 0,
        execution_time_ms: Optional[int] = None,
    ) -> Optional[SearchQuery]:
        """
        Save a search query.

        Args:
            session_id: Session identifier
            query: Search query text
            iteration: Search iteration number
            search_engine: Search engine used
            results_count: Number of results returned
            relevance_score: Average relevance score
            facts_extracted: Number of facts extracted
            execution_time_ms: Execution time in milliseconds

        Returns:
            Optional[SearchQuery]: Created search query or None if failed
        """
        try:
            with self.get_session() as session:
                search_query = SearchQuery(
                    session_id=session_id,
                    query=query,
                    iteration=iteration,
                    search_engine=search_engine,
                    results_count=results_count,
                    relevance_score=relevance_score,
                    facts_extracted=facts_extracted,
                    execution_time_ms=execution_time_ms,
                )
                session.add(search_query)
                session.flush()
                session.refresh(search_query)

                logger.debug(f"Saved search query: {query[:50]}...")
                return search_query

        except SQLAlchemyError as e:
            logger.error(f"Failed to save search query: {e}")
            return None

    def get_searches_by_session(self, session_id: str) -> List[SearchQuery]:
        """
        Get all search queries for a session.

        Args:
            session_id: Session identifier

        Returns:
            List[SearchQuery]: List of search queries
        """
        try:
            with self.SessionLocal() as session:
                return session.query(SearchQuery).filter_by(
                    session_id=session_id
                ).order_by(SearchQuery.iteration, SearchQuery.executed_at).all()

        except SQLAlchemyError as e:
            logger.error(f"Failed to get searches: {e}")
            return []

    # ==================== Aggregate Queries ====================

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a research session.

        Args:
            session_id: Session identifier

        Returns:
            Optional[Dict[str, Any]]: Summary dictionary or None if not found
        """
        try:
            with self.SessionLocal() as session:
                research_session = session.query(ResearchSession).filter_by(
                    session_id=session_id
                ).first()

                if not research_session:
                    return None

                return {
                    "session_id": research_session.session_id,
                    "target_name": research_session.target_name,
                    "status": research_session.status.value,
                    "total_facts": research_session.total_facts,
                    "total_connections": research_session.total_connections,
                    "total_risks": research_session.total_risks,
                    "total_searches": research_session.total_searches,
                    "created_at": research_session.created_at.isoformat() if research_session.created_at else None,
                    "started_at": research_session.started_at.isoformat() if research_session.started_at else None,
                    "completed_at": research_session.completed_at.isoformat() if research_session.completed_at else None,
                }

        except SQLAlchemyError as e:
            logger.error(f"Failed to get session summary: {e}")
            return None

    def cleanup_old_sessions(self, days: int = 30) -> int:
        """
        Delete sessions older than specified days.

        Args:
            days: Number of days to retain

        Returns:
            int: Number of sessions deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            with self.get_session() as session:
                deleted = session.query(ResearchSession).filter(
                    ResearchSession.created_at < cutoff_date
                ).delete()

                logger.info(f"Cleaned up {deleted} sessions older than {days} days")
                return deleted

        except SQLAlchemyError as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0
