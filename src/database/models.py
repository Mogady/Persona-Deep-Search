"""
SQLAlchemy database models for the Deep Research AI Agent.

This module defines the core data models for storing research sessions,
facts, connections, risk flags, and search queries.
"""

from datetime import datetime
from typing import List
from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime,
    ForeignKey, Index, Enum, JSON, Boolean
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class SessionStatus(enum.Enum):
    """Status of a research session."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class FactCategory(enum.Enum):
    """Categories for discovered facts."""
    BIOGRAPHICAL = "biographical"
    PROFESSIONAL = "professional"
    FINANCIAL = "financial"
    BEHAVIORAL = "behavioral"
    EDUCATIONAL = "educational"
    LEGAL = "legal"
    SOCIAL = "social"
    OTHER = "other"


class RiskSeverity(enum.Enum):
    """Severity levels for risk flags."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(enum.Enum):
    """Categories for risk flags."""
    LEGAL = "legal"
    FINANCIAL = "financial"
    REPUTATIONAL = "reputational"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"
    OTHER = "other"


class ResearchSession(Base):
    """
    Represents a complete research investigation session.

    Tracks metadata about each research request including target,
    timing, and aggregate statistics.
    """
    __tablename__ = "research_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    target_name = Column(String(255), nullable=False, index=True)
    research_depth = Column(Integer, default=5)
    status = Column(Enum(SessionStatus), default=SessionStatus.PENDING, nullable=False)

    # Aggregate counts
    total_facts = Column(Integer, default=0)
    total_connections = Column(Integer, default=0)
    total_risks = Column(Integer, default=0)
    total_searches = Column(Integer, default=0)

    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Metadata
    final_report = Column(Text, nullable=True)
    connection_graph = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    facts = relationship("Fact", back_populates="session", cascade="all, delete-orphan")
    connections = relationship("Connection", back_populates="session", cascade="all, delete-orphan")
    risk_flags = relationship("RiskFlag", back_populates="session", cascade="all, delete-orphan")
    search_queries = relationship("SearchQuery", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_session_status', 'status'),
        Index('idx_session_created', 'created_at'),
    )

    def __repr__(self) -> str:
        return (
            f"<ResearchSession(id={self.id}, session_id='{self.session_id}', "
            f"target='{self.target_name}', status={self.status.value})>"
        )


class Fact(Base):
    """
    Individual discovered facts about the research target.

    Each fact includes source attribution, confidence scoring,
    and categorization for organization.
    """
    __tablename__ = "facts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("research_sessions.session_id"), nullable=False, index=True)

    # Fact content
    content = Column(Text, nullable=False)
    source_url = Column(String(500), nullable=False)
    category = Column(Enum(FactCategory), nullable=False)

    # Scoring and verification
    confidence_score = Column(Float, nullable=False)  # 0.0 to 1.0
    is_verified = Column(Boolean, default=False)
    verification_source_count = Column(Integer, default=1)

    # Metadata
    extracted_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Additional context
    extraction_method = Column(String(50), nullable=True)  # e.g., "gpt-4", "claude"
    raw_context = Column(Text, nullable=True)  # Original text where fact was found

    # Relationships
    session = relationship("ResearchSession", back_populates="facts")

    __table_args__ = (
        Index('idx_fact_category', 'category'),
        Index('idx_fact_confidence', 'confidence_score'),
    )

    def __repr__(self) -> str:
        return (
            f"<Fact(id={self.id}, category={self.category.value}, "
            f"confidence={self.confidence_score:.2f}, content='{self.content[:50]}...')>"
        )


class Connection(Base):
    """
    Relationships discovered between entities during research.

    Maps connections between the target and other entities,
    including relationship types and supporting evidence.
    """
    __tablename__ = "connections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("research_sessions.session_id"), nullable=False, index=True)

    # Connection details
    entity_a = Column(String(255), nullable=False, index=True)
    entity_b = Column(String(255), nullable=False, index=True)
    relationship_type = Column(String(100), nullable=False)  # e.g., "board_member", "business_partner"

    # Evidence and scoring
    evidence = Column(JSON, nullable=False)  # List of evidence URLs/descriptions
    confidence_score = Column(Float, nullable=False)  # 0.0 to 1.0

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    session = relationship("ResearchSession", back_populates="connections")

    __table_args__ = (
        Index('idx_connection_entities', 'entity_a', 'entity_b'),
        Index('idx_connection_type', 'relationship_type'),
    )

    def __repr__(self) -> str:
        return (
            f"<Connection(id={self.id}, {self.entity_a} --[{self.relationship_type}]--> "
            f"{self.entity_b}, confidence={self.confidence_score:.2f})>"
        )


class RiskFlag(Base):
    """
    Potential risks or red flags discovered during research.

    Categorized by severity and type, with supporting evidence
    for due diligence assessments.
    """
    __tablename__ = "risk_flags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("research_sessions.session_id"), nullable=False, index=True)

    # Risk details
    severity = Column(Enum(RiskSeverity), nullable=False)
    category = Column(Enum(RiskCategory), nullable=False)
    description = Column(Text, nullable=False)

    # Evidence and scoring
    evidence = Column(JSON, nullable=False)  # List of evidence URLs/descriptions
    confidence_score = Column(Float, nullable=False)  # 0.0 to 1.0

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Additional context
    mitigation_notes = Column(Text, nullable=True)
    is_confirmed = Column(Boolean, default=False)

    # Relationships
    session = relationship("ResearchSession", back_populates="risk_flags")

    __table_args__ = (
        Index('idx_risk_severity', 'severity'),
        Index('idx_risk_category', 'category'),
    )

    def __repr__(self) -> str:
        return (
            f"<RiskFlag(id={self.id}, severity={self.severity.value}, "
            f"category={self.category.value}, description='{self.description[:50]}...')>"
        )


class SearchQuery(Base):
    """
    Individual search queries executed during research.

    Tracks all searches for audit trail and query optimization.
    """
    __tablename__ = "search_queries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("research_sessions.session_id"), nullable=False, index=True)

    # Query details
    query = Column(Text, nullable=False)
    iteration = Column(Integer, nullable=False)  # Which research iteration (1-7)
    search_engine = Column(String(50), default="serpapi")  # serpapi, playwright

    # Results metadata
    results_count = Column(Integer, default=0)
    relevance_score = Column(Float, nullable=True)  # Average relevance of results
    facts_extracted = Column(Integer, default=0)

    # Timing
    executed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    execution_time_ms = Column(Integer, nullable=True)  # Time to execute search

    # Relationships
    session = relationship("ResearchSession", back_populates="search_queries")

    __table_args__ = (
        Index('idx_query_iteration', 'iteration'),
        Index('idx_query_executed', 'executed_at'),
    )

    def __repr__(self) -> str:
        return (
            f"<SearchQuery(id={self.id}, iteration={self.iteration}, "
            f"query='{self.query[:50]}...', results={self.results_count})>"
        )


# Helper function to create all tables
def create_tables(engine):
    """
    Create all database tables.

    Args:
        engine: SQLAlchemy engine instance
    """
    Base.metadata.create_all(engine)


# Helper function to drop all tables (use with caution!)
def drop_tables(engine):
    """
    Drop all database tables.

    Args:
        engine: SQLAlchemy engine instance

    Warning:
        This will delete all data. Use only for testing/development.
    """
    Base.metadata.drop_all(engine)
