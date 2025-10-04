"""
Database module.

This module provides database models and repository pattern for data persistence.
"""

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
    create_tables,
    drop_tables,
)

from .repository import (
    ResearchRepository,
    DatabaseError,
)

__all__ = [
    # Models
    "Base",
    "ResearchSession",
    "Fact",
    "Connection",
    "RiskFlag",
    "SearchQuery",
    # Enums
    "SessionStatus",
    "FactCategory",
    "RiskSeverity",
    "RiskCategory",
    # Functions
    "create_tables",
    "drop_tables",
    # Repository
    "ResearchRepository",
    "DatabaseError",
]
