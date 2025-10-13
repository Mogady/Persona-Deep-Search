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
    RiskCategory
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
    # Repository
    "ResearchRepository",
    "DatabaseError",
]
