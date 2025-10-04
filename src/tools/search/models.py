"""
Data models for search results.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class SearchResult:
    """
    Structured search result from any search engine.

    Attributes:
        title: Result title
        url: Result URL
        content: Snippet/description
        score: Relevance score (lower is better, 0 = most relevant)
        source_domain: Domain name (e.g., "example.com")
        published_date: Publication date if available
        search_engine: Source engine ("serpapi", "brave", etc.)
        metadata: Additional engine-specific data
    """
    title: str
    url: str
    content: str
    score: float
    source_domain: str
    published_date: Optional[datetime] = None
    search_engine: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"<SearchResult(title='{self.title[:50]}...', "
            f"url='{self.url}', engine={self.search_engine})>"
        )
