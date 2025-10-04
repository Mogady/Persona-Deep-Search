from typing import TypedDict, List, Dict, Set, Optional
from datetime import datetime
import asyncio

class Fact(TypedDict):
    content: str
    source_url: str
    source_domain: str
    extracted_date: datetime
    confidence: float
    category: str  # biographical, professional, financial, behavioral
    entities: Dict[str, List[str]]

class Connection(TypedDict):
    entity_a: str
    entity_b: str
    relationship_type: str
    evidence: List[str]
    confidence: float
    time_period: str

class RiskFlag(TypedDict):
    severity: str  # Low, Medium, High, Critical
    category: str  # Legal, Financial, Reputational, Compliance, Behavioral
    description: str
    evidence: List[str]
    confidence: float
    recommended_follow_up: str

class SearchQuery(TypedDict):
    query: str
    timestamp: datetime
    results_count: int
    relevance_score: float

class ResearchState(TypedDict):
    # Input
    target_name: str
    research_depth: int  # Max iterations (default 7)
    session_id: str

    # Progressive Knowledge Building
    collected_facts: List[Fact]  # Accumulated facts across all iterations
    new_facts: List[Fact]  # Facts extracted in current iteration only (for optimization)
    connections: List[Connection]
    risk_flags: List[RiskFlag]
    search_history: List[SearchQuery]
    facts_before_iteration: int # for checking if new facts were found

    # Control Flow
    current_iteration: int
    next_queries: List[str]
    explored_topics: Set[str]
    raw_search_results: List[Dict]  # Temporary storage between nodes

    # Metadata
    start_time: datetime
    last_update: Optional[datetime]
    stop_event: asyncio.Event

    # Output
    final_report: Optional[str]
    connection_graph: Optional[Dict]
