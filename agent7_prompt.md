# 🤖 AGENT 7: LangGraph Orchestration Specialist - Your Assignment

## 📋 Context & Current Status

You are Agent 7 in the Deep Research AI Agent project. Your predecessors have completed:

✅ **Agent 1**: Database models, config, logger, infrastructure
✅ **Agent 2**: AI model clients (GeminiClient, AnthropicClient)
✅ **Agent 3**: Search tools (SerpApiSearch, SearchOrchestrator)
✅ **Agent 4**: Query Planner & Search Executor nodes with semantic deduplication
✅ **Agent 5**: Content Extractor & Validator nodes with confidence calibration
✅ **Agent 6**: Risk Analyzer, Connection Mapper & Report Generator nodes

**You are NOT blocked. All 7 agent nodes are ready and tested.**

---

## 🎯 Your Mission

Build the LangGraph workflow that orchestrates all 7 agent nodes into a cohesive, intelligent research system. You're the conductor—all the instruments exist, now make them play together.

---

## 🏗️ What You Need to Build

### File Structure
```
src/agents/
├── state.py              # ResearchState TypedDict schema
├── graph.py              # LangGraph workflow orchestration
└── __init__.py           # Export graph for UI

tests/integration/
└── test_langgraph_workflow.py  # End-to-end workflow tests
```

---

## 📝 Detailed Specifications

### 1. `src/agents/state.py`

**ResearchState TypedDict:**

```python
from typing import TypedDict, List, Dict, Set, Optional
from datetime import datetime

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
    collected_facts: List[Fact]
    connections: List[Connection]
    risk_flags: List[RiskFlag]
    search_history: List[SearchQuery]

    # Control Flow
    current_iteration: int
    next_queries: List[str]
    explored_topics: Set[str]
    raw_search_results: List[Dict]  # Temporary storage between nodes

    # Metadata
    start_time: datetime
    last_update: Optional[datetime]

    # Output
    final_report: Optional[str]
    connection_graph: Optional[Dict]
```

**Key Requirements:**
- Use proper type hints (TypedDict, List, Dict, Set, Optional)
- All fields from CLAUDE.md state schema
- Import from typing and datetime

---

### 2. `src/agents/graph.py`

**LangGraph Workflow:**

```python
from typing import Dict, Literal
from langgraph.graph import StateGraph, END
from src.agents.state import ResearchState
from src.agents.nodes import (
    QueryPlannerNode,
    SearchExecutorNode,
    ContentExtractorNode,
    ValidatorNode,
    RiskAnalyzerNode,
    ConnectionMapperNode,
    ReportGeneratorNode
)
from src.database.repository import ResearchRepository
from src.utils.logger import get_logger
from datetime import datetime
import uuid

class ResearchWorkflow:
    """
    LangGraph workflow orchestration for deep research agent.
    Coordinates 7 agent nodes with conditional looping.
    """

    def __init__(self, repository: ResearchRepository):
        self.repository = repository
        self.logger = get_logger(__name__)

        # Initialize all nodes
        self.planner = QueryPlannerNode()
        self.searcher = SearchExecutorNode()
        self.extractor = ContentExtractorNode()
        self.validator = ValidatorNode()
        self.risk_analyzer = RiskAnalyzerNode()
        self.connection_mapper = ConnectionMapperNode()
        self.reporter = ReportGeneratorNode()

        # Build and compile graph
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """
        Build LangGraph workflow with all nodes and conditional edges.

        Node Flow:
        START → Planner → Searcher → Extractor → Validator →
        Risk Analyzer → Connection Mapper → [Decision] → Report → END

        Decision Logic:
        - If current_iteration < research_depth AND new facts found → Loop back to Planner
        - Else → Proceed to Report Generator
        """
        workflow = StateGraph(ResearchState)

        # Add all nodes
        workflow.add_node("planner", self._run_planner)
        workflow.add_node("searcher", self._run_searcher)
        workflow.add_node("extractor", self._run_extractor)
        workflow.add_node("validator", self._run_validator)
        workflow.add_node("risk_analyzer", self._run_risk_analyzer)
        workflow.add_node("connection_mapper", self._run_connection_mapper)
        workflow.add_node("reporter", self._run_reporter)

        # Define edges
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "searcher")
        workflow.add_edge("searcher", "extractor")
        workflow.add_edge("extractor", "validator")
        workflow.add_edge("validator", "risk_analyzer")
        workflow.add_edge("risk_analyzer", "connection_mapper")

        # Conditional edge: Continue or Report?
        workflow.add_conditional_edges(
            "connection_mapper",
            self._should_continue,
            {
                "continue": "planner",  # Loop back
                "report": "reporter"    # Finish
            }
        )

        workflow.add_edge("reporter", END)

        return workflow.compile()

    def _should_continue(self, state: ResearchState) -> Literal["continue", "report"]:
        """
        Conditional logic: Continue research or generate report?

        Continue if:
        1. current_iteration < research_depth (max iterations)
        2. New facts were discovered in last iteration

        Otherwise, generate report.
        """
        current_iteration = state.get("current_iteration", 1)
        research_depth = state.get("research_depth", 7)

        # Check if max iterations reached
        if current_iteration > research_depth:
            self.logger.info(f"Max iterations ({research_depth}) reached. Generating report.")
            return "report"

        # Check if new facts were found (compare iteration count)
        facts_count = len(state.get("collected_facts", []))

        # If we're on iteration 1, always continue
        if current_iteration == 1:
            return "continue"

        # If no facts found yet and we're past iteration 3, stop
        if facts_count == 0 and current_iteration > 3:
            self.logger.warning("No facts discovered after 3 iterations. Stopping.")
            return "report"

        # Continue if we haven't hit max iterations
        if current_iteration <= research_depth:
            return "continue"

        return "report"

    def _run_planner(self, state: ResearchState) -> ResearchState:
        """Execute Query Planner node with state persistence."""
        self.logger.info(f"Running Planner - Iteration {state.get('current_iteration', 1)}")
        try:
            state = self.planner.execute(state)
            self._save_checkpoint(state, "planner")
            return state
        except Exception as e:
            self.logger.error(f"Planner node failed: {e}")
            # Add fallback queries
            if "next_queries" not in state:
                state["next_queries"] = [f'"{state["target_name"]}" information']
            return state

    def _run_searcher(self, state: ResearchState) -> ResearchState:
        """Execute Search Executor node with state persistence."""
        self.logger.info(f"Running Searcher - {len(state.get('next_queries', []))} queries")
        try:
            state = self.searcher.execute(state)
            self._save_checkpoint(state, "searcher")
            return state
        except Exception as e:
            self.logger.error(f"Searcher node failed: {e}")
            state["raw_search_results"] = []
            return state

    def _run_extractor(self, state: ResearchState) -> ResearchState:
        """Execute Content Extractor node with state persistence."""
        self.logger.info(f"Running Extractor - {len(state.get('raw_search_results', []))} results")
        try:
            state = self.extractor.execute(state)
            self._save_checkpoint(state, "extractor")
            return state
        except Exception as e:
            self.logger.error(f"Extractor node failed: {e}")
            return state

    def _run_validator(self, state: ResearchState) -> ResearchState:
        """Execute Validator node with state persistence."""
        self.logger.info(f"Running Validator - {len(state.get('collected_facts', []))} facts")
        try:
            state = self.validator.execute(state)
            self._save_checkpoint(state, "validator")
            return state
        except Exception as e:
            self.logger.error(f"Validator node failed: {e}")
            return state

    def _run_risk_analyzer(self, state: ResearchState) -> ResearchState:
        """Execute Risk Analyzer node with state persistence."""
        self.logger.info(f"Running Risk Analyzer")
        try:
            state = self.risk_analyzer.execute(state)
            self._save_checkpoint(state, "risk_analyzer")
            return state
        except Exception as e:
            self.logger.error(f"Risk Analyzer node failed: {e}")
            if "risk_flags" not in state:
                state["risk_flags"] = []
            return state

    def _run_connection_mapper(self, state: ResearchState) -> ResearchState:
        """Execute Connection Mapper node with state persistence."""
        self.logger.info(f"Running Connection Mapper")
        try:
            state = self.connection_mapper.execute(state)
            self._save_checkpoint(state, "connection_mapper")
            return state
        except Exception as e:
            self.logger.error(f"Connection Mapper node failed: {e}")
            if "connections" not in state:
                state["connections"] = []
            return state

    def _run_reporter(self, state: ResearchState) -> ResearchState:
        """Execute Report Generator node with state persistence."""
        self.logger.info(f"Running Reporter - Final report generation")
        try:
            state = self.reporter.execute(state)
            self._save_checkpoint(state, "reporter")
            return state
        except Exception as e:
            self.logger.error(f"Reporter node failed: {e}")
            state["final_report"] = "Error: Failed to generate report."
            return state

    def _save_checkpoint(self, state: ResearchState, node_name: str):
        """Persist state to database after each node."""
        try:
            session_id = state.get("session_id")
            if not session_id:
                return

            self.repository.update_session_checkpoint(
                session_id=session_id,
                node_name=node_name,
                state=state
            )
            self.logger.debug(f"Checkpoint saved after {node_name}")
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")

    def run_research(
        self,
        target_name: str,
        research_depth: int = 7,
        session_id: Optional[str] = None
    ) -> ResearchState:
        """
        Execute full research workflow for a target.

        Args:
            target_name: Name of person/entity to research
            research_depth: Max iterations (default 7)
            session_id: Optional session ID (generates new if not provided)

        Returns:
            Final ResearchState with report and all findings
        """
        # Initialize state
        if not session_id:
            session_id = str(uuid.uuid4())

        initial_state: ResearchState = {
            "target_name": target_name,
            "research_depth": research_depth,
            "session_id": session_id,
            "collected_facts": [],
            "connections": [],
            "risk_flags": [],
            "search_history": [],
            "current_iteration": 1,
            "next_queries": [],
            "explored_topics": set(),
            "raw_search_results": [],
            "start_time": datetime.utcnow(),
            "last_update": None,
            "final_report": None,
            "connection_graph": None
        }

        # Create database session
        self.repository.create_session(
            session_id=session_id,
            target_name=target_name,
            research_depth=research_depth
        )

        self.logger.info(f"Starting research for: {target_name} (session: {session_id})")

        try:
            # Execute workflow
            final_state = self.graph.invoke(initial_state)

            self.logger.info(f"Research complete for {target_name}")
            self.logger.info(f"Facts: {len(final_state.get('collected_facts', []))}")
            self.logger.info(f"Risks: {len(final_state.get('risk_flags', []))}")
            self.logger.info(f"Connections: {len(final_state.get('connections', []))}")

            # Mark session as complete
            self.repository.complete_session(session_id)

            return final_state

        except Exception as e:
            self.logger.error(f"Research workflow failed: {e}")
            self.repository.fail_session(session_id, str(e))
            raise
```

**Key Features:**
- All 7 nodes integrated
- Conditional looping based on iteration count and fact discovery
- State persistence after each node
- Comprehensive error handling (graceful degradation)
- Logging at each stage
- Database integration

---

### 3. `src/agents/__init__.py`

Export the workflow for UI:

```python
from src.agents.graph import ResearchWorkflow
from src.agents.state import ResearchState

__all__ = ["ResearchWorkflow", "ResearchState"]
```

---

### 4. Testing: `tests/integration/test_langgraph_workflow.py`

```python
import pytest
from src.agents.graph import ResearchWorkflow
from src.database.repository import ResearchRepository
from src.utils.config import Config

@pytest.fixture
def workflow():
    """Create workflow instance with test repository."""
    config = Config.from_env()
    repository = ResearchRepository(config.database.connection_string)
    return ResearchWorkflow(repository)

def test_workflow_creation(workflow):
    """Test that workflow is created and compiled."""
    assert workflow.graph is not None
    assert workflow.planner is not None

def test_should_continue_logic(workflow):
    """Test conditional logic for iterations."""
    # Max iterations not reached
    state = {"current_iteration": 3, "research_depth": 7, "collected_facts": [{"content": "test"}]}
    assert workflow._should_continue(state) == "continue"

    # Max iterations reached
    state = {"current_iteration": 8, "research_depth": 7, "collected_facts": []}
    assert workflow._should_continue(state) == "report"

@pytest.mark.integration
def test_full_workflow_execution(workflow):
    """
    Test end-to-end workflow with a real target.
    This test requires API keys and will make actual API calls.
    """
    # Use a well-known public figure
    target_name = "Satya Nadella"

    # Run with limited depth for testing
    result = workflow.run_research(target_name, research_depth=2)

    # Assertions
    assert result["target_name"] == target_name
    assert len(result["collected_facts"]) > 0, "Should discover facts"
    assert result["final_report"] is not None, "Should generate report"
    assert result["current_iteration"] >= 2, "Should run at least 2 iterations"
```

---

## 🔧 Environment & Dependencies

### Virtual Environment
```bash
source .venv/bin/activate
```

### Imports You'll Need
```python
# LangGraph
from langgraph.graph import StateGraph, END

# Agent nodes (already implemented)
from src.agents.nodes import (
    QueryPlannerNode,
    SearchExecutorNode,
    ContentExtractorNode,
    ValidatorNode,
    RiskAnalyzerNode,
    ConnectionMapperNode,
    ReportGeneratorNode
)

# Database
from src.database.repository import ResearchRepository

# Utils
from src.utils.logger import get_logger
from src.utils.config import Config

# Standard library
from typing import Dict, Literal, Optional
from datetime import datetime
import uuid
```

---

## ✅ Testing Requirements

### Integration Tests

**Test Coverage:**
1. Graph creation and compilation
2. Conditional logic (`_should_continue`)
3. Node execution sequence
4. State persistence
5. Error handling (mock node failures)
6. Full end-to-end workflow with real target

**Run Tests:**
```bash
source .venv/bin/activate
pytest tests/integration/test_langgraph_workflow.py -v
```

---

## 📊 Success Criteria

Your work is complete when:

- ✅ ResearchState TypedDict fully defined with all fields
- ✅ LangGraph workflow builds and compiles without errors
- ✅ All 7 nodes connected in correct sequence
- ✅ Conditional edge works (loops correctly)
- ✅ State persists to database after each node
- ✅ Error handling prevents cascading failures
- ✅ Integration tests pass (5+ tests)
- ✅ End-to-end test discovers facts and generates report
- ✅ Logging shows clear progress through workflow

**Target Metrics:**
- Workflow completes for known public figure
- Discovers >10 facts in 2-3 iterations
- Generates complete report with all sections
- No crashes or unhandled exceptions

---

## 🚀 Development Approach

### Step 1: Define State Schema (30 min)
- Create `state.py` with all TypedDict definitions
- Use CLAUDE.md as reference for fields
- Add proper type hints

### Step 2: Build Graph Structure (60 min)
- Create `graph.py` skeleton
- Initialize all nodes
- Add nodes to StateGraph
- Define edges (linear + conditional)
- Implement `_should_continue` logic

### Step 3: Add Node Wrappers (45 min)
- Implement `_run_planner`, `_run_searcher`, etc.
- Add error handling to each wrapper
- Add logging
- Add state persistence calls

### Step 4: Testing (60 min)
- Write integration tests
- Test conditional logic
- Test end-to-end with real target
- Verify database persistence

**Total estimated time: 3-4 hours**

---

## 📞 When You're Done

Update `PROGRESS_UPDATE.md`:
- Mark Agent 7 as ✅ COMPLETED
- Add file paths and line counts
- Document any decisions

Report completion with:
- Files created and line counts
- Test results (how many passing)
- Screenshot/log of successful end-to-end run
- Metrics (facts discovered, iterations, time)

---

## 🎯 Key Points

1. **All nodes are ready** - Just orchestrate them
2. **Use TypedDict** - Proper type hints for state
3. **Conditional looping** - Based on iteration and fact count
4. **State persistence** - Save after each node
5. **Error handling** - Graceful degradation, don't crash
6. **Testing** - End-to-end test is critical

---

Good luck, Agent 7! You're tying everything together into a working system. This is where it all comes alive! 🚀
