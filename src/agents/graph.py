from typing import Dict, Literal, Optional
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
from src.utils.config import Config
from datetime import datetime
import uuid
import asyncio

class ResearchWorkflow:
    """
    LangGraph workflow orchestration.
    Coordinates 7 nodes with conditional looping.
    """

    def __init__(self, repository: ResearchRepository, config: Config):
        """
        Initialize the research workflow.

        Args:
            repository: Database repository for persistence
            config: Configuration object with all settings
        """
        self.repository = repository
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize all nodes with config and repository
        self.planner = QueryPlannerNode(config, repository)
        self.searcher = SearchExecutorNode(config, repository)
        self.extractor = ContentExtractorNode(config, repository)
        self.validator = ValidatorNode(config, repository)
        self.risk_analyzer = RiskAnalyzerNode(config, repository)
        self.connection_mapper = ConnectionMapperNode(config, repository)
        self.reporter = ReportGeneratorNode(config, repository)

        self.logger.debug("Initialized ResearchWorkflow with all 7 nodes (config + repository enabled)")

        # Build and compile graph
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """
        Build LangGraph workflow with all nodes and conditional edges.
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
                "continue": "planner",
                "report": "reporter"
            }
        )

        workflow.add_edge("reporter", END)

        return workflow.compile()

    def _should_continue(self, state: ResearchState) -> Literal["continue", "report"]:
        """
        Conditional logic: Continue research or generate report?
        """
        # 1. Check for the stop signal first
        stop_event = state.get("stop_event")
        if stop_event and stop_event.is_set():
            self.logger.warning("Stop event received. Terminating research workflow.")
            return "report" # End the loop
        current_iteration = state.get("current_iteration", 1)
        research_depth = state.get("research_depth", 7)
        facts_before = state.get("facts_before_iteration", 0)
        facts_after = len(state.get("collected_facts", []))

        if current_iteration >= research_depth:
            self.logger.info(f"Max iterations ({research_depth}) reached. Generating report.")
            return "report"

        if current_iteration > 1 and facts_after == facts_before:
            self.logger.info("No new facts found in the last iteration. Generating report.")
            return "report"
            
        if facts_after == 0 and current_iteration > 3:
            self.logger.warning("No facts discovered after 3 iterations. Stopping.")
            return "report"

        return "continue"

    def _run_node(self, node_executor, state: ResearchState, node_name: str) -> ResearchState:
        self.logger.debug(f"Running {node_name} - Iteration {state.get('current_iteration', 1)}")
        try:
            state = node_executor(state)
            self._save_checkpoint(state, node_name)
            return state
        except Exception as e:
            self.logger.error(f"{node_name} node failed: {e}", exc_info=True)
            # To prevent crashing the whole workflow, return the state as is.
            # Specific error handling per node can be added here if needed.
            return state

    def _run_planner(self, state: ResearchState) -> ResearchState:
        state['current_iteration'] = state.get('current_iteration', 0) + 1
        state['facts_before_iteration'] = len(state.get('collected_facts', []))
        return self._run_node(self.planner.execute, state, "Planner")

    def _run_searcher(self, state: ResearchState) -> ResearchState:
        return self._run_node(self.searcher.execute, state, "Searcher")

    def _run_extractor(self, state: ResearchState) -> ResearchState:
        return self._run_node(self.extractor.execute, state, "Extractor")

    def _run_validator(self, state: ResearchState) -> ResearchState:
        return self._run_node(self.validator.execute, state, "Validator")

    def _run_risk_analyzer(self, state: ResearchState) -> ResearchState:
        return self._run_node(self.risk_analyzer.execute, state, "Risk Analyzer")

    def _run_connection_mapper(self, state: ResearchState) -> ResearchState:
        return self._run_node(self.connection_mapper.execute, state, "Connection Mapper")

    def _run_reporter(self, state: ResearchState) -> ResearchState:
        return self._run_node(self.reporter.execute, state, "Reporter")


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
        session_id: Optional[str] = None,
        stop_event: Optional[asyncio.Event] = None,

    ) -> ResearchState:
        """
        Execute full research workflow for a target.
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        if stop_event is None:
            stop_event = asyncio.Event()

        initial_state: ResearchState = {
            "target_name": target_name,
            "research_depth": research_depth,
            "session_id": session_id,
            "collected_facts": [],
            "new_facts": [],  # Initialize new_facts for iteration optimization
            "connections": [],
            "risk_flags": [],
            "search_history": [],
            "facts_before_iteration": 0,
            "current_iteration": 0,
            "next_queries": [],
            "explored_topics": set(),
            "raw_search_results": [],
            "start_time": datetime.now(),
            "last_update": None,
            "final_report": None,
            "connection_graph": None
        }

        self.repository.create_session(
            session_id=session_id,
            target_name=target_name,
            research_depth=research_depth
        )

        self.logger.info(f"Starting research for: {target_name} (session: {session_id})")

        try:
            final_state = self.graph.invoke(initial_state ,{"recursion_limit": 100})

            self.logger.info(f"Research complete for {target_name}")
            self.logger.info(f"Facts: {len(final_state.get('collected_facts', []))}")
            self.logger.info(f"Risks: {len(final_state.get('risk_flags', []))}")
            self.logger.info(f"Connections: {len(final_state.get('connections', []))}")

            self.repository.complete_session(session_id)

            return final_state

        except Exception as e:
            self.logger.error(f"Research workflow failed: {e}", exc_info=True)
            self.repository.fail_session(session_id, str(e))
            raise
