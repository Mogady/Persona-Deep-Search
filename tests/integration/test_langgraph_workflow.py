import pytest
from src.agents.graph import ResearchWorkflow
from src.database.repository import ResearchRepository
from src.utils.config import Config

@pytest.fixture
def workflow():
    """Create workflow instance with test repository and config."""
    config = Config.from_env()
    repository = ResearchRepository(config.database.database_url)
    return ResearchWorkflow(repository, config)

def test_workflow_creation(workflow):
    """Test that workflow is created and compiled."""
    assert workflow.graph is not None
    assert workflow.planner is not None
    assert len(workflow.graph.nodes) == 8

def test_should_continue_logic(workflow):
    """Test conditional logic for iterations."""
    # Continue: Not at depth and new facts found
    state = {"current_iteration": 3, "research_depth": 7, "facts_before_iteration": 5, "collected_facts": ["fact"] * 10}
    assert workflow._should_continue(state) == "continue"

    # Report: Max iterations reached
    state = {"current_iteration": 7, "research_depth": 7, "facts_before_iteration": 5, "collected_facts": ["fact"] * 10}
    assert workflow._should_continue(state) == "report"

    # Report: No new facts found
    state = {"current_iteration": 4, "research_depth": 7, "facts_before_iteration": 10, "collected_facts": ["fact"] * 10}
    assert workflow._should_continue(state) == "report"
    
    # Report: No facts found after 3 iterations
    state = {"current_iteration": 4, "research_depth": 7, "facts_before_iteration": 0, "collected_facts": []}
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
    assert len(result["final_report"]) > 100 # Report should have substantial content
