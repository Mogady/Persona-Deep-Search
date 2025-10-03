
import pytest
from src.agents.graph import ResearchWorkflow
from src.database.repository import ResearchRepository
from src.utils.config import Config
from unittest.mock import patch

@pytest.fixture(scope="module")
def test_db():
    """Fixture to set up and tear down a test database."""
    repo = ResearchRepository("sqlite:///:memory:")
    repo.create_tables()
    yield repo
    repo.drop_tables()

@pytest.fixture
def workflow(test_db):
    """Create a workflow instance with the test database."""
    config = Config.from_env()
    return ResearchWorkflow(test_db, config)

# Mock the AI calls to avoid external dependencies and costs
@patch('src.tools.models.gemini_client.GeminiClient.generate')
@patch('src.tools.models.anthropic_client.AnthropicClient.generate')
@patch('src.tools.search.SearchOrchestrator.search')
def test_facts_persisted_after_validation(mock_search, mock_anthropic, mock_gemini, workflow, test_db):
    """Test that facts are saved to the database after a workflow run."""
    
    # Mock the external calls to return some plausible data
    mock_search.return_value = [{"title": "Test Search Result", "url": "http://example.com", "content": "This is a test search result about Test Target."}]
    mock_gemini.side_effect = [
        # Planner response
        '["who is Test Target?"]',
        # Extractor response
        '[{"content": "Test Target is a test entity.", "source_url": "http://example.com", "category": "professional", "confidence": 0.9}]',
        # Other Gemini calls
        '{"people": ["Test Target"], "companies": [], "locations": []}'
    ]
    mock_anthropic.return_value = '{"contradictions": []}' # Validator response

    # Run the workflow
    result = workflow.run_research("Test Target", research_depth=1)
    session_id = result.get("session_id")

    assert session_id is not None, "Workflow should return a session_id"

    # Check the database directly
    facts = test_db.get_facts_by_session(session_id)

    assert len(facts) > 0, "Facts should be saved to the database"
    assert facts[0].content == "Test Target is a test entity."
    assert facts[0].confidence_score > 0
    assert facts[0].session_id == session_id
