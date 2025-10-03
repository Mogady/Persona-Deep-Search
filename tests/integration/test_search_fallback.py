"""
Integration tests for SearchOrchestrator fallback scenarios.

Tests the BraveSearch fallback logic when SerpApi fails or returns insufficient results.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.tools.search import SearchOrchestrator, SearchResult
from src.utils.config import Config


@pytest.fixture
def config():
    """Load config from environment."""
    return Config.from_env()


@pytest.fixture
def orchestrator(config):
    """Create SearchOrchestrator instance."""
    return SearchOrchestrator(config)


def test_serpapi_success_no_fallback(orchestrator):
    """Test that Brave is NOT used when SerpApi returns enough results."""
    # Mock SerpApi to return sufficient results
    mock_results = [
        MagicMock(title=f"Result {i}", url=f"http://example.com/{i}", content="test")
        for i in range(5)
    ]

    with patch.object(orchestrator.serp_search, 'search', return_value=mock_results):
        # Mock Brave to track if it's called
        if orchestrator.brave_search:
            with patch.object(orchestrator.brave_search, 'search') as brave_mock:
                results = orchestrator.search("test query")

                # Verify SerpApi results returned
                assert len(results) == 5
                # Verify Brave was NOT called
                brave_mock.assert_not_called()
        else:
            results = orchestrator.search("test query")
            assert len(results) == 5


def test_serpapi_few_results_supplements_with_brave(orchestrator):
    """Test that Brave supplements when SerpApi returns < 3 results."""
    if not orchestrator.brave_search:
        pytest.skip("Brave Search not configured")

    # Mock SerpApi to return only 2 results
    serp_results = [
        MagicMock(title="Result 1", url="http://example.com/1", content="test"),
        MagicMock(title="Result 2", url="http://example.com/2", content="test"),
    ]

    # Mock Brave to return additional results
    brave_results = [
        MagicMock(title="Brave Result 1", url="http://brave.com/1", content="test"),
        MagicMock(title="Brave Result 2", url="http://brave.com/2", content="test"),
    ]

    with patch.object(orchestrator.serp_search, 'search', return_value=serp_results):
        with patch.object(orchestrator.brave_search, 'search', return_value=brave_results):
            with patch.object(orchestrator.serp_search, '_deduplicate_results', side_effect=lambda x: x):
                results = orchestrator.search("test query")

                # Should have combined results
                assert len(results) == 4
                orchestrator.brave_search.search.assert_called_once()


def test_serpapi_failure_triggers_brave_fallback(orchestrator):
    """Test that Brave is used when SerpApi fails completely."""
    if not orchestrator.brave_search:
        pytest.skip("Brave Search not configured")

    # Mock SerpApi to raise an exception
    with patch.object(orchestrator.serp_search, 'search', side_effect=Exception("API Error")):
        # Mock Brave to succeed
        brave_results = [
            MagicMock(title="Brave Result", url="http://brave.com/1", content="test")
        ]
        with patch.object(orchestrator.brave_search, 'search', return_value=brave_results):
            results = orchestrator.search("test query")

            # Should return Brave results
            assert len(results) == 1
            orchestrator.brave_search.search.assert_called_once()


def test_both_providers_fail_returns_empty(orchestrator):
    """Test graceful degradation when both providers fail."""
    if not orchestrator.brave_search:
        pytest.skip("Brave Search not configured")

    # Mock both to fail
    with patch.object(orchestrator.serp_search, 'search', side_effect=Exception("SerpApi Error")):
        with patch.object(orchestrator.brave_search, 'search', side_effect=Exception("Brave Error")):
            results = orchestrator.search("test query")

            # Should return empty list (graceful degradation)
            assert results == []


def test_serpapi_only_when_brave_not_configured():
    """Test that system works with only SerpApi (no Brave configured)."""
    # Create orchestrator with no Brave API key
    mock_config = MagicMock()
    mock_config.search.serpapi_key = "test_key"
    mock_config.search.brave_api_key = None

    orch = SearchOrchestrator(mock_config)

    # Verify Brave is not initialized
    assert orch.brave_search is None

    # Mock SerpApi
    mock_results = [MagicMock(title="Result", url="http://example.com/1", content="test")]
    with patch.object(orch.serp_search, 'search', return_value=mock_results):
        results = orch.search("test query")

        # Should work fine with just SerpApi
        assert len(results) == 1


def test_brave_supplementation_failure_returns_serpapi_results(orchestrator):
    """Test that if Brave supplementation fails, SerpApi results are still returned."""
    if not orchestrator.brave_search:
        pytest.skip("Brave Search not configured")

    # Mock SerpApi to return 2 results
    serp_results = [
        MagicMock(title="Result 1", url="http://example.com/1", content="test"),
        MagicMock(title="Result 2", url="http://example.com/2", content="test"),
    ]

    with patch.object(orchestrator.serp_search, 'search', return_value=serp_results):
        # Mock Brave to fail during supplementation
        with patch.object(orchestrator.brave_search, 'search', side_effect=Exception("Brave Error")):
            results = orchestrator.search("test query")

            # Should still return SerpApi results even though Brave failed
            assert len(results) == 2


@pytest.mark.integration
def test_real_search_with_fallback(orchestrator):
    """
    Integration test with real API calls.

    This test requires valid API keys and will make actual requests.
    """
    # Simple query
    query = "artificial intelligence"

    results = orchestrator.search(query, max_results=5)

    # Basic assertions
    assert isinstance(results, list)
    if results:  # May be empty if APIs are down
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(hasattr(r, 'title') for r in results)
        assert all(hasattr(r, 'url') for r in results)
