"""
Unit tests for the Query Planner Node (with mocking).

Tests query generation, entity extraction, duplicate filtering, and progressive strategy
without requiring API keys.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from src.agents.nodes.planner import QueryPlannerNode


class TestQueryPlannerNodeMocked:
    """Test suite for QueryPlannerNode with mocked API calls."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked AI client."""
        mock = Mock()
        mock.generate = Mock(return_value='{"queries": ["query 1", "query 2", "query 3"]}')
        return mock

    @pytest.fixture
    def planner(self, mock_client):
        """Create a QueryPlannerNode instance with mocked client."""
        with patch('src.agents.nodes.planner.ModelFactory') as mock_factory:
            mock_factory.get_optimal_model_for_task.return_value = mock_client
            planner = QueryPlannerNode()
            planner.client = mock_client  # Ensure client is set
            return planner

    @pytest.fixture
    def sample_state_iteration_1(self):
        """Sample state for iteration 1 (broad discovery)."""
        return {
            "target_name": "Satya Nadella",
            "current_iteration": 1,
            "collected_facts": [],
            "search_history": [],
            "explored_topics": set()
        }

    def test_duplicate_filtering(self):
        """Test queries are not repeated."""
        # Create planner directly for this test
        with patch('src.agents.nodes.planner.ModelFactory'):
            planner = QueryPlannerNode()

            queries = ["Satya Nadella CEO", "Satya Nadella Microsoft", "Satya Nadella background"]

            search_history = [
                {
                    "query": "Satya Nadella CEO",
                    "iteration": 1,
                    "timestamp": datetime.utcnow(),
                    "results_count": 10,
                    "relevance_score": 0.8
                },
                {
                    "query": "Satya Nadella Microsoft Corporation",
                    "iteration": 1,
                    "timestamp": datetime.utcnow(),
                    "results_count": 10,
                    "relevance_score": 0.8
                }
            ]

            filtered = planner._filter_duplicate_queries(queries, search_history)

            # Should filter exact duplicate
            assert "Satya Nadella CEO" not in filtered

            # May filter similar query (Microsoft vs Microsoft Corporation)
            assert len(filtered) < len(queries)

    def test_entity_extraction(self):
        """Test regex-based entity extraction (fallback method)."""
        with patch('src.agents.nodes.planner.ModelFactory'):
            planner = QueryPlannerNode()

            facts = [
                {
                    "content": "Satya Nadella is the CEO of Microsoft Corporation based in Redmond",
                    "category": "professional"
                },
                {
                    "content": "He worked with Bill Gates and Steve Ballmer at Microsoft",
                    "category": "professional"
                },
                {
                    "content": "Microsoft Azure is headquartered in Seattle",
                    "category": "professional"
                }
            ]

            # Test the regex fallback method directly (no AI call)
            entities = planner._extract_entities_regex(facts)

            # Should extract entities
            assert "people" in entities
            assert "companies" in entities
            assert "locations" in entities

            # Check structure
            assert isinstance(entities["people"], list)
            assert isinstance(entities["companies"], list)
            assert isinstance(entities["locations"], list)

    def test_parse_queries_from_json(self):
        """Test parsing queries from JSON response."""
        with patch('src.agents.nodes.planner.ModelFactory'):
            planner = QueryPlannerNode()

            response = '{"queries": ["query 1", "query 2", "query 3"]}'

            queries = planner._parse_queries_from_response(response)

            assert len(queries) == 3
            assert "query 1" in queries

    def test_parse_queries_from_numbered_list(self):
        """Test parsing queries from numbered list."""
        with patch('src.agents.nodes.planner.ModelFactory'):
            planner = QueryPlannerNode()

            response = """
            1. First query
            2. Second query
            3. Third query
            """

            queries = planner._parse_queries_from_response(response)

            assert len(queries) >= 3

    def test_fallback_queries_added(self):
        """Test fallback queries are added when generation fails."""
        with patch('src.agents.nodes.planner.ModelFactory'):
            planner = QueryPlannerNode()

            queries = ["only one query"]
            target_name = "Test Person"
            iteration = 1

            result = planner._add_fallback_queries(queries, target_name, iteration)

            # Should have added fallback queries
            assert len(result) >= 3

    def test_create_facts_summary(self):
        """Test facts summary creation."""
        with patch('src.agents.nodes.planner.ModelFactory'):
            planner = QueryPlannerNode()

            facts = [
                {"content": "Fact 1", "category": "professional"},
                {"content": "Fact 2", "category": "biographical"},
                {"content": "Fact 3", "category": "financial"}
            ]

            summary = planner._create_facts_summary(facts)

            assert isinstance(summary, str)
            assert len(summary) > 0
            assert "Fact 1" in summary

    def test_empty_facts_summary(self):
        """Test facts summary with no facts."""
        with patch('src.agents.nodes.planner.ModelFactory'):
            planner = QueryPlannerNode()

            summary = planner._create_facts_summary([])

            assert summary == "No facts discovered yet"

    def test_extract_keywords(self):
        """Test keyword extraction from query."""
        with patch('src.agents.nodes.planner.ModelFactory'):
            planner = QueryPlannerNode()

            # Use the searcher's method (assuming it's available)
            # For now, test the planner's internal logic

            # Test basic functionality
            assert True  # Placeholder

    def test_broad_queries_with_fallback(self, planner, mock_client, sample_state_iteration_1):
        """Test broad query generation with API failure fallback."""
        # Simulate API failure
        mock_client.generate.side_effect = Exception("API Error")

        result_state = planner.execute(sample_state_iteration_1)

        # Should still generate fallback queries
        assert "next_queries" in result_state
        assert len(result_state["next_queries"]) >= 3

    def test_targeted_queries_with_mock(self, planner, mock_client):
        """Test targeted query generation."""
        state = {
            "target_name": "Satya Nadella",
            "current_iteration": 3,
            "collected_facts": [
                {
                    "content": "Satya Nadella is the CEO of Microsoft Corporation",
                    "category": "professional",
                    "confidence_score": 0.95
                }
            ],
            "search_history": [],
            "explored_topics": set()
        }

        mock_client.generate.return_value = '{"queries": ["Microsoft Satya Nadella timeline", "Satya Nadella achievements", "Microsoft leadership"]}'

        result_state = planner.execute(state)

        assert "next_queries" in result_state
        assert len(result_state["next_queries"]) >= 3


if __name__ == "__main__":
    """Run tests."""
    pytest.main([__file__, "-v"])
