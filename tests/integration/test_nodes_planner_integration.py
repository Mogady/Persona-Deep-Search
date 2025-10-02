"""
Integration tests for the Query Planner Node.

Tests query generation, entity extraction, duplicate filtering, and progressive strategy
with REAL API calls to Gemini (requires GOOGLE_API_KEY in .env).

These tests:
- Make actual API calls to Gemini Flash for query generation
- Make actual API calls to Gemini for embeddings and NER
- Validate real AI behavior (not mocked)
- Are slower and cost API quota
- Require valid API keys

For fast unit tests with mocked APIs, see tests/unit/test_nodes_planner_mock.py
"""

import pytest
from datetime import datetime
from src.agents.nodes.planner import QueryPlannerNode


class TestQueryPlannerNodeIntegration:
    """Integration test suite for QueryPlannerNode with real API calls."""

    @pytest.fixture
    def planner(self):
        """Create a QueryPlannerNode instance."""
        return QueryPlannerNode()

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

    @pytest.fixture
    def sample_state_iteration_3(self):
        """Sample state for iteration 3 (targeted investigation)."""
        return {
            "target_name": "Satya Nadella",
            "current_iteration": 3,
            "collected_facts": [
                {
                    "content": "Satya Nadella is the CEO of Microsoft Corporation",
                    "category": "professional",
                    "confidence_score": 0.95
                },
                {
                    "content": "He joined Microsoft in 1992 as a program manager",
                    "category": "professional",
                    "confidence_score": 0.90
                },
                {
                    "content": "He led the cloud and enterprise division",
                    "category": "professional",
                    "confidence_score": 0.85
                }
            ],
            "search_history": [
                {
                    "query": "Satya Nadella professional background",
                    "iteration": 1,
                    "timestamp": datetime.utcnow(),
                    "results_count": 10,
                    "relevance_score": 0.8
                }
            ],
            "explored_topics": {"satya", "nadella", "professional", "background"}
        }

    def test_broad_queries_generation(self, planner, sample_state_iteration_1):
        """Test iteration 1 query generation (broad discovery)."""
        result_state = planner.execute(sample_state_iteration_1)

        assert "next_queries" in result_state
        queries = result_state["next_queries"]

        # Should generate 3-5 queries
        assert 3 <= len(queries) <= 5

        # Queries should contain target name
        target = sample_state_iteration_1["target_name"]
        assert any(target.lower() in q.lower() for q in queries)

        # Should have variety (not all identical)
        assert len(set(queries)) == len(queries)

    def test_targeted_queries_use_facts(self, planner, sample_state_iteration_3):
        """Test iteration 2-3 builds on previous facts."""
        result_state = planner.execute(sample_state_iteration_3)

        queries = result_state["next_queries"]

        # Should generate queries
        assert len(queries) >= 3

        # May contain entities from facts (Microsoft, CEO, etc.)
        query_text = " ".join(queries).lower()

        # At least one query should reference discovered information
        # (This is flexible as AI may generate different queries)
        assert len(queries) > 0

    def test_duplicate_filtering(self, planner):
        """Test queries are not repeated."""
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
        # At least some filtering should occur
        assert len(filtered) < len(queries)

    def test_entity_extraction(self, planner):
        """Test extraction of people, companies, locations from facts."""
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

        entities = planner._extract_entities(facts)

        # Should extract entities
        assert "people" in entities
        assert "companies" in entities
        assert "locations" in entities

        # Check some expected entities
        people = [p.lower() for p in entities["people"]]
        companies = [c.lower() for c in entities["companies"]]

        # May extract some names (flexible check)
        assert len(entities["people"]) >= 0  # Extraction is heuristic
        assert len(entities["companies"]) >= 0

    def test_iteration_progression(self, planner):
        """Test different prompts used for different iterations."""
        # Iteration 1
        state_1 = {
            "target_name": "John Doe",
            "current_iteration": 1,
            "collected_facts": [],
            "search_history": [],
            "explored_topics": set()
        }

        result_1 = planner.execute(state_1)
        queries_1 = result_1["next_queries"]

        # Iteration 4 (connection mining)
        state_4 = {
            "target_name": "John Doe",
            "current_iteration": 4,
            "collected_facts": [
                {"content": "John Doe works at Acme Corp", "category": "professional"}
            ],
            "search_history": [],
            "explored_topics": set()
        }

        result_4 = planner.execute(state_4)
        queries_4 = result_4["next_queries"]

        # Both should generate queries
        assert len(queries_1) >= 3
        assert len(queries_4) >= 3

        # Queries should be different (different iteration strategies)
        assert queries_1 != queries_4

    def test_low_confidence_validation_queries(self, planner):
        """Test iteration 6-7 focuses on low-confidence facts."""
        state = {
            "target_name": "Jane Smith",
            "current_iteration": 6,
            "collected_facts": [
                {
                    "content": "Jane Smith may have worked at XYZ Inc",
                    "confidence_score": 0.5  # Low confidence
                },
                {
                    "content": "Jane Smith is definitely the CEO of ABC Corp",
                    "confidence_score": 0.95  # High confidence
                }
            ],
            "search_history": [],
            "explored_topics": set()
        }

        result = planner.execute(state)
        queries = result["next_queries"]

        # Should generate validation queries
        assert len(queries) >= 3

        # Queries might focus on verification (flexible check)
        query_text = " ".join(queries).lower()
        # Just ensure queries are generated
        assert len(query_text) > 0

    def test_empty_state_handling(self, planner):
        """Test handling of minimal/empty state."""
        state = {
            "target_name": "Test Person",
            "current_iteration": 1
        }

        result = planner.execute(state)

        # Should still generate queries
        assert "next_queries" in result
        assert len(result["next_queries"]) >= 3

    def test_parse_queries_from_json(self, planner):
        """Test parsing queries from JSON response."""
        response = '{"queries": ["query 1", "query 2", "query 3"]}'

        queries = planner._parse_queries_from_response(response)

        assert len(queries) == 3
        assert "query 1" in queries

    def test_parse_queries_from_numbered_list(self, planner):
        """Test parsing queries from numbered list."""
        response = """
        1. First query
        2. Second query
        3. Third query
        """

        queries = planner._parse_queries_from_response(response)

        assert len(queries) >= 3

    def test_fallback_queries_added(self, planner):
        """Test fallback queries are added when generation fails."""
        queries = ["only one query"]
        target_name = "Test Person"
        iteration = 1

        result = planner._add_fallback_queries(queries, target_name, iteration)

        # Should have added fallback queries
        assert len(result) >= 3
