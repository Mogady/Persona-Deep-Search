"""
Integration test for Query Planner → Search Executor flow.

Tests the complete flow from query planning to search execution.
"""

import pytest
from datetime import datetime
from src.agents.nodes import QueryPlannerNode, SearchExecutorNode


class TestPlannerSearcherFlow:
    """Integration tests for planner → searcher workflow."""

    @pytest.fixture
    def planner(self):
        """Create QueryPlannerNode instance."""
        return QueryPlannerNode()

    @pytest.fixture
    def searcher(self):
        """Create SearchExecutorNode instance."""
        return SearchExecutorNode()

    @pytest.fixture
    def initial_state(self):
        """Create initial research state."""
        return {
            "target_name": "Satya Nadella",
            "current_iteration": 1,
            "collected_facts": [],
            "search_history": [],
            "explored_topics": set()
        }

    def test_planner_to_searcher_flow(self, planner, searcher, initial_state):
        """Test complete flow: planner generates → searcher executes."""
        # Step 1: Planner generates queries
        state_after_planning = planner.execute(initial_state)

        # Verify planner output
        assert "next_queries" in state_after_planning
        assert len(state_after_planning["next_queries"]) >= 3
        assert len(state_after_planning["next_queries"]) <= 5

        # Log queries for debugging
        print(f"\nGenerated queries: {state_after_planning['next_queries']}")

        # Step 2: Searcher executes queries
        state_after_search = searcher.execute(state_after_planning)

        # Verify searcher output
        assert "raw_search_results" in state_after_search
        assert "search_history" in state_after_search
        assert "current_iteration" in state_after_search

        # Iteration should increment
        assert state_after_search["current_iteration"] == 2

        # Search history should be updated
        assert len(state_after_search["search_history"]) > 0

        # Should have some results (unless all queries fail)
        # Note: This test requires actual API access
        results = state_after_search["raw_search_results"]
        print(f"\nRetrieved {len(results)} search results")

        # Search history should match number of queries
        assert len(state_after_search["search_history"]) == len(state_after_planning["next_queries"])

    def test_multi_iteration_flow(self, planner, searcher):
        """Test multiple iterations of planner → searcher flow."""
        state = {
            "target_name": "Sundar Pichai",
            "current_iteration": 1,
            "collected_facts": [],
            "search_history": [],
            "explored_topics": set()
        }

        # Iteration 1
        state = planner.execute(state)
        assert state["current_iteration"] == 1
        assert len(state["next_queries"]) >= 3

        state = searcher.execute(state)
        assert state["current_iteration"] == 2
        assert len(state["search_history"]) > 0

        # Iteration 2
        state = planner.execute(state)
        assert state["current_iteration"] == 2
        iteration_2_queries = state["next_queries"]

        state = searcher.execute(state)
        assert state["current_iteration"] == 3

        # Queries should be different between iterations
        # (though this depends on AI generation)
        print(f"\nIteration 2 queries: {iteration_2_queries}")

    def test_query_deduplication_across_iterations(self, planner, searcher):
        """Test that queries are not repeated across iterations."""
        state = {
            "target_name": "Tim Cook",
            "current_iteration": 1,
            "collected_facts": [],
            "search_history": [],
            "explored_topics": set()
        }

        # Iteration 1
        state = planner.execute(state)
        queries_1 = set(state["next_queries"])

        state = searcher.execute(state)

        # Iteration 2
        state = planner.execute(state)
        queries_2 = set(state["next_queries"])

        # Queries should be mostly different
        # (some overlap is acceptable, but not 100%)
        overlap = len(queries_1 & queries_2)
        total = len(queries_1 | queries_2)

        if total > 0:
            overlap_ratio = overlap / total
            # Allow some overlap but not too much
            assert overlap_ratio < 0.8, "Too many duplicate queries across iterations"

    def test_explored_topics_tracking(self, planner, searcher, initial_state):
        """Test that explored topics are tracked correctly."""
        # Execute planner
        state = planner.execute(initial_state)

        # Execute searcher
        state = searcher.execute(state)

        # Explored topics should be updated
        assert "explored_topics" in state
        assert len(state["explored_topics"]) > 0

        # Topics should be extracted from queries
        print(f"\nExplored topics: {state['explored_topics']}")

    def test_search_results_structure(self, planner, searcher, initial_state):
        """Test that search results have correct structure."""
        # Execute flow
        state = planner.execute(initial_state)
        state = searcher.execute(state)

        results = state.get("raw_search_results", [])

        if results:  # Only test if we have results
            # Check first result structure
            result = results[0]

            # Should have required fields
            assert hasattr(result, 'title')
            assert hasattr(result, 'url')
            assert hasattr(result, 'content')
            assert hasattr(result, 'source_domain')
            assert hasattr(result, 'search_engine')

            print(f"\nSample result: {result.title[:50]}... from {result.source_domain}")

    def test_error_handling_invalid_state(self, planner, searcher):
        """Test error handling with invalid/minimal state."""
        # Missing target_name
        invalid_state = {
            "current_iteration": 1,
            "collected_facts": [],
            "search_history": []
        }

        # Planner should handle gracefully
        state = planner.execute(invalid_state)

        # Should still produce some output
        assert "next_queries" in state

    def test_no_queries_to_execute(self, searcher):
        """Test searcher with no queries."""
        state = {
            "next_queries": [],
            "current_iteration": 1,
            "search_history": []
        }

        result = searcher.execute(state)

        # Should handle gracefully
        assert "raw_search_results" in result
        assert result["raw_search_results"] == []

    def test_state_persistence_across_flow(self, planner, searcher, initial_state):
        """Test that state is preserved and updated correctly."""
        original_target = initial_state["target_name"]

        # Execute flow
        state = planner.execute(initial_state)
        state = searcher.execute(state)

        # Original fields should persist
        assert state["target_name"] == original_target

        # New fields should be added
        assert "next_queries" in state
        assert "raw_search_results" in state
        assert "search_history" in state
        assert "current_iteration" in state

        # Iteration should increment
        assert state["current_iteration"] == 2


if __name__ == "__main__":
    """Run integration tests with verbose output."""
    pytest.main([__file__, "-v", "-s"])
