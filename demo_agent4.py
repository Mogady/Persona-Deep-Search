#!/usr/bin/env python3
"""
Demo script for Agent 4: Query Planner & Search Executor Nodes

This demonstrates the complete flow from query planning to search execution.
Requires API keys in .env file.
"""

import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.nodes import QueryPlannerNode, SearchExecutorNode
from src.utils.logger import get_logger

logger = get_logger(__name__)


def demo_planner_searcher_flow():
    """Demonstrate the Query Planner → Search Executor flow."""

    print("\n" + "="*80)
    print("AGENT 4 DEMO: Query Planning & Search Execution")
    print("="*80 + "\n")

    # Initialize nodes
    print("Initializing nodes...")
    planner = QueryPlannerNode()
    searcher = SearchExecutorNode()
    print("✅ Nodes initialized\n")

    # Create initial research state
    target_name = "Satya Nadella"
    state = {
        "target_name": target_name,
        "current_iteration": 1,
        "collected_facts": [],
        "search_history": [],
        "explored_topics": set()
    }

    print(f"🎯 Research Target: {target_name}")
    print(f"📊 Initial Iteration: {state['current_iteration']}\n")

    # Iteration 1: Broad Discovery
    print("-" * 80)
    print("ITERATION 1: BROAD DISCOVERY")
    print("-" * 80)

    print("\n1️⃣ Query Planning...")
    state = planner.execute(state)
    queries = state.get("next_queries", [])

    print(f"\n   Generated {len(queries)} queries:")
    for i, query in enumerate(queries, 1):
        print(f"   {i}. {query}")

    print("\n2️⃣ Search Execution...")
    state = searcher.execute(state)

    results = state.get("raw_search_results", [])
    search_history = state.get("search_history", [])

    print(f"\n   Retrieved {len(results)} search results")
    print(f"   Search history entries: {len(search_history)}")
    print(f"   Current iteration: {state['current_iteration']}")

    if results:
        print(f"\n   Sample results:")
        for i, result in enumerate(results[:3], 1):
            print(f"   {i}. {result.title[:60]}...")
            print(f"      Source: {result.source_domain}")

    # Show explored topics
    explored = state.get("explored_topics", set())
    print(f"\n   Explored topics: {', '.join(list(explored)[:10])}")

    # Simulate iteration 2 with some facts
    print("\n" + "-" * 80)
    print("ITERATION 2: TARGETED INVESTIGATION (with simulated facts)")
    print("-" * 80)

    # Add some simulated facts
    state["collected_facts"] = [
        {
            "content": "Satya Nadella is the CEO of Microsoft Corporation",
            "category": "professional",
            "confidence_score": 0.95
        },
        {
            "content": "He joined Microsoft in 1992 as a program manager",
            "category": "professional",
            "confidence_score": 0.90
        }
    ]

    print("\n1️⃣ Query Planning...")
    state = planner.execute(state)
    queries = state.get("next_queries", [])

    print(f"\n   Generated {len(queries)} queries:")
    for i, query in enumerate(queries, 1):
        print(f"   {i}. {query}")

    print("\n   Note: Queries should now be more targeted based on discovered facts")

    # Summary
    print("\n" + "="*80)
    print("DEMO SUMMARY")
    print("="*80)
    print(f"\n✅ Total iterations completed: {state['current_iteration'] - 1}")
    print(f"✅ Total search queries executed: {len(state['search_history'])}")
    print(f"✅ Total search results collected: {len(results)}")
    print(f"✅ Total facts discovered: {len(state['collected_facts'])}")
    print(f"✅ Explored topics: {len(explored)}")

    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("  - Agent 5: Extract facts from search results")
    print("  - Agent 6: Analyze risks and map connections")
    print("  - Agent 7: Integrate into LangGraph workflow")
    print("  - Agent 8: Build Chainlit UI\n")


if __name__ == "__main__":
    try:
        demo_planner_searcher_flow()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Valid API keys in .env file (GOOGLE_API_KEY, ANTHROPIC_API_KEY, SERPAPI_KEY)")
        print("  2. All dependencies installed (pip install -r requirements.txt)")
        sys.exit(1)
