"""
Integration tests for ContentExtractorNode and ValidatorNode with real API calls.

These tests verify the end-to-end flow from search results to validated facts.
"""

import pytest
from datetime import datetime

from src.agents.nodes.extractor import ContentExtractorNode
from src.agents.nodes.validator import ValidatorNode
from src.tools.search.models import SearchResult


@pytest.fixture
def sample_search_results():
    """Create realistic search results for testing."""
    return [
        SearchResult(
            title="Satya Nadella - CEO of Microsoft | Biography",
            url="https://www.microsoft.com/en-us/about/leadership/satya-nadella",
            content="""Satya Nadella is Chairman and Chief Executive Officer of Microsoft.
            Before being named CEO in February 2014, Nadella held leadership roles in both
            enterprise and consumer businesses across the company. Joining Microsoft in 1992,
            Nadella quickly became known as a leader who could span a breadth of technologies
            and businesses to drive transformation. He holds a bachelor's degree in electrical
            engineering from Mangalore University, a master's degree in computer science from
            the University of Wisconsin – Milwaukee and a master's degree in business administration
            from the University of Chicago Booth School of Business.""",
            score=0.0,
            source_domain="microsoft.com",
            search_engine="serpapi"
        ),
        SearchResult(
            title="Microsoft CEO Satya Nadella's Journey",
            url="https://www.nytimes.com/tech/microsoft-ceo-nadella",
            content="""Satya Nadella, who became Microsoft's CEO in 2014, has transformed
            the company into a cloud computing powerhouse. Born in Hyderabad, India, Nadella
            joined Microsoft in 1992 after working at Sun Microsystems. Under his leadership,
            Microsoft's market capitalization has grown significantly.""",
            score=1.0,
            source_domain="nytimes.com",
            search_engine="serpapi"
        ),
        SearchResult(
            title="Nadella's Education and Early Career",
            url="https://www.linkedin.com/in/satyanadella",
            content="""Satya Nadella studied electrical engineering at Mangalore University
            before earning his MS in Computer Science from University of Wisconsin-Milwaukee.
            He later completed an MBA from the University of Chicago Booth School of Business.
            He worked at Sun Microsystems before joining Microsoft.""",
            score=2.0,
            source_domain="linkedin.com",
            search_engine="serpapi"
        )
    ]


class TestExtractorIntegration:
    """Integration tests for ContentExtractorNode with real API calls."""

    @pytest.mark.integration
    def test_extractor_execute_with_real_api(self, sample_search_results):
        """Test extractor with real Gemini API calls."""
        extractor = ContentExtractorNode()

        state = {
            "raw_search_results": sample_search_results,
            "target_name": "Satya Nadella",
            "collected_facts": []
        }

        result = extractor.execute(state)

        # Verify state structure
        assert "collected_facts" in result
        facts = result["collected_facts"]

        # Should extract multiple facts
        assert len(facts) > 0, "Should extract at least some facts"

        # Verify fact structure
        for fact in facts:
            assert "content" in fact
            assert "category" in fact
            assert "confidence" in fact
            assert "source_url" in fact
            assert "source_domain" in fact
            assert "entities" in fact

            # Verify confidence is in valid range
            assert 0.0 <= fact["confidence"] <= 1.0

            # Verify category is valid
            assert fact["category"] in ["biographical", "professional", "financial", "behavioral"]

            # Verify entities structure
            assert isinstance(fact["entities"], dict)
            assert "people" in fact["entities"]
            assert "companies" in fact["entities"]
            assert "locations" in fact["entities"]
            assert "dates" in fact["entities"]

        # Verify at least one fact mentions Satya Nadella
        nadella_mentioned = any("Nadella" in fact["content"] for fact in facts)
        assert nadella_mentioned, "Should extract facts about Satya Nadella"

        # Verify at least one fact mentions Microsoft
        microsoft_mentioned = any(
            "Microsoft" in fact["content"] or
            "Microsoft" in fact["entities"].get("companies", [])
            for fact in facts
        )
        assert microsoft_mentioned, "Should extract facts about Microsoft"

        print(f"\n[Extractor Test] Extracted {len(facts)} facts")
        for i, fact in enumerate(facts[:3], 1):
            print(f"  Fact {i}: {fact['content'][:80]}... (confidence: {fact['confidence']:.2f})")


class TestValidatorIntegration:
    """Integration tests for ValidatorNode with real API calls."""

    @pytest.mark.integration
    def test_validator_execute_with_real_api(self, sample_search_results):
        """Test validator with real Claude API calls."""
        # First extract facts
        extractor = ContentExtractorNode()
        extractor_state = {
            "raw_search_results": sample_search_results,
            "target_name": "Satya Nadella",
            "collected_facts": []
        }
        extractor_result = extractor.execute(extractor_state)

        # Now validate facts
        validator = ValidatorNode()
        validator_state = {
            "collected_facts": extractor_result["collected_facts"],
            "target_name": "Satya Nadella"
        }

        result = validator.execute(validator_state)

        # Verify state structure
        assert "collected_facts" in result
        facts = result["collected_facts"]

        # Should have validated facts
        assert len(facts) > 0, "Should have validated facts"

        # Verify validation metadata
        for fact in facts:
            # Check if validation adjustments were applied
            if "validation_adjustments" in fact:
                assert isinstance(fact["validation_adjustments"], list)

            # Check if corroboration count was calculated
            if "corroborations" in fact:
                assert isinstance(fact["corroborations"], int)
                assert fact["corroborations"] >= 0

            # Confidence should still be in valid range
            assert 0.0 <= fact["confidence"] <= 1.0

        print(f"\n[Validator Test] Validated {len(facts)} facts")
        for i, fact in enumerate(facts[:3], 1):
            adjustments = fact.get("validation_adjustments", [])
            corroborations = fact.get("corroborations", 0)
            print(
                f"  Fact {i}: confidence={fact['confidence']:.2f}, "
                f"corroborations={corroborations}, "
                f"adjustments={len(adjustments)}"
            )


class TestExtractorValidatorFlow:
    """Integration tests for the complete extraction and validation flow."""

    @pytest.mark.integration
    def test_end_to_end_flow(self, sample_search_results):
        """Test the complete flow from search results to validated facts."""
        # Step 1: Extract facts
        extractor = ContentExtractorNode()
        state = {
            "raw_search_results": sample_search_results,
            "target_name": "Satya Nadella",
            "collected_facts": []
        }

        state = extractor.execute(state)
        initial_facts_count = len(state["collected_facts"])

        print(f"\n[End-to-End Test] Step 1: Extracted {initial_facts_count} facts")

        # Calculate average initial confidence
        initial_avg_confidence = sum(
            f["confidence"] for f in state["collected_facts"]
        ) / len(state["collected_facts"]) if state["collected_facts"] else 0

        # Step 2: Validate facts
        validator = ValidatorNode()
        state = validator.execute(state)
        validated_facts_count = len(state["collected_facts"])

        print(f"[End-to-End Test] Step 2: Validated {validated_facts_count} facts")

        # Calculate average final confidence
        final_avg_confidence = sum(
            f["confidence"] for f in state["collected_facts"]
        ) / len(state["collected_facts"]) if state["collected_facts"] else 0

        # Verify facts were processed
        assert validated_facts_count == initial_facts_count, "Fact count should remain same"

        # Print sample validated facts
        print(f"\n[End-to-End Test] Sample validated facts:")
        for i, fact in enumerate(state["collected_facts"][:3], 1):
            print(f"\n  Fact {i}:")
            print(f"    Content: {fact['content'][:100]}...")
            print(f"    Category: {fact['category']}")
            print(f"    Confidence: {fact['confidence']:.2f}")
            print(f"    Source: {fact['source_domain']}")
            if "corroborations" in fact:
                print(f"    Corroborations: {fact['corroborations']}")

        # Print metrics
        print(f"\n[End-to-End Test] Metrics:")
        print(f"  Initial avg confidence: {initial_avg_confidence:.2f}")
        print(f"  Final avg confidence: {final_avg_confidence:.2f}")

        # Verify we have some high-confidence facts
        high_confidence_facts = [f for f in state["collected_facts"] if f["confidence"] >= 0.7]
        print(f"  High confidence facts (≥0.7): {len(high_confidence_facts)}")

        # Verify we have facts in different categories
        categories = set(f["category"] for f in state["collected_facts"])
        print(f"  Categories found: {categories}")

        # Assertions
        assert len(high_confidence_facts) > 0, "Should have at least one high-confidence fact"
        assert len(categories) >= 2, "Should have facts in multiple categories"


@pytest.mark.integration
def test_extractor_with_empty_results():
    """Test extractor handles empty results gracefully."""
    extractor = ContentExtractorNode()

    state = {
        "raw_search_results": [],
        "target_name": "Test Person",
        "collected_facts": []
    }

    result = extractor.execute(state)

    assert "collected_facts" in result
    assert len(result["collected_facts"]) == 0


@pytest.mark.integration
def test_validator_with_empty_facts():
    """Test validator handles empty facts gracefully."""
    validator = ValidatorNode()

    state = {
        "collected_facts": [],
        "target_name": "Test Person"
    }

    result = validator.execute(state)

    assert "collected_facts" in result
    assert len(result["collected_facts"]) == 0


if __name__ == "__main__":
    # Run integration tests manually
    pytest.main([__file__, "-v", "-m", "integration"])
