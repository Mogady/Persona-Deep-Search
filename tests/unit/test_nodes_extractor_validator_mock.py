"""
Unit tests for ContentExtractorNode and ValidatorNode (mocked, no API calls).

These tests verify the logic of extraction and validation without making actual API calls.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.agents.nodes.extractor import ContentExtractorNode
from src.agents.nodes.validator import ValidatorNode
from src.tools.search.models import SearchResult


class TestContentExtractorNode:
    """Test suite for ContentExtractorNode."""

    @pytest.fixture
    def extractor(self):
        """Create ContentExtractorNode instance."""
        with patch('src.agents.nodes.extractor.ModelFactory'):
            return ContentExtractorNode()

    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results."""
        return [
            SearchResult(
                title="Satya Nadella Biography",
                url="https://microsoft.com/ceo",
                content="Satya Nadella became CEO of Microsoft in February 2014. He was born in Hyderabad, India.",
                score=0.0,
                source_domain="microsoft.com",
                search_engine="serpapi"
            ),
            SearchResult(
                title="Microsoft CEO Profile",
                url="https://nytimes.com/tech/nadella",
                content="Nadella holds a degree from MIT and joined Microsoft in 1992.",
                score=1.0,
                source_domain="nytimes.com",
                search_engine="serpapi"
            )
        ]

    def test_preliminary_confidence_authoritative_domain(self, extractor):
        """Test that authoritative domains get confidence boost."""
        fact = {"content": "Test fact"}

        # Test .gov domain
        boost = extractor._assign_preliminary_confidence(fact, "example.gov")
        assert boost > 0, "Government domain should get positive boost"

        # Test .edu domain
        boost = extractor._assign_preliminary_confidence(fact, "mit.edu")
        assert boost > 0, "Educational domain should get positive boost"

    def test_preliminary_confidence_high_quality_news(self, extractor):
        """Test that high-quality news sources get confidence boost."""
        fact = {"content": "Test fact"}

        boost = extractor._assign_preliminary_confidence(fact, "nytimes.com")
        assert boost > 0, "NYT should get positive boost"

        boost = extractor._assign_preliminary_confidence(fact, "reuters.com")
        assert boost > 0, "Reuters should get positive boost"

    def test_preliminary_confidence_low_quality_source(self, extractor):
        """Test that low-quality sources get penalized."""
        fact = {"content": "Test fact"}

        boost = extractor._assign_preliminary_confidence(fact, "someblog.com")
        assert boost < 0, "Blog domain should get negative boost"

    def test_categorize_fact_biographical(self, extractor):
        """Test biographical fact categorization."""
        category = extractor._categorize_fact("He was born in New York in 1980")
        assert category == "biographical"

        category = extractor._categorize_fact("Graduated from MIT with honors")
        assert category == "biographical"

    def test_categorize_fact_professional(self, extractor):
        """Test professional fact categorization."""
        category = extractor._categorize_fact("He is the CEO of Acme Corp")
        assert category == "professional"

        category = extractor._categorize_fact("Worked as a director at Microsoft")
        assert category == "professional"

    def test_categorize_fact_financial(self, extractor):
        """Test financial fact categorization."""
        category = extractor._categorize_fact("His salary is $5 million per year")
        assert category == "financial"

        category = extractor._categorize_fact("Invested $10 million in the startup")
        assert category == "financial"

    def test_categorize_fact_behavioral(self, extractor):
        """Test behavioral fact categorization."""
        category = extractor._categorize_fact("He said the company will expand")
        assert category == "behavioral"

        category = extractor._categorize_fact("Alleged to have violated company policy")
        assert category == "behavioral"

    def test_normalize_entities(self, extractor):
        """Test entity normalization."""
        entities = {
            "people": ["satya nadella", "BILL GATES"],
            "companies": ["Microsoft", "  Apple Inc  "],
            "locations": ["new york", "SAN FRANCISCO"],
            "dates": ["2014", " February 2020 "]
        }

        normalized = extractor._normalize_entities(entities)

        # Check title case for people
        assert "Satya Nadella" in normalized["people"]
        assert "Bill Gates" in normalized["people"]

        # Check trimming
        assert "Apple Inc" in normalized["companies"]

        # Check uniqueness (no duplicates)
        assert len(normalized["people"]) == len(set(normalized["people"]))

    def test_filter_low_quality_facts_too_short(self, extractor):
        """Test filtering of facts that are too short."""
        facts = [
            {"content": "Short", "confidence": 0.8},
            {"content": "This is a proper length fact with sufficient detail", "confidence": 0.8}
        ]

        filtered = extractor._filter_low_quality_facts(facts)
        assert len(filtered) == 1
        assert filtered[0]["content"].startswith("This is a proper")

    def test_filter_low_quality_facts_low_confidence(self, extractor):
        """Test filtering of facts with low confidence."""
        facts = [
            {"content": "This is a fact with very low confidence score", "confidence": 0.1},
            {"content": "This is a fact with acceptable confidence", "confidence": 0.6}
        ]

        filtered = extractor._filter_low_quality_facts(facts)
        assert len(filtered) == 1
        assert filtered[0]["confidence"] == 0.6

    def test_filter_low_quality_facts_vague_content(self, extractor):
        """Test filtering of vague facts."""
        facts = [
            {"content": "He is well-known in the industry for various achievements", "confidence": 0.8},
            {"content": "He became CEO of Microsoft in 2014", "confidence": 0.8}
        ]

        filtered = extractor._filter_low_quality_facts(facts)
        assert len(filtered) == 1
        assert "CEO of Microsoft" in filtered[0]["content"]

    def test_filter_low_quality_facts_duplicates(self, extractor):
        """Test deduplication of facts."""
        facts = [
            {"content": "Satya Nadella is CEO of Microsoft", "confidence": 0.8},
            {"content": "Satya Nadella is CEO of Microsoft", "confidence": 0.7},
            {"content": "He holds a degree from MIT", "confidence": 0.8}
        ]

        filtered = extractor._filter_low_quality_facts(facts)
        assert len(filtered) == 2, "Should remove duplicate"

    def test_format_results_for_batch(self, extractor, sample_search_results):
        """Test batch formatting of search results."""
        formatted = extractor._format_results_for_batch(sample_search_results)

        assert "Result 1" in formatted
        assert "Result 2" in formatted
        assert "microsoft.com" in formatted
        assert "nytimes.com" in formatted


class TestValidatorNode:
    """Test suite for ValidatorNode."""

    @pytest.fixture
    def validator(self):
        """Create ValidatorNode instance."""
        with patch('src.agents.nodes.validator.ModelFactory'):
            return ValidatorNode()

    @pytest.fixture
    def sample_facts(self):
        """Create sample facts for testing."""
        return [
            {
                "content": "Satya Nadella became CEO of Microsoft in February 2014",
                "confidence": 0.7,
                "source_domain": "microsoft.com",
                "extracted_date": datetime.utcnow()
            },
            {
                "content": "Nadella holds a degree from MIT",
                "confidence": 0.6,
                "source_domain": "mit.edu",
                "extracted_date": datetime.utcnow()
            },
            {
                "content": "He joined Microsoft in 1992",
                "confidence": 0.5,
                "source_domain": "blog.example.com",
                "extracted_date": datetime.utcnow() - timedelta(days=200)
            }
        ]

    def test_score_source_authority_high_quality_news(self, validator):
        """Test source authority scoring for high-quality news."""
        score = validator._score_source_authority("nytimes.com")
        assert score >= 0.8, "NYT should have high authority score"

        score = validator._score_source_authority("reuters.com")
        assert score >= 0.8, "Reuters should have high authority score"

    def test_score_source_authority_gov_domain(self, validator):
        """Test source authority scoring for government domains."""
        score = validator._score_source_authority("example.gov")
        assert score >= 0.8, "Government domains should have high authority"

    def test_score_source_authority_edu_domain(self, validator):
        """Test source authority scoring for educational domains."""
        score = validator._score_source_authority("mit.edu")
        assert score >= 0.8, "Educational domains should have high authority"

    def test_score_source_authority_low_quality(self, validator):
        """Test source authority scoring for low-quality sources."""
        score = validator._score_source_authority("someblog.com")
        assert score <= 0.5, "Blogs should have low authority"

        score = validator._score_source_authority("reddit.com")
        assert score <= 0.5, "Social media should have low authority"

    def test_check_recency_recent_fact(self, validator):
        """Test recency check for recent facts."""
        fact = {
            "content": "Recent fact",
            "extracted_date": datetime.utcnow()
        }

        is_recent = validator._check_recency(fact)
        assert is_recent is True, "Recently extracted facts should be marked as recent"

    def test_check_recency_old_fact(self, validator):
        """Test recency check for old facts."""
        fact = {
            "content": "Old fact",
            "extracted_date": datetime.utcnow() - timedelta(days=200)
        }

        is_recent = validator._check_recency(fact)
        assert is_recent is False, "Old facts should not be marked as recent"

    def test_check_recency_no_date(self, validator):
        """Test recency check for facts without date."""
        fact = {"content": "Fact without date"}

        is_recent = validator._check_recency(fact)
        assert is_recent is False, "Facts without date should not be marked as recent"

    def test_build_corroboration_map_empty(self, validator, sample_facts):
        """Test corroboration map building with no groups."""
        fact_groups = {}

        corroboration_map = validator._build_corroboration_map(sample_facts, fact_groups)

        assert isinstance(corroboration_map, dict)
        assert len(corroboration_map) == 0

    def test_build_corroboration_map_with_groups(self, validator, sample_facts):
        """Test corroboration map building with fact groups."""
        # Create a mock group
        fact_groups = {
            "group_1": [sample_facts[0], sample_facts[1]]
        }

        corroboration_map = validator._build_corroboration_map(sample_facts, fact_groups)

        # Facts in the group should have corroboration count
        assert 0 in corroboration_map or 1 in corroboration_map

    def test_compute_cosine_similarity_matrix(self, validator):
        """Test cosine similarity matrix computation."""
        # Create sample embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Identical to first
            [0.0, 1.0, 0.0]   # Orthogonal
        ])

        similarity = validator._compute_cosine_similarity_matrix(embeddings)

        # Check shape
        assert similarity.shape == (3, 3)

        # Check diagonal (self-similarity should be ~1.0)
        assert abs(similarity[0, 0] - 1.0) < 0.01

        # Check identical vectors
        assert abs(similarity[0, 1] - 1.0) < 0.01

        # Check orthogonal vectors
        assert abs(similarity[0, 2]) < 0.01

    def test_cluster_by_similarity(self, validator):
        """Test clustering by similarity threshold."""
        # Create similarity matrix
        similarities = np.array([
            [1.0, 0.9, 0.2],
            [0.9, 1.0, 0.1],
            [0.2, 0.1, 1.0]
        ])

        clusters = validator._cluster_by_similarity(similarities, threshold=0.8)

        # Should have 2 clusters: [0, 1] and [2]
        assert len(clusters) >= 1

        # Find the cluster with items 0 and 1
        cluster_01 = None
        for cluster in clusters:
            if 0 in cluster and 1 in cluster:
                cluster_01 = cluster
                break

        assert cluster_01 is not None, "Items 0 and 1 should be in same cluster"

    def test_simple_text_matching(self, validator, sample_facts):
        """Test simple text-based fact grouping fallback."""
        groups = validator._simple_text_matching(sample_facts)

        assert isinstance(groups, dict)
        # Should only include groups with 2+ facts
        for group_facts in groups.values():
            assert len(group_facts) >= 2

    def test_evaluate_source_authority(self, validator, sample_facts):
        """Test source authority evaluation."""
        source_scores = validator._evaluate_source_authority(sample_facts)

        assert isinstance(source_scores, dict)
        assert "microsoft.com" in source_scores
        assert "mit.edu" in source_scores

        # MIT should have high authority
        assert source_scores["mit.edu"] >= 0.8

    @patch.object(ValidatorNode, '_use_semantic_similarity_for_grouping')
    def test_cross_reference_facts_calls_semantic_grouping(
        self,
        mock_semantic_grouping,
        validator,
        sample_facts
    ):
        """Test that cross-referencing uses semantic similarity grouping."""
        mock_semantic_grouping.return_value = {}

        validator._cross_reference_facts(sample_facts)

        mock_semantic_grouping.assert_called_once()

    def test_cross_reference_facts_single_fact(self, validator):
        """Test cross-referencing with single fact returns empty."""
        single_fact = [{"content": "Single fact", "confidence": 0.7}]

        result = validator._cross_reference_facts(single_fact)

        assert result == {}


def test_extractor_execute_no_results():
    """Test extractor execute with no search results."""
    with patch('src.agents.nodes.extractor.ModelFactory'):
        extractor = ContentExtractorNode()

        state = {
            "raw_search_results": [],
            "target_name": "Test Person",
            "collected_facts": []
        }

        result = extractor.execute(state)

        assert "collected_facts" in result
        assert len(result["collected_facts"]) == 0


def test_validator_execute_no_facts():
    """Test validator execute with no facts."""
    with patch('src.agents.nodes.validator.ModelFactory'):
        validator = ValidatorNode()

        state = {
            "collected_facts": [],
            "target_name": "Test Person"
        }

        result = validator.execute(state)

        assert "collected_facts" in result
        assert len(result["collected_facts"]) == 0
