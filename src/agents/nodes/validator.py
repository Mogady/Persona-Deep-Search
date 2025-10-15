"""
Validator Node - Validates and cross-references facts.

This node uses a model for complex reasoning to validate facts,
detect contradictions, and adjust confidence scores.
"""

from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict
import json
import numpy as np
from src.utils.similarity import cosine_similarity

from src.tools.models import ModelFactory
from src.utils.logger import get_logger
from src.utils.config import Config
from src.database.repository import ResearchRepository
from src.database.models import FactCategory
from src.prompts.templates.validator_prompt import (
    VALIDATION_SYSTEM_PROMPT,
    CONTRADICTION_DETECTION_PROMPT,

)
from src.utils.json_parser import parse_json_object


class ValidatorNode:
    """
    Validates and cross-references facts using complex reasoning.
    Adjusts confidence scores based on multiple factors.
    """

    # Authoritative domain patterns
    AUTHORITATIVE_DOMAINS = {
        "gov": 0.9,    # Government sites
        "edu": 0.85,   # Educational institutions
        "org": 0.75,   # Official organizations
    }

    # High-quality news outlets
    HIGH_QUALITY_NEWS = {
        "nytimes.com": 0.85,
        "wsj.com": 0.85,
        "reuters.com": 0.9,
        "bloomberg.com": 0.85,
        "ft.com": 0.85,
        "economist.com": 0.85,
        "bbc.com": 0.85,
        "apnews.com": 0.9,
        "theguardian.com": 0.8,
        "washingtonpost.com": 0.8,
    }

    def __init__(self, config: Config, repository: ResearchRepository):
        """
        Initialize the validator.

        Args:
            config: Configuration object with all settings
            repository: Database repository for persistence
        """
        self.config = config
        self.repository = repository
        self.client = ModelFactory.get_optimal_model_for_task("complex_reasoning")
        # Also get a client for embeddings
        self.gemini_client = ModelFactory.get_optimal_model_for_task("extraction")
        self.logger = get_logger(__name__)

        # Load config values
        self.temperature = config.performance.validation_temperature
        self.max_concurrent_llm_calls = config.performance.max_concurrent_llm_calls

    def execute(self, state: Dict) -> Dict:
        """
        Validate facts and adjust confidence scores.

        Args:
            state: Current ResearchState dict with:
                - new_facts: List[Dict] (newly extracted facts in current iteration)
                - collected_facts: List[Dict] (all accumulated facts for context)

        Returns:
            Updated state with validated new facts merged back into collected_facts
        """
        new_facts = state.get("new_facts", [])
        collected_facts = state.get("collected_facts", [])
        target_name = state.get("target_name", "")

        if not new_facts:
            self.logger.info("No new facts to validate in this iteration")
            return state

        self.logger.info(
            f"Validating {len(new_facts)} new facts for '{target_name}' "
            f"(total accumulated: {len(collected_facts)})"
        )

        # Step 1: Cross-reference new facts with ALL facts (for context)
        # We need all facts to check for corroboration and contradictions
        fact_groups = self._cross_reference_facts(new_facts)

        # Step 2: Detect contradictions (check new facts against all facts)
        contradictions = self._detect_contradictions(new_facts)

        # Step 3: Evaluate source authority for new facts
        source_scores = self._evaluate_source_authority(new_facts)

        # Step 4: Calculate final confidence scores for new facts
        validated_new_facts = self._calculate_final_confidence_scores(
            new_facts,
            fact_groups,
            contradictions,
            source_scores
        )

        # Step 5: Update collected_facts - replace the new ones with validated versions
        # Since new_facts were just added to collected_facts by extractor,
        # we need to replace them with validated versions
        # Remove last N facts (the new ones) and add validated versions
        if len(collected_facts) >= len(new_facts):
            collected_facts = collected_facts[:-len(new_facts)]
        collected_facts.extend(validated_new_facts)

        state["collected_facts"] = collected_facts

        # Log validation metrics
        self._log_validation_metrics(validated_new_facts, contradictions)

        # Save validated facts to database
        session_id = state.get("session_id")
        if session_id and validated_new_facts:
            try:
                # Convert facts to database format
                db_facts = []
                for fact in validated_new_facts:
                    # Map category string to FactCategory enum
                    category_str = fact.get("category", "professional").upper()
                    try:
                        category = FactCategory[category_str]
                    except KeyError:
                        category = FactCategory.PROFESSIONAL  # Default fallback

                    db_facts.append({
                        "session_id": session_id,
                        "content": fact["content"],
                        "source_url": fact["source_url"],
                        "category": category,
                        "confidence_score": fact["confidence"],
                        "extraction_method": "gemini_pro",
                        "raw_context": fact.get("source_domain", "")
                    })

                # Batch save to database
                saved_count = self.repository.save_facts_batch(db_facts)
                self.logger.debug(f"Saved {saved_count}/{len(db_facts)} validated facts to database")

            except Exception as e:
                self.logger.error(f"Failed to save facts to database: {e}", exc_info=True)
                # Don't fail the workflow if DB save fails

        return state

    def _cross_reference_facts(self, facts: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group similar facts and find corroborating evidence.

        Uses semantic similarity (embeddings) for grouping.

        Args:
            facts: List of fact dictionaries

        Returns:
            Dictionary mapping group IDs to lists of similar facts
        """
        if len(facts) < 2:
            return {}

        try:
            self.logger.info("Cross-referencing facts using semantic similarity")

            # Use embeddings-based grouping (more accurate)
            fact_groups = self._use_semantic_similarity_for_grouping(facts)

            self.logger.info(f"Identified {len(fact_groups)} fact groups")

            return fact_groups

        except Exception as e:
            self.logger.error(f"Cross-referencing failed: {e}")
            return {}

    def _use_semantic_similarity_for_grouping(
        self,
        facts: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Use Gemini embeddings to group semantically similar facts.

        Args:
            facts: List of fact dictionaries

        Returns:
            Dictionary of fact groups
        """
        try:
            # Extract fact contents
            fact_contents = [f.get("content", "") for f in facts]

            # Generate embeddings
            self.logger.info(f"Generating embeddings for {len(fact_contents)} facts")
            embeddings = self.gemini_client.generate_embeddings(fact_contents)

            # Convert to numpy array for easier computation
            embeddings_array = np.array(embeddings)

            # Compute pairwise cosine similarity
            n_facts = len(embeddings_array)
            similarities = np.zeros((n_facts, n_facts))
            for i in range(n_facts):
                similarities[i] = cosine_similarity(embeddings_array[i], embeddings_array)

            # Group facts by similarity threshold
            similarity_threshold = 0.65  # High threshold for similar facts
            groups = self._cluster_by_similarity(similarities, similarity_threshold)

            # Build fact groups dictionary
            fact_groups = {}
            for group_id, indices in enumerate(groups):
                if len(indices) > 1:  # Only keep groups with 2+ facts
                    group_facts = [facts[i] for i in indices]
                    fact_groups[f"group_{group_id}"] = group_facts

            return fact_groups

        except Exception as e:
            self.logger.error(f"Semantic similarity grouping failed: {e}")
            # Fallback to simple text matching
            return self._simple_text_matching(facts)

    def _cluster_by_similarity(
        self,
        similarities: np.ndarray,
        threshold: float
    ) -> List[List[int]]:
        """
        Cluster facts by similarity threshold.

        Args:
            similarities: Similarity matrix
            threshold: Similarity threshold (0.0 to 1.0)

        Returns:
            List of clusters (each cluster is a list of indices)
        """
        n = similarities.shape[0]
        visited = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue

            # Find all facts similar to fact i
            cluster = [i]
            visited.add(i)

            for j in range(i + 1, n):
                if j not in visited and similarities[i, j] >= threshold:
                    cluster.append(j)
                    visited.add(j)

            clusters.append(cluster)

        return clusters

    def _simple_text_matching(self, facts: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Fallback: Simple text-based fact grouping.

        Args:
            facts: List of fact dictionaries

        Returns:
            Dictionary of fact groups
        """
        groups = defaultdict(list)

        for fact in facts:
            content = fact.get("content", "").lower()
            # Simple keyword-based grouping
            key_words = set(content.split()[:5])  # First 5 words as key
            key = "_".join(sorted(key_words))
            groups[key].append(fact)

        # Filter groups with only 1 fact
        return {k: v for k, v in groups.items() if len(v) > 1}

    def _detect_contradictions(self, facts: List[Dict]) -> List[Dict]:
        """
        Detect contradicting facts using Claude's reasoning.

        Args:
            facts: List of fact dictionaries

        Returns:
            List of contradiction records
        """
        if len(facts) < 2:
            return []

        try:
            self.logger.info("Detecting contradictions using Claude")

            # Prepare facts for Claude
            facts_json = json.dumps(
                [
                    {
                        "index": i,
                        "content": f.get("content", ""),
                        "source": f.get("source_domain", "")
                    }
                    for i, f in enumerate(facts)
                ],
                indent=2
            )

            # Use Claude for contradiction detection with rate limiting
            target_name = facts[0].get("entities", {}).get("people", [""])[0] if facts else ""
            prompt = CONTRADICTION_DETECTION_PROMPT.format(
                target_name=target_name,
                facts_json=facts_json
            )

            response = self.client.generate(
                prompt=prompt,
                system_instruction=VALIDATION_SYSTEM_PROMPT,
                temperature=self.temperature
            )

            # Parse response
            result = parse_json_object(response)
            contradictions = result.get("contradictions", [])

            self.logger.info(f"Detected {len(contradictions)} contradictions")

            return contradictions

        except Exception as e:
            self.logger.error(f"Contradiction detection failed: {e}")
            return []

    def _evaluate_source_authority(self, facts: List[Dict]) -> Dict[str, float]:
        """
        Evaluate source authority for all domains.

        Args:
            facts: List of fact dictionaries

        Returns:
            Dictionary mapping domains to authority scores
        """
        # Extract unique domains
        domains = set(f.get("source_domain", "") for f in facts if f.get("source_domain"))

        if not domains:
            return {}

        source_scores = {}

        for domain in domains:
            score = self._score_source_authority(domain)
            source_scores[domain] = score

        self.logger.info(f"Evaluated authority for {len(source_scores)} domains")

        return source_scores

    def _score_source_authority(self, source_domain: str) -> float:
        """
        Rate domain reliability (0.0-1.0).

        Args:
            source_domain: Domain name

        Returns:
            Authority score
        """
        # Check high-quality news
        if source_domain in self.HIGH_QUALITY_NEWS:
            return self.HIGH_QUALITY_NEWS[source_domain]

        # Check authoritative TLDs
        for tld, score in self.AUTHORITATIVE_DOMAINS.items():
            if source_domain.endswith(f".{tld}"):
                return score

        # Check for low-quality patterns
        low_quality_patterns = ["blog", "forum", "reddit", "twitter", "facebook"]
        for pattern in low_quality_patterns:
            if pattern in source_domain.lower():
                return 0.3

        # Default medium authority
        return 0.5

    def _calculate_final_confidence_scores(
        self,
        facts: List[Dict],
        fact_groups: Dict[str, List[Dict]],
        contradictions: List[Dict],
        source_scores: Dict[str, float]
    ) -> List[Dict]:
        """
        Calculate final confidence scores for all facts.

        Algorithm:
        - Base score from extraction
        - +0.2 if multiple independent sources (corroborations >= 2)
        - +0.1 if authoritative domain
        - +0.1 if recent information (< 6 months)
        - -0.3 if contradictions found
        - -0.1 if single source only
        - Clamp to [0.0, 1.0]

        Args:
            facts: List of fact dictionaries
            fact_groups: Groups of similar facts
            contradictions: List of contradictions
            source_scores: Domain authority scores

        Returns:
            List of facts with updated confidence scores
        """
        # Build corroboration map
        corroboration_map = self._build_corroboration_map(facts, fact_groups)

        # Build contradiction map
        contradiction_indices = set()
        for contradiction in contradictions:
            fact_indices = contradiction.get("fact_indices", [])
            contradiction_indices.update(fact_indices)

        # Update confidence for each fact
        for i, fact in enumerate(facts):
            base_confidence = fact.get("confidence", 0.5)
            adjustments = []

            # Corroboration boost
            corroborations = corroboration_map.get(i, 0)
            if corroborations >= 2:
                adjustments.append(("multiple_sources", 0.2))
            elif corroborations == 0:
                adjustments.append(("single_source", -0.1))

            # Source authority boost
            source_domain = fact.get("source_domain", "")
            source_score = source_scores.get(source_domain, 0.5)
            if source_score >= 0.8:
                adjustments.append(("authoritative_source", 0.1))

            # Recency boost
            is_recent = self._check_recency(fact)
            if is_recent:
                adjustments.append(("recent_info", 0.1))

            # Contradiction penalty
            if i in contradiction_indices:
                adjustments.append(("contradiction", -0.3))

            # Calculate final confidence
            total_adjustment = sum(adj[1] for adj in adjustments)
            final_confidence = base_confidence + total_adjustment

            # Clamp to [0.0, 1.0]
            final_confidence = max(0.0, min(1.0, final_confidence))

            # Update fact
            fact["confidence"] = final_confidence
            fact["validation_adjustments"] = adjustments
            fact["corroborations"] = corroborations

            self.logger.debug(
                f"Fact {i}: {base_confidence:.2f} -> {final_confidence:.2f} "
                f"(adjustments: {adjustments})"
            )

        return facts

    def _build_corroboration_map(
        self,
        facts: List[Dict],
        fact_groups: Dict[str, List[Dict]]
    ) -> Dict[int, int]:
        """
        Build map of fact index to corroboration count.

        Args:
            facts: List of facts
            fact_groups: Groups of similar facts

        Returns:
            Dictionary mapping fact index to corroboration count
        """
        corroboration_map = defaultdict(int)

        # For each group, count corroborations
        for group_facts in fact_groups.values():
            # Find indices of facts in this group
            group_indices = []
            for group_fact in group_facts:
                for i, fact in enumerate(facts):
                    if fact.get("content") == group_fact.get("content"):
                        group_indices.append(i)
                        break

            # Count corroborations (group size - 1)
            corroboration_count = len(group_indices) - 1

            for idx in group_indices:
                corroboration_map[idx] = corroboration_count

        return dict(corroboration_map)

    def _check_recency(self, fact: Dict) -> bool:
        """
        Check if fact is from recent information (< 6 months).

        Args:
            fact: Fact dictionary

        Returns:
            True if recent, False otherwise
        """
        extracted_date = fact.get("extracted_date")

        if isinstance(extracted_date, datetime):
            age = datetime.utcnow() - extracted_date
            return age < timedelta(days=180)  # 6 months

        # If no date, assume not recent
        return False

    def _log_validation_metrics(
        self,
        facts: List[Dict],
        contradictions: List[Dict]
    ) -> None:
        """
        Log validation metrics for monitoring.

        Args:
            facts: List of validated facts
            contradictions: List of contradictions
        """
        if not facts:
            return

        # Calculate average confidence before and after
        avg_confidence = sum(f.get("confidence", 0) for f in facts) / len(facts)

        # Count facts with high confidence
        high_confidence = sum(1 for f in facts if f.get("confidence", 0) >= 0.8)

        # Count corroborated facts
        corroborated = sum(1 for f in facts if f.get("corroborations", 0) > 0)

        self.logger.info(
            f"Validation metrics: "
            f"total={len(facts)}, "
            f"avg_confidence={avg_confidence:.2f}, "
            f"high_confidence={high_confidence}, "
            f"corroborated={corroborated}, "
            f"contradictions={len(contradictions)}"
        )
