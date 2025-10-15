"""
Content Extractor Node - Extracts structured facts from search results.

This node processes raw search results and extracts atomic, verified facts.
"""

from typing import Dict, List
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

from src.tools.models import ModelFactory
from src.utils.logger import get_logger
from src.utils.config import Config
from src.utils.rate_limiter import get_llm_rate_limiter
from src.database.repository import ResearchRepository
from src.tools.search.models import SearchResult
from src.prompts.templates.extractor_prompt import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT,
    BATCH_EXTRACTION_PROMPT
)
from src.utils.json_parser import parse_json_object, parse_json_array



class ContentExtractorNode:
    """
    Extracts structured facts from raw search results.
    Uses a model for precise extraction with entity recognition.
    """

    # Authoritative domain patterns for preliminary confidence scoring
    AUTHORITATIVE_DOMAINS = {
        "gov": 0.15,      # Government sites
        "edu": 0.12,      # Educational institutions
        "org": 0.10,      # Official organizations
    }

    # High-quality news domains
    HIGH_QUALITY_NEWS = {
        "nytimes.com", "wsj.com", "reuters.com", "bloomberg.com",
        "ft.com", "economist.com", "bbc.com", "apnews.com",
        "theguardian.com", "washingtonpost.com", "forbes.com"
    }

    # Low-quality patterns (reduce confidence)
    LOW_QUALITY_PATTERNS = {
        "blog", "forum", "reddit", "twitter", "facebook",
        "quora", "yahoo.answers", "pinterest"
    }

    def __init__(self, config: Config, repository: ResearchRepository):
        """
        Initialize the content extractor.

        Args:
            config: Configuration object with all settings
            repository: Database repository for persistence
        """
        self.config = config
        self.repository = repository
        self.client = ModelFactory.get_optimal_model_for_task("extraction")
        self.flash_client = ModelFactory.get_optimal_model_for_task("categorization")
        self.logger = get_logger(__name__)

        # Load config values
        self.batch_size = config.performance.extraction_batch_size
        self.temperature = config.performance.fact_extraction_temperature
        self.categorization_temp = config.performance.categorization_temperature
        self.max_concurrent_llm_calls = config.performance.max_concurrent_llm_calls

        # Initialize rate limiter for LLM API calls
        self.rate_limiter = get_llm_rate_limiter(max_concurrent=self.max_concurrent_llm_calls)


    def execute(self, state: Dict) -> Dict:
        """
        Extract facts from raw_search_results.

        Args:
            state: Current ResearchState dict with:
                - raw_search_results: List[SearchResult]
                - target_name: str
                - collected_facts: List[Dict] (append new facts)

        Returns:
            Updated state with new facts added to collected_facts
        """
        raw_results = state.get("raw_search_results", [])
        target_name = state.get("target_name", "")
        collected_facts = state.get("collected_facts", [])

        if not raw_results:
            self.logger.warning("No search results to extract from")
            return state

        if not target_name:
            self.logger.error("No target_name provided")
            return state

        self.logger.debug(
            f"Extracting facts from {len(raw_results)} search results for '{target_name}'"
        )

        # Extract facts with batch processing
        new_facts = self._batch_extract_facts(raw_results, target_name)

        # Filter low-quality facts
        filtered_facts = self._filter_low_quality_facts(new_facts)

        self.logger.info(
            f"Extracted {len(filtered_facts)} facts "
            f"(filtered out {len(new_facts) - len(filtered_facts)} low-quality)"
        )

        # Set new_facts for current iteration (for downstream optimization)
        state["new_facts"] = filtered_facts

        # Add to collected facts (accumulate across iterations)
        collected_facts.extend(filtered_facts)
        state["collected_facts"] = collected_facts

        self.logger.debug(
            f"Total accumulated facts: {len(collected_facts)} "
            f"(+{len(filtered_facts)} new this iteration)"
        )

        # Log extraction metrics
        self._log_extraction_metrics(filtered_facts)

        return state

    def _batch_extract_facts(
        self,
        results: List[SearchResult],
        target_name: str
    ) -> List[Dict]:
        """
        Extract facts from multiple results efficiently with concurrent batch processing.

        Processes batches in parallel using ThreadPoolExecutor and RateLimiter.

        Args:
            results: List of search results
            target_name: Name of the research target

        Returns:
            List of extracted facts
        """
        all_facts = []
        batch_size = self.batch_size  # Use config value

        # Create batches
        batches = [results[i:i + batch_size] for i in range(0, len(results), batch_size)]

        self.logger.debug(
            f"Processing {len(results)} results in {len(batches)} batches "
            f"(batch_size={batch_size}) with concurrency={self.max_concurrent_llm_calls}"
        )

        # Process batches concurrently with rate limiting
        try:
            with ThreadPoolExecutor(max_workers=self.max_concurrent_llm_calls) as executor:
                future_to_batch = {
                    executor.submit(
                        self._extract_single_batch,
                        batch,
                        batch_idx + 1,
                        target_name
                    ): batch_idx
                    for batch_idx, batch in enumerate(batches)
                }

                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_facts = future.result(timeout=120)
                        all_facts.extend(batch_facts)
                        self.logger.debug(
                            f"Batch {batch_idx + 1}/{len(batches)}: Extracted {len(batch_facts)} facts"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Batch {batch_idx + 1} failed: {e}",
                            exc_info=True
                        )
                        continue

        except Exception as e:
            self.logger.error(f"Concurrent batch extraction failed: {e}", exc_info=True)
            # Fallback to sequential processing
            self.logger.warning("Falling back to sequential batch processing")
            for batch_idx, batch in enumerate(batches):
                try:
                    batch_facts = self._extract_single_batch(batch, batch_idx + 1, target_name)
                    all_facts.extend(batch_facts)
                except Exception as e:
                    self.logger.error(f"Sequential batch {batch_idx + 1} failed: {e}")
                    continue

        return all_facts

    def _extract_single_batch(
        self,
        batch: List[SearchResult],
        batch_num: int,
        target_name: str
    ) -> List[Dict]:
        """
        Extract facts from a single batch with rate limiting.

        Args:
            batch: List of search results in this batch
            batch_num: Batch number for logging
            target_name: Name of the research target

        Returns:
            List of extracted facts from this batch
        """
        batch_facts = []

        try:
            # Build batch prompt
            results_text = self._format_results_for_batch(batch)

            system_prompt = EXTRACTION_SYSTEM_PROMPT.format(target_name=target_name)
            user_prompt = BATCH_EXTRACTION_PROMPT.format(
                num_results=len(batch),
                target_name=target_name,
                results_text=results_text
            )

            # Use rate limiter to control concurrent API calls
            with self.rate_limiter.acquire():
                response = self.client.generate(
                    prompt=user_prompt,
                    system_instruction=system_prompt,
                    temperature=self.temperature
                )

            extracted_data = parse_json_object(response)
            raw_facts = extracted_data.get("facts", [])

            # Normalize and enrich facts
            for fact in raw_facts:
                # Ensure required fields exist
                if "content" not in fact:
                    continue

                # Add metadata
                fact.setdefault("category", "biographical")
                fact.setdefault("confidence", 0.5)
                fact.setdefault("entities", {})
                fact.setdefault("extracted_date", datetime.now())

                # Normalize entities
                fact["entities"] = self._normalize_entities(fact.get("entities", {}))

                # Adjust preliminary confidence based on source
                source_domain = fact.get("source_domain", "")

                if source_domain:
                    confidence_boost = self._assign_preliminary_confidence(
                        fact,
                        source_domain
                    )
                    fact["confidence"] = min(1.0, fact["confidence"] + confidence_boost)

                batch_facts.append(fact)

        except Exception as e:
            self.logger.error(f"Batch {batch_num} extraction failed: {e}", exc_info=True)

        return batch_facts

    def _format_results_for_batch(self, results: List[SearchResult]) -> str:
        """
        Format search results for batch processing.

        Args:
            results: List of search results

        Returns:
            Formatted text for batch extraction
        """
        formatted = []

        for idx, result in enumerate(results, 1):
            formatted.append(f"""
                --- Result {idx} ---
                Title: {result.title}
                URL: {result.url}
                Domain: {result.source_domain}
                Content: {result.content[:1000]}  # Limit content length
                """)

        return "\n".join(formatted)

    def _assign_preliminary_confidence(
        self,
        fact: Dict,
        source_domain: str
    ) -> float:
        """
        Assign preliminary confidence boost based on source quality.

        Args:
            fact: Fact dictionary
            source_domain: Source domain name

        Returns:
            Confidence boost value (0.0 to 0.2)
        """
        boost = 0.0

        # Check for authoritative TLDs
        for tld, score in self.AUTHORITATIVE_DOMAINS.items():
            if source_domain.endswith(f".{tld}"):
                boost += score
                break

        # Check for high-quality news
        if source_domain in self.HIGH_QUALITY_NEWS:
            boost += 0.1

        # Penalize low-quality sources
        for pattern in self.LOW_QUALITY_PATTERNS:
            if pattern in source_domain.lower():
                boost -= 0.1
                break

        # Ensure boost is in valid range
        return max(-0.2, min(0.2, boost))

    def _normalize_entities(self, entities: Dict) -> Dict:
        """
        Normalize extracted entities (dates, amounts, names).

        Args:
            entities: Dictionary of entity lists

        Returns:
            Normalized entity dictionary
        """
        normalized = {
            "people": [],
            "companies": [],
            "locations": [],
            "dates": []
        }

        # Normalize people names
        if "people" in entities:
            for name in entities["people"]:
                if isinstance(name, str) and name.strip():
                    # Title case for names
                    normalized["people"].append(name.strip().title())

        # Normalize companies
        if "companies" in entities:
            for company in entities["companies"]:
                if isinstance(company, str) and company.strip():
                    normalized["companies"].append(company.strip())

        # Normalize locations
        if "locations" in entities:
            for location in entities["locations"]:
                if isinstance(location, str) and location.strip():
                    normalized["locations"].append(location.strip())

        # Normalize dates
        if "dates" in entities:
            for date in entities["dates"]:
                if isinstance(date, str) and date.strip():
                    normalized["dates"].append(date.strip())

        # Remove duplicates
        for key in normalized:
            normalized[key] = list(set(normalized[key]))

        return normalized

    def _filter_low_quality_facts(self, facts: List[Dict]) -> List[Dict]:
        """
        Filter out vague, redundant, or low-quality facts using Gemini Flash AI.

        Falls back to keyword-based filtering if AI fails.

        Quality criteria:
        - Fact must have substantive content (> 20 chars)
        - Confidence must be reasonable (> 0.25)
        - Must not be too vague or generic
        - Must not be duplicate

        Args:
            facts: List of facts to filter

        Returns:
            Filtered list of high-quality facts
        """
        if not facts:
            return []

        # Basic pre-filtering (too short, too low confidence)
        prefiltered = []
        for fact in facts:
            content = fact.get("content", "").strip()
            if len(content) >= 20 and fact.get("confidence", 0) >= 0.25:
                prefiltered.append(fact)

        if not prefiltered:
            return []

        # Try AI-powered quality filtering
        try:
            return self._filter_facts_with_ai(prefiltered)
        except Exception as e:
            self.logger.warning(f"AI quality filtering failed: {e}, using keyword fallback")
            return self._filter_facts_keyword_fallback(prefiltered)

    def _filter_facts_with_ai(self, facts: List[Dict]) -> List[Dict]:
        """Use Gemini Flash to evaluate fact quality in batches with concurrent execution."""

        # Process in batches of 10 for efficiency
        batch_size = 10
        batches = [facts[i:i + batch_size] for i in range(0, len(facts), batch_size)]

        self.logger.info(
            f"AI filtering {len(facts)} facts in {len(batches)} batches "
            f"with concurrency={self.max_concurrent_llm_calls}"
        )

        all_filtered = []

        # Process batches concurrently with rate limiting
        try:
            with ThreadPoolExecutor(max_workers=self.max_concurrent_llm_calls) as executor:
                future_to_batch = {
                    executor.submit(
                        self._filter_single_batch_with_ai,
                        batch,
                        batch_idx + 1
                    ): batch_idx
                    for batch_idx, batch in enumerate(batches)
                }

                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_filtered = future.result(timeout=120)
                        all_filtered.extend(batch_filtered)
                        self.logger.debug(
                            f"Filter batch {batch_idx + 1}/{len(batches)}: "
                            f"Kept {len(batch_filtered)} facts"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Filter batch {batch_idx + 1} failed: {e}",
                            exc_info=True
                        )
                        # On failure, keep all facts in this batch to be safe
                        all_filtered.extend(batches[batch_idx])
                        continue

        except Exception as e:
            self.logger.error(f"Concurrent AI filtering failed: {e}", exc_info=True)
            # Fallback to sequential processing
            self.logger.warning("Falling back to sequential AI filtering")
            for batch_idx, batch in enumerate(batches):
                try:
                    batch_filtered = self._filter_single_batch_with_ai(batch, batch_idx + 1)
                    all_filtered.extend(batch_filtered)
                except Exception as e:
                    self.logger.error(f"Sequential filter batch {batch_idx + 1} failed: {e}")
                    # Keep all facts on failure
                    all_filtered.extend(batch)
                    continue

        # Deduplicate across all batches
        seen_content = set()
        final_filtered = []
        for fact in all_filtered:
            content_normalized = fact.get("content", "").lower().strip()
            if content_normalized not in seen_content:
                seen_content.add(content_normalized)
                final_filtered.append(fact)

        self.logger.info(f"AI filtering: {len(facts)} → {len(final_filtered)} facts")
        return final_filtered

    def _filter_single_batch_with_ai(self, batch: List[Dict], batch_num: int) -> List[Dict]:
        """Filter a single batch of facts using AI with rate limiting."""

        # Create numbered list for AI
        facts_text = "\n".join([
            f"{idx + 1}. {fact['content']}"
            for idx, fact in enumerate(batch)
        ])

        prompt = f"""Evaluate each fact for quality. Return indices of facts to KEEP.

                REJECT if fact is:
                - Vague or generic (e.g., "is well-known", "has experience", "is successful")
                - Not verifiable or too broad
                - Redundant with others in the list
                - Contains only metadata (e.g., "according to sources", "it is reported that")

                KEEP if fact is:
                - Specific and concrete
                - Verifiable
                - Contains unique information
                - Actionable for due diligence

                Facts to evaluate:
                {facts_text}

                Return JSON array of indices to KEEP: [1, 3, 5, ...]
                ONLY return the JSON array, nothing else."""

        # Use rate limiter to control concurrent API calls
        with self.rate_limiter.acquire():
            response = self.flash_client.generate(
                prompt=prompt,
                temperature=self.categorization_temp,
            )

        # Parse AI response
        keep_indices = parse_json_array(response, fallback=[])

        # Filter batch based on AI decisions
        filtered_batch = []
        for idx in keep_indices:
            # Convert 1-indexed to 0-indexed
            fact_idx = idx - 1
            if 0 <= fact_idx < len(batch):
                filtered_batch.append(batch[fact_idx])

        return filtered_batch

    def _filter_facts_keyword_fallback(self, facts: List[Dict]) -> List[Dict]:
        """Fallback keyword-based quality filtering."""
        filtered = []
        seen_content = set()

        vague_patterns = [
            r"^(he|she|they) (is|are|was|were) (well-known|famous|successful)",
            r"^(the|a|an) (person|individual|man|woman)",
            r"^according to (sources|reports)",
            r"^it is (said|reported|believed) that",
        ]

        for fact in facts:
            content = fact.get("content", "").strip()

            # Skip if vague
            if any(re.match(pattern, content.lower()) for pattern in vague_patterns):
                continue

            # Skip if duplicate
            content_normalized = content.lower().strip()
            if content_normalized in seen_content:
                continue

            seen_content.add(content_normalized)
            filtered.append(fact)

        return filtered

    def _log_extraction_metrics(self, facts: List[Dict]) -> None:
        """
        Log extraction metrics for monitoring.

        Args:
            facts: List of extracted facts
        """
        if not facts:
            return

        # Count by category
        categories = defaultdict(int)
        for fact in facts:
            category = fact.get("category", "unknown")
            categories[category] += 1

        # Calculate average confidence
        avg_confidence = sum(f.get("confidence", 0) for f in facts) / len(facts)

        # Count unique sources
        unique_sources = len(set(f.get("source_domain", "") for f in facts))

        self.logger.info(
            f"Extraction metrics: "
            f"total={len(facts)}, "
            f"avg_confidence={avg_confidence:.2f}, "
            f"unique_sources={unique_sources}, "
            f"categories={dict(categories)}"
        )
