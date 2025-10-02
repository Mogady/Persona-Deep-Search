"""
Content Extractor Node - Extracts structured facts from search results.

This node processes raw search results and extracts atomic, verified facts
using Gemini Pro's advanced NER and extraction capabilities.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
import re

from src.tools.models import ModelFactory
from src.utils.logger import get_logger
from src.tools.search.models import SearchResult
from src.prompts.templates.extractor_prompt import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT,
    BATCH_EXTRACTION_PROMPT
)


class ContentExtractorNode:
    """
    Extracts structured facts from raw search results.
    Uses Gemini Pro 2.5 for precise extraction with entity recognition.
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

    def __init__(self):
        """Initialize the content extractor with Gemini Pro."""
        self.client = ModelFactory.get_optimal_model_for_task("extraction")  # Gemini Pro
        self.logger = get_logger(__name__)
        self.logger.info("Initialized ContentExtractorNode with Gemini Pro")

    def execute(self, state: Dict) -> Dict:
        """
        Extract facts from raw_search_results.

        Args:
            state: Current ResearchState dict with:
                - raw_search_results: List[SearchResult] from Agent 4
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

        self.logger.info(
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

        # Add to collected facts
        collected_facts.extend(filtered_facts)
        state["collected_facts"] = collected_facts

        # Log extraction metrics
        self._log_extraction_metrics(filtered_facts)

        return state

    def _batch_extract_facts(
        self,
        results: List[SearchResult],
        target_name: str
    ) -> List[Dict]:
        """
        Extract facts from multiple results efficiently (batch processing).

        Processes 5-10 results per API call to reduce costs.

        Args:
            results: List of search results
            target_name: Name of the research target

        Returns:
            List of extracted facts
        """
        all_facts = []
        batch_size = 5  # Process 5 results per API call

        # Process in batches
        for i in range(0, len(results), batch_size):
            batch = results[i:i + batch_size]

            try:
                # Build batch prompt
                results_text = self._format_results_for_batch(batch)

                system_prompt = EXTRACTION_SYSTEM_PROMPT.format(target_name=target_name)
                user_prompt = BATCH_EXTRACTION_PROMPT.format(
                    num_results=len(batch),
                    target_name=target_name,
                    results_text=results_text
                )

                # Call Gemini Pro
                response = self.client.generate(
                    prompt=user_prompt,
                    system_instruction=system_prompt,
                    temperature=0.3  # Low temperature for consistency
                )

                # Parse JSON response
                extracted_data = self.client._parse_json_from_markdown(response)
                batch_facts = extracted_data.get("facts", [])

                self.logger.info(
                    f"Batch {i // batch_size + 1}: "
                    f"Extracted {len(batch_facts)} facts from {len(batch)} results"
                )

                # Normalize and enrich facts
                for fact in batch_facts:
                    # Ensure required fields exist
                    if "content" not in fact:
                        continue

                    # Add metadata
                    fact.setdefault("category", "biographical")
                    fact.setdefault("confidence", 0.5)
                    fact.setdefault("entities", {})
                    fact.setdefault("extracted_date", datetime.utcnow())

                    # Normalize entities
                    fact["entities"] = self._normalize_entities(fact.get("entities", {}))

                    # Adjust preliminary confidence based on source
                    source_url = fact.get("source_url", "")
                    source_domain = fact.get("source_domain", "")

                    if source_domain:
                        confidence_boost = self._assign_preliminary_confidence(
                            fact,
                            source_domain
                        )
                        fact["confidence"] = min(1.0, fact["confidence"] + confidence_boost)

                    all_facts.append(fact)

            except Exception as e:
                self.logger.error(f"Batch extraction failed for batch {i // batch_size + 1}: {e}")
                # Continue with next batch
                continue

        return all_facts

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

    def _extract_facts_from_result(
        self,
        result: SearchResult,
        target_name: str
    ) -> List[Dict]:
        """
        Extract facts from a single search result.

        This is a fallback method if batch processing fails.

        Args:
            result: Single search result
            target_name: Name of the research target

        Returns:
            List of extracted facts
        """
        try:
            system_prompt = EXTRACTION_SYSTEM_PROMPT.format(target_name=target_name)
            user_prompt = EXTRACTION_USER_PROMPT.format(
                target_name=target_name,
                title=result.title,
                url=result.url,
                domain=result.source_domain,
                content=result.content[:2000]  # Limit content
            )

            # Call Gemini Pro
            response = self.client.generate(
                prompt=user_prompt,
                system_instruction=system_prompt,
                temperature=0.3
            )

            # Parse JSON response
            extracted_data = self.client._parse_json_from_markdown(response)
            facts = extracted_data.get("facts", [])

            # Enrich facts with metadata
            for fact in facts:
                fact["source_url"] = result.url
                fact["source_domain"] = result.source_domain
                fact["extracted_date"] = datetime.utcnow()

                # Normalize entities
                fact["entities"] = self._normalize_entities(fact.get("entities", {}))

            return facts

        except Exception as e:
            self.logger.error(f"Failed to extract from {result.url}: {e}")
            return []

    def _categorize_fact(self, fact_content: str) -> str:
        """
        Categorize fact as biographical, professional, financial, or behavioral.

        Uses keyword matching as a fallback if LLM doesn't categorize.

        Args:
            fact_content: The fact statement

        Returns:
            Category string
        """
        content_lower = fact_content.lower()

        # Keyword-based categorization
        if any(kw in content_lower for kw in ["born", "birth", "family", "married", "education", "graduated", "degree"]):
            return "biographical"
        elif any(kw in content_lower for kw in ["ceo", "cto", "president", "director", "manager", "founder", "employee", "worked", "career"]):
            return "professional"
        elif any(kw in content_lower for kw in ["salary", "compensation", "investment", "revenue", "profit", "stock", "shares", "$", "million", "billion"]):
            return "financial"
        elif any(kw in content_lower for kw in ["statement", "said", "claimed", "alleged", "controversy", "scandal", "lawsuit", "accused"]):
            return "behavioral"
        else:
            return "biographical"  # Default

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
        Filter out vague, redundant, or low-quality facts.

        Quality criteria:
        - Fact must have substantive content (> 20 chars)
        - Confidence must be reasonable (> 0.2)
        - Must not be too vague or generic
        - Must not be duplicate

        Args:
            facts: List of facts to filter

        Returns:
            Filtered list of facts
        """
        filtered = []
        seen_content = set()

        # Vague patterns to reject
        vague_patterns = [
            r"^(he|she|they) (is|are|was|were) (well-known|famous|successful)",
            r"^(the|a|an) (person|individual|man|woman)",
            r"^according to (sources|reports)",
            r"^it is (said|reported|believed) that",
        ]

        for fact in facts:
            content = fact.get("content", "").strip()

            # Skip if too short
            if len(content) < 20:
                continue

            # Skip if confidence too low
            if fact.get("confidence", 0) < 0.25:
                continue

            # Skip if vague
            if any(re.match(pattern, content.lower()) for pattern in vague_patterns):
                self.logger.debug(f"Filtered vague fact: {content[:50]}...")
                continue

            # Skip if duplicate (case-insensitive)
            content_normalized = content.lower().strip()
            if content_normalized in seen_content:
                continue

            # Add to filtered list
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
