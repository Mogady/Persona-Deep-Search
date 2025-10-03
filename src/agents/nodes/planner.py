"""
Query Planner Node - Generates intelligent search queries using progressive strategy.

This node is responsible for creating search queries that adapt based on the research
iteration and previously discovered information.
"""

from typing import Dict, List, Set
import re
import numpy as np

from src.tools.models import ModelFactory
from src.utils.logger import get_logger
from src.utils.similarity import cosine_similarity
from src.utils.config import Config
from src.utils.json_parser import parse_json_array
from src.database.repository import ResearchRepository
from src.prompts.templates.planner_prompt import (
    BROAD_DISCOVERY_PROMPT,
    TARGETED_INVESTIGATION_PROMPT,
    CONNECTION_MINING_PROMPT,
    VALIDATION_PROMPT,
)

class QueryPlannerNode:
    """
    Generates intelligent search queries using progressive search strategy.
    Uses Gemini Flash 2.5 for fast query generation.
    """

    def __init__(self, config: Config, repository: ResearchRepository):
        """
        Initialize the query planner with Gemini Flash client.

        Args:
            config: Configuration object with all settings
            repository: Database repository for persistence
        """
        self.config = config
        self.repository = repository
        self.client = ModelFactory.get_optimal_model_for_task("query_generation")
        self.logger = get_logger(__name__)

        self.min_queries = config.performance.min_queries_per_iteration
        self.max_queries = config.performance.max_queries_per_iteration
        self.temperature = config.performance.query_generation_temperature

    def execute(self, state: Dict) -> Dict:
        """
        Main node execution method.

        Args:
            state: Current ResearchState dict with:
                - target_name: str
                - current_iteration: int
                - collected_facts: List[Dict]
                - search_history: List[Dict]
                - explored_topics: Set[str]

        Returns:
            Updated state with next_queries: List[str] (3-5 queries)
        """
        target_name = state.get("target_name", "")
        current_iteration = state.get("current_iteration", 1)
        collected_facts = state.get("collected_facts", [])
        search_history = state.get("search_history", [])
        explored_topics = state.get("explored_topics", set())

        self.logger.info(
            f"Planning queries for iteration {current_iteration}, "
            f"target: {target_name}, "
            f"facts collected: {len(collected_facts)}"
        )

        # Generate queries based on iteration
        if current_iteration == 1:
            queries = self._generate_broad_queries(target_name)
        elif current_iteration in [2, 3]:
            queries = self._generate_targeted_queries(target_name, collected_facts, explored_topics)
        elif current_iteration in [4, 5]:
            queries = self._generate_connection_queries(target_name, collected_facts)
        else:  # 6, 7
            queries = self._generate_validation_queries(target_name, collected_facts)

        # Filter out duplicate queries
        queries = self._filter_duplicate_queries(queries, search_history)

        # Ensure we have min queries and don't exceed max
        if len(queries) < self.min_queries:
            self.logger.warning(
                f"Only {len(queries)} queries generated, adding fallback queries (min: {self.min_queries})"
            )
            queries = self._add_fallback_queries(queries, target_name, current_iteration)

        queries = queries[:self.max_queries]  # Limit to max configured

        self.logger.info(f"Generated {len(queries)} queries for iteration {current_iteration}")

        state["next_queries"] = queries

        return state

    def _generate_broad_queries(self, target_name: str) -> List[str]:
        """
        Iteration 1: Broad discovery queries.

        Args:
            target_name: Name of research target

        Returns:
            List of broad discovery queries
        """
        self.logger.info(f"Generating broad discovery queries for: {target_name}")

        prompt = BROAD_DISCOVERY_PROMPT.format(target_name=target_name)

        try:
            response = self.client.generate(prompt, temperature=self.temperature)
            queries = self._parse_queries_from_response(response)

            if not queries:
                # Fallback to hardcoded broad queries
                queries = [
                    f'"{target_name}" professional background',
                    f'"{target_name}" company employment history',
                    f'"{target_name}" news',
                    f'"{target_name}" LinkedIn profile',
                    f'"{target_name}" education'
                ]

            self.logger.info(f"Generated {len(queries)} broad queries")
            return queries

        except Exception as e:
            self.logger.error(f"Error generating broad queries: {e}")
            # Return fallback queries
            return [
                f'"{target_name}" professional background',
                f'"{target_name}" company employment',
                f'"{target_name}" news'
            ]

    def _generate_targeted_queries(
        self,
        target_name: str,
        facts: List[Dict],
        explored_topics: Set[str]
    ) -> List[str]:
        """
        Iterations 2-3: Build on discovered facts.

        Args:
            target_name: Name of research target
            facts: Previously collected facts
            explored_topics: Already-explored topics

        Returns:
            List of targeted investigation queries
        """
        self.logger.info(f"Generating targeted queries for: {target_name}")

        # Extract entities from facts
        entities = self._extract_entities(facts)

        # Create facts summary
        facts_summary = "\n".join(facts[:10])  # Top 10 facts

        prompt = TARGETED_INVESTIGATION_PROMPT.format(
            target_name=target_name,
            facts_summary=facts_summary,
            people=", ".join(entities.get("people", [])),
            companies=", ".join(entities.get("companies", [])),
            locations=", ".join(entities.get("locations", [])),
            explored_topics=", ".join(list(explored_topics)[:10])
        )

        try:
            response = self.client.generate(prompt, temperature=self.temperature)
            queries = self._parse_queries_from_response(response)

            if not queries and entities.get("companies"):
                # Fallback: use discovered companies
                company = entities["companies"][0]
                queries = [
                    f'"{target_name}" "{company}" role',
                    f'"{target_name}" "{company}" timeline',
                    f'"{target_name}" "{company}" achievements'
                ]

            self.logger.info(f"Generated {len(queries)} targeted queries")
            return queries

        except Exception as e:
            self.logger.error(f"Error generating targeted queries: {e}")
            return []

    def _generate_connection_queries(
        self,
        target_name: str,
        facts: List[Dict]
    ) -> List[str]:
        """
        Iterations 4-5: Deep connection mining.

        Args:
            target_name: Name of research target
            facts: Previously collected facts

        Returns:
            List of connection mining queries
        """
        self.logger.info(f"Generating connection queries for: {target_name}")

        # Extract entities from facts
        entities = self._extract_entities(facts)

        prompt = CONNECTION_MINING_PROMPT.format(
            target_name=target_name,
            people=", ".join(entities.get("people", [])),
            companies=", ".join(entities.get("companies", []))
        )

        try:
            response = self.client.generate(prompt, temperature=self.temperature)
            queries = self._parse_queries_from_response(response)

            if not queries:
                # Fallback connection queries
                queries = [
                    f'"{target_name}" board member',
                    f'"{target_name}" business partner',
                    f'"{target_name}" investments'
                ]

            self.logger.info(f"Generated {len(queries)} connection queries")
            return queries

        except Exception as e:
            self.logger.error(f"Error generating connection queries: {e}")
            return []

    def _generate_validation_queries(
        self,
        target_name: str,
        facts: List[Dict]
    ) -> List[str]:
        """
        Iterations 6-7: Verify low-confidence facts.

        Args:
            target_name: Name of research target
            facts: Previously collected facts

        Returns:
            List of validation queries
        """
        self.logger.info(f"Generating validation queries for: {target_name}")

        # Find low-confidence facts (< 0.7)
        low_confidence_facts = [
            f for f in facts
            if f.get("confidence_score", 1.0) < 0.7
        ][:5]

        if not low_confidence_facts:
            # If no low-confidence facts, verify recent facts
            low_confidence_facts = facts[-5:] if facts else []

        facts_text = "\n".join([
            f"- {f.get('content', '')}" for f in low_confidence_facts
        ])

        prompt = VALIDATION_PROMPT.format(
            target_name=target_name,
            low_confidence_facts=facts_text
        )

        try:
            response = self.client.generate(prompt, temperature=self.temperature)
            queries = self._parse_queries_from_response(response)

            if not queries and low_confidence_facts:
                # Fallback: create verification queries from facts
                first_fact = low_confidence_facts[0].get("content", "")
                queries = [
                    f'"{target_name}" {first_fact[:50]} verification',
                    f'"{target_name}" authoritative source'
                ]

            self.logger.info(f"Generated {len(queries)} validation queries")
            return queries

        except Exception as e:
            self.logger.error(f"Error generating validation queries: {e}")
            return []

    def _filter_duplicate_queries(
        self,
        queries: List[str],
        search_history: List[Dict]
    ) -> List[str]:
        """
        Remove queries similar to previous searches using semantic similarity.
        Uses Gemini embeddings for cosine similarity calculation.

        Args:
            queries: Candidate queries
            search_history: Previous search queries

        Returns:
            Filtered list of unique queries
        """
        if not search_history:
            return queries

        # Extract previous query texts
        previous_queries = [
            q.get("query", "") for q in search_history
        ]

        try:
            # Use semantic similarity with embeddings
            filtered = self._filter_with_embeddings(queries, previous_queries)
            self.logger.info(
                f"Filtered {len(queries) - len(filtered)} duplicate/similar queries "
                f"using semantic similarity"
            )
            return filtered

        except Exception as e:
            # Fallback to word overlap if embedding fails
            self.logger.warning(f"Embedding-based filtering failed, cancel filtering: {e}")
            return queries

    def _filter_with_embeddings(
        self,
        queries: List[str],
        previous_queries: List[str]
    ) -> List[str]:
        """
        Filter queries using cosine similarity with Gemini embeddings.

        Args:
            queries: Candidate queries
            previous_queries: Previous query texts

        Returns:
            Filtered list of unique queries
        """
        if not previous_queries:
            return queries

        # Generate embeddings for all queries
        all_queries = queries + previous_queries
        embeddings = self.client.generate_embeddings(
            all_queries,
            task_type="SEMANTIC_SIMILARITY"
        )

        # Split embeddings
        query_embeddings = embeddings[:len(queries)]
        prev_embeddings = embeddings[len(queries):]

        # Convert to numpy arrays for cosine similarity
        query_vecs = np.array(query_embeddings)
        prev_vecs = np.array(prev_embeddings)

        # Filter queries based on cosine similarity
        filtered = []
        similarity_threshold = 0.85  # 85% similarity threshold

        for i, query in enumerate(queries):
            # Check for exact match first
            if query.lower() in [pq.lower() for pq in previous_queries]:
                continue

            # Calculate cosine similarity with all previous queries
            query_vec = query_vecs[i]
            similarities = cosine_similarity(query_vec, prev_vecs)

            # Check if any previous query is too similar
            max_similarity = np.max(similarities) if len(similarities) > 0 else 0.0

            if max_similarity > similarity_threshold:
                self.logger.debug(
                    f"Filtering semantically similar query: {query} "
                    f"(similarity={max_similarity:.3f})"
                )
                continue

            filtered.append(query)

        return filtered

    def _extract_entities(self, facts: List[Dict]) -> Dict[str, List[str]]:
        """
        Extract people, companies, locations from facts using AI-powered NER.
        Falls back to regex if AI extraction fails.

        Args:
            facts: List of facts

        Returns:
            Dict with entity lists (people, companies, locations)
        """
        if not facts:
            return {"people": [], "companies": [], "locations": []}

        try:
            # Combine fact contents
            combined_text = "\n".join([
                f.get("content", "") for f in facts
            ])

            # Use Gemini's advanced entity extraction
            entities = self.client.extract_entities_advanced(combined_text)

            # Limit results
            return {
                "people": entities.get("people", [])[:10],
                "companies": entities.get("companies", [])[:10],
                "locations": entities.get("locations", [])[:10]
            }

        except Exception as e:
            self.logger.warning(f"AI entity extraction failed, falling back to regex: {e}")
            return self._extract_entities_regex(facts)

    def _extract_entities_regex(self, facts: List[Dict]) -> Dict[str, List[str]]:
        """
        Fallback regex-based entity extraction.

        Args:
            facts: List of facts

        Returns:
            Dict with entity lists (people, companies, locations)
        """
        entities = {
            "people": set(),
            "companies": set(),
            "locations": set()
        }

        # Simple entity extraction from fact content
        for fact in facts:
            content = fact.get("content", "")
            category = fact.get("category", "")

            # Extract based on category
            if category == "professional":
                # Look for company names (capitalized words, often with Inc, Corp, LLC)
                company_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Corp|LLC|Ltd|Group|Partners))?)\b'
                companies = re.findall(company_pattern, content)
                entities["companies"].update(companies)

            # Extract person names (simplified - capitalized words pattern)
            name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
            names = re.findall(name_pattern, content)
            entities["people"].update(names)

            # Extract locations (capitalized single/double words, common city/country names)
            location_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
            if any(word in content.lower() for word in ['city', 'country', 'state', 'located', 'based']):
                locations = re.findall(location_pattern, content)
                entities["locations"].update(locations)

        # Convert to sorted lists
        return {
            "people": sorted(list(entities["people"]))[:10],
            "companies": sorted(list(entities["companies"]))[:10],
            "locations": sorted(list(entities["locations"]))[:10]
        }


    def _parse_queries_from_response(self, response: str) -> List[str]:
        """
        Parse queries from AI response.
        Expects strict JSON array format: ["query1", "query2", ...]
        """
        try:

            queries = parse_json_array(response, fallback=[])

            cleaned = []
            for q in queries:
                if isinstance(q, str) and len(q.strip()) > 10:
                    cleaned.append(q.strip())

            self.logger.debug(f"Parsed {len(cleaned)} queries from response")
            return cleaned

        except Exception as e:
            self.logger.error(f"Failed to parse queries: {e}")
            return []

    def _add_fallback_queries(
        self,
        queries: List[str],
        target_name: str,
        iteration: int
    ) -> List[str]:
        """
        Add fallback queries when not enough queries generated.

        Args:
            queries: Current queries list
            target_name: Research target
            iteration: Current iteration

        Returns:
            Extended queries list
        """
        fallback_templates = {
            1: [
                f'"{target_name}" biography',
                f'"{target_name}" career',
                f'"{target_name}" profile'
            ],
            2: [
                f'"{target_name}" professional experience',
                f'"{target_name}" work history'
            ],
            3: [
                f'"{target_name}" achievements',
                f'"{target_name}" projects'
            ],
            4: [
                f'"{target_name}" connections',
                f'"{target_name}" network'
            ],
            5: [
                f'"{target_name}" partnerships',
                f'"{target_name}" collaborations'
            ],
            6: [
                f'"{target_name}" official source',
                f'"{target_name}" verified information'
            ],
            7: [
                f'"{target_name}" authoritative profile',
                f'"{target_name}" official biography'
            ]
        }

        fallback = fallback_templates.get(iteration, [
            f'"{target_name}" information',
            f'"{target_name}" details'
        ])

        # Add fallback queries that aren't already in the list
        for fb in fallback:
            if fb not in queries and len(queries) < 5:
                queries.append(fb)

        return queries
