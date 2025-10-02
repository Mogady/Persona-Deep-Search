"""
Query Planner Node - Generates intelligent search queries using progressive strategy.

This node is responsible for creating search queries that adapt based on the research
iteration and previously discovered information.
"""

from typing import Dict, List, Set, Any
from datetime import datetime
import json
import re
import numpy as np
from collections import defaultdict

from src.tools.models import ModelFactory
from src.utils.logger import get_logger
from src.prompts.templates.planner_prompt import (
    BROAD_DISCOVERY_PROMPT,
    TARGETED_INVESTIGATION_PROMPT,
    CONNECTION_MINING_PROMPT,
    VALIDATION_PROMPT,
    get_prompt_for_iteration
)


class QueryPlannerNode:
    """
    Generates intelligent search queries using progressive search strategy.
    Uses Gemini Flash 2.5 for fast query generation.
    """

    def __init__(self):
        """Initialize the query planner with Gemini Flash client."""
        self.client = ModelFactory.get_optimal_model_for_task("query_generation")
        self.logger = get_logger(__name__)
        self.logger.info("Initialized QueryPlannerNode with Gemini Flash")

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

        # Ensure we have at least 3 queries and at most 5
        if len(queries) < 3:
            self.logger.warning(
                f"Only {len(queries)} queries generated, adding fallback queries"
            )
            queries = self._add_fallback_queries(queries, target_name, current_iteration)

        queries = queries[:5]  # Limit to 5

        self.logger.info(f"Generated {len(queries)} queries for iteration {current_iteration}")

        # Update state
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
            response = self.client.generate(prompt)
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
        facts_summary = self._create_facts_summary(facts[:10])  # Top 10 facts

        prompt = TARGETED_INVESTIGATION_PROMPT.format(
            target_name=target_name,
            facts_summary=facts_summary,
            people=", ".join(entities.get("people", [])[:5]),
            companies=", ".join(entities.get("companies", [])[:5]),
            locations=", ".join(entities.get("locations", [])[:3]),
            explored_topics=", ".join(list(explored_topics)[:10])
        )

        try:
            response = self.client.generate(prompt)
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
            people=", ".join(entities.get("people", [])[:8]),
            companies=", ".join(entities.get("companies", [])[:8])
        )

        try:
            response = self.client.generate(prompt)
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
            response = self.client.generate(prompt)
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
            self.logger.warning(f"Embedding-based filtering failed, using word overlap: {e}")
            return self._filter_with_word_overlap(queries, previous_queries)

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
                self.logger.debug(f"Filtering exact duplicate: {query}")
                continue

            # Calculate cosine similarity with all previous queries
            query_vec = query_vecs[i]
            similarities = self._cosine_similarity(query_vec, prev_vecs)

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

    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2_array: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between a vector and an array of vectors.

        Args:
            vec1: Single embedding vector
            vec2_array: Array of embedding vectors

        Returns:
            Array of similarity scores (0.0 to 1.0)
        """
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)

        # Handle empty array
        if len(vec2_array) == 0:
            return np.array([])

        vec2_norms = vec2_array / np.linalg.norm(vec2_array, axis=1, keepdims=True)

        # Calculate cosine similarity
        similarities = np.dot(vec2_norms, vec1_norm)

        return similarities

    def _filter_with_word_overlap(
        self,
        queries: List[str],
        previous_queries: List[str]
    ) -> List[str]:
        """
        Fallback filter using word overlap (Jaccard similarity).

        Args:
            queries: Candidate queries
            previous_queries: Previous query texts

        Returns:
            Filtered list of unique queries
        """
        filtered = []

        for query in queries:
            query_lower = query.lower()

            # Check for exact match
            if query_lower in [pq.lower() for pq in previous_queries]:
                self.logger.debug(f"Filtering duplicate query: {query}")
                continue

            # Check for high similarity (simple word overlap)
            is_similar = False
            query_words = set(query_lower.split())

            for prev_query in previous_queries:
                prev_words = set(prev_query.lower().split())
                overlap = len(query_words & prev_words)
                total = len(query_words | prev_words)

                if total > 0 and overlap / total > 0.7:  # 70% similarity threshold
                    self.logger.debug(f"Filtering similar query: {query}")
                    is_similar = True
                    break

            if not is_similar:
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
            # Combine fact contents (limit to prevent huge prompts)
            combined_text = "\n".join([
                f.get("content", "") for f in facts[:20]  # Top 20 facts
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

    def _create_facts_summary(self, facts: List[Dict]) -> str:
        """
        Create a concise summary of facts.

        Args:
            facts: List of facts

        Returns:
            Summary string
        """
        if not facts:
            return "No facts discovered yet"

        summary_lines = []
        for i, fact in enumerate(facts[:10], 1):
            content = fact.get("content", "")[:100]  # Truncate long facts
            summary_lines.append(f"{i}. {content}")

        return "\n".join(summary_lines)

    def _parse_queries_from_response(self, response: str) -> List[str]:
        """
        Parse queries from AI response (expects JSON format).
        Handles markdown code blocks, raw JSON, and fallback to line parsing.

        Args:
            response: AI model response

        Returns:
            List of extracted queries
        """
        if not response or not response.strip():
            self.logger.warning("Empty response from LLM")
            return []

        try:
            # Strategy 1: Try to extract JSON from markdown code blocks
            markdown_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            markdown_match = re.search(markdown_pattern, response, re.DOTALL)

            if markdown_match:
                json_str = markdown_match.group(1)
                try:
                    data = json.loads(json_str)
                    queries = data.get("queries", [])

                    if isinstance(queries, list) and queries:
                        cleaned = [q.strip() for q in queries if q and q.strip()]
                        self.logger.debug(f"Extracted {len(cleaned)} queries from markdown JSON")
                        return cleaned
                except json.JSONDecodeError:
                    self.logger.debug("Failed to parse JSON from markdown block")

            # Strategy 2: Try to find raw JSON in response
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    queries = data.get("queries", [])

                    if isinstance(queries, list) and queries:
                        cleaned = [q.strip() for q in queries if q and q.strip()]
                        self.logger.debug(f"Extracted {len(cleaned)} queries from raw JSON")
                        return cleaned
                except json.JSONDecodeError:
                    self.logger.debug("Failed to parse raw JSON")

            # Strategy 3: Fallback - extract queries from numbered/bulleted lines
            lines = response.split('\n')
            queries = []
            for line in lines:
                line = line.strip()
                # Look for numbered lists, bullets, or quoted strings
                if re.match(r'^\d+\.', line) or line.startswith('-') or line.startswith('*'):
                    # Remove numbering/bullets
                    query = re.sub(r'^[\d\-\*\.]+\s*', '', line).strip()
                    # Remove quotes if present
                    query = query.strip('"\'')
                    if query and len(query) > 5:  # Minimum query length
                        queries.append(query)

            if queries:
                self.logger.debug(f"Extracted {len(queries)} queries from text lines")
                return queries[:5]

            self.logger.warning("Could not extract queries from response")
            return []

        except Exception as e:
            self.logger.error(f"Error parsing queries: {e}")
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
