from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.tools.models import ModelFactory
from src.utils.logger import get_logger
from src.utils.config import Config
from src.utils.rate_limiter import get_llm_rate_limiter
from src.database.repository import ResearchRepository
from src.utils.json_parser import parse_json_array
from src.prompts.templates.connection_mapper_prompt import CONNECTION_MAPPER_PROMPT
import json
import re
import numpy as np
from src.utils.similarity import cosine_similarity

class ConnectionMapperNode:
    """
    Maps relationships between entities found in the facts using Gemini Pro.
    Includes JSON parsing, validation, deduplication, and bidirectional handling.
    """

    # Valid relationship types
    VALID_RELATIONSHIP_TYPES = {
        "Employment",
        "Investment",
        "Board/Advisory",
        "Family/Personal",
        "Co-founder/Business partner",
        "Educational",
        "Other"
    }

    # Symmetric relationships (A→B same as B→A)
    SYMMETRIC_RELATIONSHIPS = {"Family/Personal", "Educational", "Co-founder/Business partner"}

    # Relationship type aliases for normalization
    RELATIONSHIP_ALIASES = {
        "employee": "Employment",
        "employer": "Employment",
        "investor": "Investment",
        "invested in": "Investment",
        "board member": "Board/Advisory",
        "advisor": "Board/Advisory",
        "advisory": "Board/Advisory",
        "family": "Family/Personal",
        "personal": "Family/Personal",
        "co-founder": "Co-founder/Business partner",
        "partner": "Co-founder/Business partner",
        "education": "Educational",
        "school": "Educational",
        "university": "Educational",
    }

    def __init__(self, config: Config, repository: ResearchRepository):
        """
        Initialize the connection mapper with Gemini Pro.

        Args:
            config: Configuration object with all settings
            repository: Database repository for persistence
        """
        self.config = config
        self.repository = repository
        self.client = ModelFactory.get_optimal_model_for_task("connection_mapping") # Gemini Pro
        self.logger = get_logger(__name__)

        # Get Gemini client for embeddings (for semantic deduplication)
        self.gemini_client = ModelFactory.get_optimal_model_for_task("extraction")

        # Load config values
        self.temperature = config.performance.connection_mapping_temperature
        self.max_concurrent_llm_calls = config.performance.max_concurrent_llm_calls

        # Initialize rate limiter for LLM API calls
        self.rate_limiter = get_llm_rate_limiter(max_concurrent=self.max_concurrent_llm_calls)

        self.logger.info(
            f"Initialized ConnectionMapperNode with Gemini Pro and RateLimiter "
            f"(temp: {self.temperature}, concurrency: {self.max_concurrent_llm_calls})"
        )

    def execute(self, state: Dict) -> Dict:
        """
        Extracts connections from NEW facts and adds them to the state.

        Args:
            state: Current ResearchState with:
                - new_facts: List[Dict] (newly extracted facts in current iteration)
                - collected_facts: List[Dict] (all accumulated facts for context)

        Returns:
            Updated state with new 'connections' appended.
        """
        self.logger.info("Executing Connection Mapper Node")

        new_facts = state.get('new_facts', [])
        collected_facts = state.get('collected_facts', [])

        if not new_facts:
            self.logger.info("No new facts to map connections from in this iteration.")
            if 'connections' not in state:
                state['connections'] = []
            return state

        self.logger.info(
            f"Mapping connections from {len(new_facts)} new facts "
            f"(total accumulated: {len(collected_facts)})"
        )

        # Check if new facts contain entities
        has_entities = any(
            'entities' in fact and fact.get('entities')
            for fact in new_facts
        )

        if not has_entities:
            self.logger.warning("No entities found in new facts. Skipping connection mapping.")
            if 'connections' not in state:
                state['connections'] = []
            return state

        # Serialize new facts with datetime handling
        facts_serializable = self._make_facts_serializable(new_facts)
        prompt = CONNECTION_MAPPER_PROMPT.format(facts=json.dumps(facts_serializable, indent=2))

        try:
            # Use rate limiter to control concurrent API calls
            with self.rate_limiter.acquire():
                response = self.client.generate(prompt)

            # Parse JSON with markdown extraction
            connections = parse_json_array(response)

            if not connections:
                self.logger.warning("No connections found in LLM response.")
                if 'connections' not in state:
                    state['connections'] = []
                return state

            # Validate and clean connections
            valid_connections = []
            for conn in connections:
                validated_conn = self._validate_connection_structure(conn, new_facts)
                if validated_conn:
                    valid_connections.append(validated_conn)

            self.logger.info(f"Validated {len(valid_connections)} out of {len(connections)} connections.")

            # Calibrate confidence based on evidence
            calibrated_connections = [self._calibrate_confidence(conn, new_facts) for conn in valid_connections]

            # Handle bidirectional relationships
            normalized_connections = [self._normalize_bidirectional(conn) for conn in calibrated_connections]

            # Deduplicate connections
            unique_connections = self._deduplicate_connections(normalized_connections)

            # Initialize connections if not present
            if 'connections' not in state:
                state['connections'] = []

            state['connections'].extend(unique_connections)

            # Log comprehensive metrics
            self._log_metrics(unique_connections)

            self.logger.info(f"Added {len(unique_connections)} unique connections (deduplicated from {len(valid_connections)}).")

            # Save connections to database
            session_id = state.get("session_id")
            if session_id and unique_connections:
                try:
                    # Convert connections to database format
                    db_connections = []
                    for conn in unique_connections:
                        db_connections.append({
                            "session_id": session_id,
                            "entity_a": conn["entity_a"],
                            "entity_b": conn["entity_b"],
                            "relationship_type": conn["relationship_type"],
                            "evidence": conn["evidence"],
                            "confidence": conn["confidence"]
                        })

                    # Batch save to database
                    saved_count = self.repository.save_connections_batch(db_connections)
                    self.logger.info(f"Saved {saved_count}/{len(db_connections)} connections to database")

                except Exception as e:
                    self.logger.error(f"Failed to save connections to database: {e}", exc_info=True)
                    # Don't fail the workflow if DB save fails

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from connection mapping response: {e}")
            if 'connections' not in state:
                state['connections'] = []
        except Exception as e:
            self.logger.error(f"An error occurred during connection mapping: {e}")
            if 'connections' not in state:
                state['connections'] = []

        return state


    def _validate_connection_structure(self, connection: Dict, facts: List[Dict]) -> Optional[Dict]:
        """
        Validate connection structure and required fields.

        Args:
            connection: Connection dictionary to validate
            facts: All collected facts for reference

        Returns:
            Validated connection dict or None if invalid
        """
        # Required fields
        required_fields = ['entity_a', 'entity_b', 'relationship_type', 'evidence', 'confidence']

        for field in required_fields:
            if field not in connection:
                self.logger.warning(f"Connection missing required field '{field}': {connection}")
                return None

        # Validate entities are non-empty
        if not connection['entity_a'] or not isinstance(connection['entity_a'], str):
            self.logger.warning(f"Invalid entity_a: {connection}")
            return None

        if not connection['entity_b'] or not isinstance(connection['entity_b'], str):
            self.logger.warning(f"Invalid entity_b: {connection}")
            return None

        # Check for self-connections
        if connection['entity_a'].lower().strip() == connection['entity_b'].lower().strip():
            self.logger.warning(f"Self-connection detected: {connection['entity_a']}")
            return None

        # Normalize and validate relationship type
        relationship_type = connection['relationship_type']
        normalized_type = self._normalize_relationship_type(relationship_type)

        if normalized_type not in self.VALID_RELATIONSHIP_TYPES:
            self.logger.warning(f"Invalid relationship type '{relationship_type}'. Defaulting to 'Other'.")
            connection['relationship_type'] = 'Other'
        else:
            connection['relationship_type'] = normalized_type

        # Validate confidence
        try:
            confidence = float(connection['confidence'])
            if not (0.0 <= confidence <= 1.0):
                self.logger.warning(f"Confidence {confidence} out of range. Clamping to [0.0, 1.0].")
                connection['confidence'] = max(0.0, min(1.0, confidence))
            else:
                connection['confidence'] = confidence
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid confidence value '{connection['confidence']}'. Defaulting to 0.5.")
            connection['confidence'] = 0.5

        # Validate evidence
        if not isinstance(connection['evidence'], list) or len(connection['evidence']) == 0:
            self.logger.warning(f"Connection has empty or invalid evidence: {connection}")
            return None

        # Validate or add time_period
        if 'time_period' not in connection or not connection['time_period']:
            connection['time_period'] = 'Unknown'

        # Add timestamp
        connection['identified_at'] = datetime.utcnow().isoformat()

        # Add directionality flag
        connection['is_symmetric'] = connection['relationship_type'] in self.SYMMETRIC_RELATIONSHIPS

        return connection

    def _normalize_relationship_type(self, rel_type: str) -> str:
        """
        Normalize relationship type using aliases.

        Args:
            rel_type: Raw relationship type string

        Returns:
            Normalized relationship type
        """
        rel_lower = rel_type.lower().strip()

        # Check aliases
        for alias, normalized in self.RELATIONSHIP_ALIASES.items():
            if alias in rel_lower:
                return normalized

        # Check if it's already a valid type
        for valid_type in self.VALID_RELATIONSHIP_TYPES:
            if rel_type.strip() == valid_type:
                return valid_type

        # Default to Other
        return "Other"

    def _calibrate_confidence(self, connection: Dict, facts: List[Dict]) -> Dict:
        """
        Calibrate connection confidence based on evidence quality and quantity.

        Args:
            connection: Connection dictionary
            facts: All collected facts

        Returns:
            Connection with calibrated confidence
        """
        base_confidence = connection['confidence']
        evidence_count = len(connection['evidence'])

        # Create fact lookup for efficient matching
        fact_contents = {fact.get('content', ''): fact for fact in facts}

        # Find matching facts for evidence
        evidence_facts = []
        for evidence_text in connection['evidence']:
            # Try exact match first
            if evidence_text in fact_contents:
                evidence_facts.append(fact_contents[evidence_text])
            else:
                # Try partial match
                for fact_content, fact in fact_contents.items():
                    if evidence_text in fact_content or fact_content in evidence_text:
                        evidence_facts.append(fact)
                        break

        # Calculate average fact confidence
        if evidence_facts:
            avg_fact_confidence = sum(f.get('confidence', 0.5) for f in evidence_facts) / len(evidence_facts)
        else:
            avg_fact_confidence = 0.5

        # Calculate source authority
        source_domains = [f.get('source_domain', '') for f in evidence_facts]
        authoritative_sources = sum(1 for domain in source_domains
                                    if any(ext in domain for ext in ['.gov', '.edu', '.org']))

        # Adjust confidence
        adjustment = 0.0

        # Boost for multiple evidence sources
        if evidence_count >= 3:
            adjustment += 0.2
        elif evidence_count >= 2:
            adjustment += 0.1
        else:
            # Single source penalty
            adjustment -= 0.1

        # Boost for authoritative sources
        if authoritative_sources > 0:
            adjustment += 0.1

        # Factor in fact confidence
        if avg_fact_confidence >= 0.8:
            adjustment += 0.1
        elif avg_fact_confidence < 0.5:
            adjustment -= 0.1

        # Apply adjustment
        final_confidence = base_confidence + adjustment

        # Clamp to [0.0, 1.0]
        connection['confidence'] = max(0.0, min(1.0, final_confidence))

        self.logger.debug(f"Calibrated connection confidence: {base_confidence:.2f} -> {connection['confidence']:.2f}")

        return connection

    def _normalize_bidirectional(self, connection: Dict) -> Dict:
        """
        Normalize bidirectional relationships for consistent storage.

        Args:
            connection: Connection dictionary

        Returns:
            Normalized connection
        """
        # For symmetric relationships, sort entities alphabetically
        if connection['is_symmetric']:
            entity_a = connection['entity_a']
            entity_b = connection['entity_b']

            # Normalize entity names (title case, strip)
            normalized_a = entity_a.strip().title()
            normalized_b = entity_b.strip().title()

            # Sort alphabetically
            if normalized_a > normalized_b:
                connection['entity_a'] = normalized_b
                connection['entity_b'] = normalized_a

        return connection

    def _deduplicate_connections(self, connections: List[Dict]) -> List[Dict]:
        """
        Deduplicate connections using entity pairs and semantic similarity.

        Args:
            connections: List of connection dictionaries

        Returns:
            Deduplicated list of connections
        """
        if len(connections) <= 1:
            return connections

        # First pass: Remove exact duplicates (same entities + relationship type)
        exact_dedup = self._deduplicate_exact(connections)

        # Second pass: Semantic deduplication using embeddings
        try:
            semantic_dedup = self._deduplicate_semantic(exact_dedup)
            return semantic_dedup
        except Exception as e:
            self.logger.warning(f"Semantic deduplication failed: {e}. Using exact deduplication only.")
            return exact_dedup

    def _deduplicate_exact(self, connections: List[Dict]) -> List[Dict]:
        """
        Remove exact duplicate connections (same entities + relationship type).

        Args:
            connections: List of connection dictionaries

        Returns:
            List without exact duplicates
        """
        seen = {}
        unique = []

        for conn in connections:
            # Create key from entities + relationship type
            key = self._create_connection_key(conn)

            if key not in seen:
                seen[key] = conn
                unique.append(conn)
            else:
                # Merge with existing connection
                existing = seen[key]
                merged = self._merge_connections([existing, conn])
                # Update in both dict and list
                seen[key] = merged
                # Replace in unique list
                for i, c in enumerate(unique):
                    if self._create_connection_key(c) == key:
                        unique[i] = merged
                        break

        dedup_count = len(connections) - len(unique)
        if dedup_count > 0:
            self.logger.info(f"Removed {dedup_count} exact duplicate connections.")

        return unique

    def _deduplicate_semantic(self, connections: List[Dict]) -> List[Dict]:
        """
        Deduplicate connections using semantic similarity on relationship descriptions.

        Args:
            connections: List of connection dictionaries

        Returns:
            Deduplicated list
        """
        if len(connections) <= 1:
            return connections

        # Create descriptions for similarity comparison
        descriptions = [
            f"{conn['entity_a']} - {conn['relationship_type']} - {conn['entity_b']}"
            for conn in connections
        ]

        # Generate embeddings
        embeddings = self.gemini_client.generate_embeddings(descriptions)

        if embeddings and len(embeddings) == len(connections):
            # Calculate cosine similarity matrix
            n_connections = len(embeddings)
            similarity_matrix = np.zeros((n_connections, n_connections))
            embeddings_array = np.array(embeddings)
            for i in range(n_connections):
                similarity_matrix[i] = cosine_similarity(embeddings_array[i], embeddings_array)

            # Group similar connections (threshold: 0.90 - stricter than risks)
            groups = self._group_by_similarity(similarity_matrix, threshold=0.90)

            # Merge duplicates within each group
            unique_connections = []
            for group in groups:
                merged_conn = self._merge_connections([connections[i] for i in group])
                unique_connections.append(merged_conn)

            dedup_count = len(connections) - len(unique_connections)
            if dedup_count > 0:
                self.logger.info(f"Deduplicated {dedup_count} similar connections using embeddings.")

            return unique_connections

        return connections

    def _create_connection_key(self, connection: Dict) -> Tuple[str, str, str]:
        """
        Create unique key for connection.

        Args:
            connection: Connection dictionary

        Returns:
            Tuple of (entity_a, entity_b, relationship_type)
        """
        return (
            connection['entity_a'].lower().strip(),
            connection['entity_b'].lower().strip(),
            connection['relationship_type']
        )

    def _merge_connections(self, connections: List[Dict]) -> Dict:
        """
        Merge duplicate connections, combining evidence and keeping highest confidence.

        Args:
            connections: List of duplicate connections to merge

        Returns:
            Merged connection dictionary
        """
        if len(connections) == 1:
            return connections[0]

        # Start with the first connection as base
        merged = connections[0].copy()

        # Combine all evidence
        all_evidence = []
        for conn in connections:
            all_evidence.extend(conn['evidence'])

        # Deduplicate evidence
        merged['evidence'] = list(set(all_evidence))

        # Keep highest confidence
        merged['confidence'] = max(conn['confidence'] for conn in connections)

        # Combine time periods if different
        time_periods = set(conn.get('time_period', 'Unknown') for conn in connections if conn.get('time_period'))
        if len(time_periods) > 1:
            merged['time_period'] = ' | '.join(sorted(time_periods))

        return merged

    def _cosine_similarity_matrix(self, embeddings: List[List[float]]) -> np.ndarray:
        """Calculate cosine similarity matrix for embeddings."""
        embeddings_array = np.array(embeddings)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        normalized = embeddings_array / (norms + 1e-10)

        # Compute cosine similarity
        similarity = np.dot(normalized, normalized.T)

        return similarity

    def _group_by_similarity(self, similarity_matrix: np.ndarray, threshold: float = 0.90) -> List[List[int]]:
        """Group indices by similarity threshold."""
        n = len(similarity_matrix)
        visited = set()
        groups = []

        for i in range(n):
            if i in visited:
                continue

            # Find all similar items
            group = [i]
            visited.add(i)

            for j in range(i + 1, n):
                if j not in visited and similarity_matrix[i][j] >= threshold:
                    group.append(j)
                    visited.add(j)

            groups.append(group)

        return groups

    def _log_metrics(self, connections: List[Dict]):
        """
        Log comprehensive connection mapping metrics.

        Args:
            connections: List of final connections
        """
        if not connections:
            self.logger.info("Connection Metrics: No connections identified.")
            return

        # Count by relationship type
        type_counts = defaultdict(int)
        for conn in connections:
            type_counts[conn['relationship_type']] += 1

        # Calculate average confidence
        avg_confidence = sum(c['confidence'] for c in connections) / len(connections)

        # Count unique entities
        unique_entities = set()
        for conn in connections:
            unique_entities.add(conn['entity_a'])
            unique_entities.add(conn['entity_b'])

        # Calculate evidence statistics
        total_evidence = sum(len(c['evidence']) for c in connections)
        avg_evidence = total_evidence / len(connections)

        # Calculate connection density (connections per entity)
        density = len(connections) / len(unique_entities) if unique_entities else 0

        # Log metrics
        self.logger.info(f"=== Connection Mapping Metrics ===")
        self.logger.info(f"Total Connections: {len(connections)}")
        self.logger.info(f"Unique Entities: {len(unique_entities)}")
        self.logger.info(f"By Relationship Type: {dict(type_counts)}")
        self.logger.info(f"Average Confidence: {avg_confidence:.2f}")
        self.logger.info(f"Average Evidence per Connection: {avg_evidence:.1f}")
        self.logger.info(f"Connection Density: {density:.2f}")
        self.logger.info(f"Total Evidence Items: {total_evidence}")

    def _make_facts_serializable(self, facts: List[Dict]) -> List[Dict]:
        """
        Convert facts to JSON-serializable format by handling datetime objects.

        Args:
            facts: List of fact dictionaries

        Returns:
            List of facts with datetime objects converted to ISO strings
        """
        serializable_facts = []
        for fact in facts:
            fact_copy = fact.copy()
            # Convert datetime objects to ISO strings
            for key, value in fact_copy.items():
                if isinstance(value, datetime):
                    fact_copy[key] = value.isoformat()
            serializable_facts.append(fact_copy)
        return serializable_facts
