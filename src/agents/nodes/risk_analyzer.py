from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
from src.tools.models import ModelFactory
from src.utils.logger import get_logger
from src.utils.config import Config
from src.database.repository import ResearchRepository
from src.utils.json_parser import parse_json_array
from src.prompts.templates.risk_analyzer_prompt import RISK_ANALYSIS_PROMPT
import json
import numpy as np
from src.utils.similarity import cosine_similarity

class RiskAnalyzerNode:
    """
    Analyzes validated facts for potential risks.
    Includes JSON parsing, validation, deduplication, and confidence calibration.
    """

    # Valid severity levels
    VALID_SEVERITIES = {"Low", "Medium", "High", "Critical"}

    # Valid risk categories
    VALID_CATEGORIES = {"Legal", "Financial", "Reputational", "Compliance", "Behavioral"}

    def __init__(self, config: Config, repository: ResearchRepository):
        """
        Initialize the risk analyzer.

        Args:
            config: Configuration object with all settings
            repository: Database repository for persistence
        """
        self.config = config
        self.repository = repository
        self.client = ModelFactory.get_optimal_model_for_task("risk_analysis")
        self.logger = get_logger(__name__)

        # Get client for embeddings (for semantic deduplication)
        self.gemini_client = ModelFactory.get_optimal_model_for_task("extraction")

        # Load config values
        self.temperature = config.performance.risk_analysis_temperature
        self.max_concurrent_llm_calls = config.performance.max_concurrent_llm_calls

    def execute(self, state: Dict) -> Dict:
        """
        Analyzes NEW facts for risks and adds them to the state.

        Args:
            state: Current ResearchState with:
                - new_facts: List[Dict] (newly extracted facts in current iteration)
                - collected_facts: List[Dict] (all accumulated facts for context)

        Returns:
            Updated state with new 'risk_flags' appended.
        """
        self.logger.info("Executing Risk Analyzer Node")

        new_facts = state.get('new_facts', [])
        collected_facts = state.get('collected_facts', [])

        if not new_facts:
            self.logger.info("No new facts to analyze for risks in this iteration.")
            if 'risk_flags' not in state:
                state['risk_flags'] = []
            return state

        self.logger.debug(
            f"Analyzing {len(new_facts)} new facts for risks "
            f"(total accumulated: {len(collected_facts)})"
        )

        # Serialize new facts with datetime handling
        facts_serializable = self._make_facts_serializable(new_facts)
        prompt = RISK_ANALYSIS_PROMPT.format(facts=json.dumps(facts_serializable, indent=2))

        try:

            response = self.client.generate(prompt)

            # Parse JSON with markdown extraction
            risks = parse_json_array(response)

            if not risks:
                self.logger.warning("No risks found in LLM response.")
                if 'risk_flags' not in state:
                    state['risk_flags'] = []
                return state

            # Validate and clean risks
            valid_risks = []
            for risk in risks:
                validated_risk = self._validate_risk_structure(risk)
                if validated_risk:
                    valid_risks.append(validated_risk)

            self.logger.debug(f"Validated {len(valid_risks)} out of {len(risks)} risks.")

            # Calibrate confidence based on evidence
            calibrated_risks = [self._calibrate_confidence(risk, new_facts) for risk in valid_risks]

            # Deduplicate risks
            unique_risks = self._deduplicate_risks(calibrated_risks)

            # Auto-adjust severity based on evidence and confidence
            final_risks = [self._adjust_severity(risk) for risk in unique_risks]

            # Initialize risk_flags if not present
            if 'risk_flags' not in state:
                state['risk_flags'] = []

            state['risk_flags'].extend(final_risks)

            # Log comprehensive metrics
            self._log_metrics(final_risks)

            self.logger.info(f"Added {len(final_risks)} unique risks (deduplicated from {len(valid_risks)}).")

            # Save risk flags to database
            session_id = state.get("session_id")
            if session_id and final_risks:
                try:
                    # Convert risk flags to database format (lowercase enums for DB)
                    db_risks = []
                    for risk in final_risks:
                        db_risks.append({
                            "session_id": session_id,
                            "severity": risk["severity"].lower(),
                            "category": risk["category"].lower(),
                            "description": risk["description"],
                            "evidence": risk["evidence"],
                            "confidence": risk["confidence"],
                            "recommended_follow_up": risk.get("recommended_follow_up", "")
                        })

                    # Batch save to database
                    saved_count = self.repository.save_risk_flags_batch(db_risks)
                    self.logger.debug(f"Saved {saved_count}/{len(db_risks)} risk flags to database")

                except Exception as e:
                    self.logger.error(f"Failed to save risk flags to database: {e}", exc_info=True)
                    # Don't fail the workflow if DB save fails

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from risk analysis response: {e}")
            if 'risk_flags' not in state:
                state['risk_flags'] = []
        except Exception as e:
            self.logger.error(f"An error occurred during risk analysis: {e}")
            if 'risk_flags' not in state:
                state['risk_flags'] = []

        return state

    def _validate_risk_structure(self, risk: Dict) -> Optional[Dict]:
        """
        Validate risk structure and required fields.

        Args:
            risk: Risk dictionary to validate

        Returns:
            Validated risk dict or None if invalid
        """
        # Required fields
        required_fields = ['severity', 'category', 'description', 'evidence', 'confidence']

        for field in required_fields:
            if field not in risk:
                self.logger.warning(f"Risk missing required field '{field}': {risk}")
                return None

        # Validate and normalize severity (keep capitalized for internal use)
        if risk['severity'].lower().title() not in self.VALID_SEVERITIES:
            self.logger.warning(f"Invalid severity '{risk['severity']}'. Defaulting to 'Medium'.")
            risk['severity'] = 'Medium'
        # Keep capitalized internally - will be lowercased when saving to DB

        # Validate and normalize category (keep capitalized for internal use)
        if risk['category'].lower().title() not in self.VALID_CATEGORIES:
            self.logger.warning(f"Invalid category '{risk['category']}'. Defaulting to 'Reputational'.")
            risk['category'] = 'Reputational'
        # Keep capitalized internally - will be lowercased when saving to DB

        # Validate confidence
        try:
            confidence = float(risk['confidence'])
            if not (0.0 <= confidence <= 1.0):
                self.logger.warning(f"Confidence {confidence} out of range. Clamping to [0.0, 1.0].")
                risk['confidence'] = max(0.0, min(1.0, confidence))
            else:
                risk['confidence'] = confidence
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid confidence value '{risk['confidence']}'. Defaulting to 0.5.")
            risk['confidence'] = 0.5

        # Validate evidence
        if not isinstance(risk['evidence'], list) or len(risk['evidence']) == 0:
            self.logger.warning(f"Risk has empty or invalid evidence: {risk}")
            return None

        # Validate description
        if not risk['description'] or not isinstance(risk['description'], str):
            self.logger.warning(f"Risk has invalid description: {risk}")
            return None

        # Add default values for optional fields
        if 'recommended_follow_up' not in risk:
            risk['recommended_follow_up'] = "Further investigation recommended."

        # Add timestamp
        risk['identified_at'] = datetime.utcnow().isoformat()

        return risk

    def _calibrate_confidence(self, risk: Dict, facts: List[Dict]) -> Dict:
        """
        Calibrate risk confidence based on evidence quality and quantity.

        Args:
            risk: Risk dictionary
            facts: All collected facts

        Returns:
            Risk with calibrated confidence
        """
        base_confidence = risk['confidence']
        evidence_count = len(risk['evidence'])

        # Create fact lookup for efficient matching
        fact_contents = {fact.get('content', ''): fact for fact in facts}

        # Find matching facts for evidence
        evidence_facts = []
        for evidence_text in risk['evidence']:
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
        risk['confidence'] = max(0.0, min(1.0, final_confidence))

        self.logger.debug(f"Calibrated risk confidence: {base_confidence:.2f} -> {risk['confidence']:.2f}")

        return risk

    def _deduplicate_risks(self, risks: List[Dict]) -> List[Dict]:
        """
        Deduplicate risks using semantic similarity.

        Args:
            risks: List of risk dictionaries

        Returns:
            Deduplicated list of risks
        """
        if len(risks) <= 1:
            return risks

        # Extract descriptions for similarity comparison
        descriptions = [risk['description'] for risk in risks]

        try:
            # Use Gemini embeddings for semantic similarity
            embeddings = self.gemini_client.generate_embeddings(descriptions)

            if embeddings and len(embeddings) == len(risks):
                # Calculate cosine similarity matrix
                n_risks = len(embeddings)
                similarity_matrix = np.zeros((n_risks, n_risks))
                embeddings_array = np.array(embeddings)
                for i in range(n_risks):
                    similarity_matrix[i] = cosine_similarity(embeddings_array[i], embeddings_array)

                # Group similar risks (threshold: 0.85)
                groups = self._group_by_similarity(similarity_matrix, threshold=0.85)

                # Merge duplicates within each group
                unique_risks = []
                for group in groups:
                    merged_risk = self._merge_risks([risks[i] for i in group])
                    unique_risks.append(merged_risk)

                dedup_count = len(risks) - len(unique_risks)
                if dedup_count > 0:
                    self.logger.info(f"Deduplicated {dedup_count} similar risks using embeddings.")

                return unique_risks

        except Exception as e:
            self.logger.warning(f"Embeddings-based deduplication failed: {e}. Using fallback.")

        # Fallback: Simple text-based deduplication
        return self._deduplicate_risks_simple(risks)

    def _deduplicate_risks_simple(self, risks: List[Dict]) -> List[Dict]:
        """
        Fallback deduplication using simple text matching.

        Args:
            risks: List of risk dictionaries

        Returns:
            Deduplicated list of risks
        """
        seen_descriptions = set()
        unique_risks = []

        for risk in risks:
            desc_lower = risk['description'].lower().strip()

            # Check for exact match
            if desc_lower not in seen_descriptions:
                seen_descriptions.add(desc_lower)
                unique_risks.append(risk)
            else:
                self.logger.debug(f"Removed duplicate risk: {risk['description'][:50]}...")

        return unique_risks

    def _group_by_similarity(self, similarity_matrix: np.ndarray, threshold: float = 0.85) -> List[List[int]]:
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

    def _merge_risks(self, risks: List[Dict]) -> Dict:
        """
        Merge duplicate risks, combining evidence and keeping highest confidence.

        Args:
            risks: List of duplicate risks to merge

        Returns:
            Merged risk dictionary
        """
        if len(risks) == 1:
            return risks[0]

        # Start with the first risk as base
        merged = risks[0].copy()

        # Combine all evidence
        all_evidence = []
        for risk in risks:
            all_evidence.extend(risk['evidence'])

        # Deduplicate evidence
        merged['evidence'] = list(set(all_evidence))

        # Keep highest confidence
        merged['confidence'] = max(risk['confidence'] for risk in risks)

        # Keep most severe severity
        severity_order = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
        merged['severity'] = max(risks, key=lambda r: severity_order.get(r['severity'], 0))['severity']

        return merged

    def _adjust_severity(self, risk: Dict) -> Dict:
        """
        Auto-adjust severity based on evidence count and confidence.

        Args:
            risk: Risk dictionary

        Returns:
            Risk with adjusted severity
        """
        evidence_count = len(risk['evidence'])
        confidence = risk['confidence']
        current_severity = risk['severity']

        severity_order = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        severity_names = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Critical'}

        current_level = severity_order[current_severity]

        # Escalate for high confidence + multiple evidence
        if confidence >= 0.8 and evidence_count >= 3 and current_level < 4:
            new_level = min(current_level + 1, 4)
            risk['severity'] = severity_names[new_level]
            self.logger.debug(f"Escalated risk severity: {current_severity} -> {risk['severity']}")

        # Downgrade for low confidence
        elif confidence < 0.4 and current_level > 1:
            new_level = max(current_level - 1, 1)
            risk['severity'] = severity_names[new_level]
            self.logger.debug(f"Downgraded risk severity: {current_severity} -> {risk['severity']}")

        return risk

    def _log_metrics(self, risks: List[Dict]):
        """
        Log comprehensive risk analysis metrics.

        Args:
            risks: List of final risks
        """
        if not risks:
            self.logger.info("Risk Metrics: No risks identified.")
            return

        # Count by severity
        severity_counts = defaultdict(int)
        for risk in risks:
            severity_counts[risk['severity']] += 1

        # Count by category
        category_counts = defaultdict(int)
        for risk in risks:
            category_counts[risk['category']] += 1

        # Calculate average confidence
        avg_confidence = sum(r['confidence'] for r in risks) / len(risks)

        # Calculate evidence statistics
        total_evidence = sum(len(r['evidence']) for r in risks)
        avg_evidence = total_evidence / len(risks)

        # Log metrics
        self.logger.info(f"=== Risk Analysis Metrics ===")
        self.logger.info(f"Total Risks: {len(risks)}")
        self.logger.info(f"By Severity: {dict(severity_counts)}")
        self.logger.info(f"By Category: {dict(category_counts)}")
        self.logger.info(f"Average Confidence: {avg_confidence:.2f}")
        self.logger.info(f"Average Evidence per Risk: {avg_evidence:.1f}")
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
