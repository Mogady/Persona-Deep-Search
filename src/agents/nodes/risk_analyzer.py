from typing import Dict, List, Optional, Set
from datetime import datetime
from collections import defaultdict
from src.tools.models import ModelFactory
from src.utils.logger import get_logger
from src.prompts.templates.risk_analyzer_prompt import RISK_ANALYSIS_PROMPT
import json
import re
import numpy as np

class RiskAnalyzerNode:
    """
    Analyzes validated facts for potential risks using Claude Sonnet.
    Includes JSON parsing, validation, deduplication, and confidence calibration.
    """

    # Valid severity levels
    VALID_SEVERITIES = {"Low", "Medium", "High", "Critical"}

    # Valid risk categories
    VALID_CATEGORIES = {"Legal", "Financial", "Reputational", "Compliance", "Behavioral"}

    def __init__(self):
        self.client = ModelFactory.get_optimal_model_for_task("risk_analysis") # Claude Sonnet
        self.logger = get_logger(__name__)

        # Get Gemini client for embeddings (for semantic deduplication)
        self.gemini_client = ModelFactory.get_optimal_model_for_task("extraction")

    def execute(self, state: Dict) -> Dict:
        """
        Analyzes facts for risks and adds them to the state.

        Args:
            state: Current ResearchState with 'collected_facts'.

        Returns:
            Updated state with 'risk_flags' populated.
        """
        self.logger.info("Executing Risk Analyzer Node")

        facts = state.get('collected_facts', [])
        if not facts:
            self.logger.info("No facts to analyze for risks.")
            if 'risk_flags' not in state:
                state['risk_flags'] = []
            return state

        prompt = RISK_ANALYSIS_PROMPT.format(facts=json.dumps(facts, indent=2))

        try:
            response = self.client.generate(prompt)

            # Parse JSON with markdown extraction
            risks = self._parse_json_from_markdown(response)

            if not risks:
                self.logger.warning("No risks found in LLM response.")
                if 'risk_flags' not in state:
                    state['risk_flags'] = []
                return state

            # Validate and clean risks
            valid_risks = []
            for risk in risks:
                validated_risk = self._validate_risk_structure(risk, facts)
                if validated_risk:
                    valid_risks.append(validated_risk)

            self.logger.info(f"Validated {len(valid_risks)} out of {len(risks)} risks.")

            # Calibrate confidence based on evidence
            calibrated_risks = [self._calibrate_confidence(risk, facts) for risk in valid_risks]

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

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from risk analysis response: {e}")
            if 'risk_flags' not in state:
                state['risk_flags'] = []
        except Exception as e:
            self.logger.error(f"An error occurred during risk analysis: {e}")
            if 'risk_flags' not in state:
                state['risk_flags'] = []

        return state

    def _parse_json_from_markdown(self, response: str) -> List[Dict]:
        """
        Parse JSON from LLM response, handling markdown code blocks.

        Args:
            response: Raw LLM response text

        Returns:
            List of risk dictionaries
        """
        if not response or not response.strip():
            self.logger.warning("Empty response from LLM.")
            return []

        # Try to extract JSON from markdown code blocks
        json_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
        match = re.search(json_pattern, response, re.DOTALL)

        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse JSON from markdown block.")

        # Try to parse the entire response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in the response
        array_pattern = r'\[.*?\]'
        match = re.search(array_pattern, response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        self.logger.error("Could not extract valid JSON from response.")
        return []

    def _validate_risk_structure(self, risk: Dict, facts: List[Dict]) -> Optional[Dict]:
        """
        Validate risk structure and required fields.

        Args:
            risk: Risk dictionary to validate
            facts: All collected facts for reference

        Returns:
            Validated risk dict or None if invalid
        """
        # Required fields
        required_fields = ['severity', 'category', 'description', 'evidence', 'confidence']

        for field in required_fields:
            if field not in risk:
                self.logger.warning(f"Risk missing required field '{field}': {risk}")
                return None

        # Validate severity
        if risk['severity'] not in self.VALID_SEVERITIES:
            self.logger.warning(f"Invalid severity '{risk['severity']}'. Defaulting to 'Medium'.")
            risk['severity'] = 'Medium'

        # Validate category
        if risk['category'] not in self.VALID_CATEGORIES:
            self.logger.warning(f"Invalid category '{risk['category']}'. Defaulting to 'Reputational'.")
            risk['category'] = 'Reputational'

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
                similarity_matrix = self._cosine_similarity_matrix(embeddings)

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

    def _cosine_similarity_matrix(self, embeddings: List[List[float]]) -> np.ndarray:
        """Calculate cosine similarity matrix for embeddings."""
        embeddings_array = np.array(embeddings)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        normalized = embeddings_array / (norms + 1e-10)

        # Compute cosine similarity
        similarity = np.dot(normalized, normalized.T)

        return similarity

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
