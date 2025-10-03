"""
Anthropic Claude client for complex reasoning and analysis tasks.

This module provides a robust client for Claude models with retry logic,
structured output support, and comprehensive logging.
"""

import json
import re
from typing import List, Dict, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import anthropic

from src.utils.logger import get_logger
from src.utils.json_parser import parse_json_object
from src.utils.config import get_config


class AnthropicClient:
    """
    Client for Anthropic Claude models.

    Handles complex reasoning tasks including:
    - Risk assessment
    - Report generation
    - Complex analysis
    - Strategic reasoning
    """

    def __init__(self, api_key: str, model_name: str, max_retries: int = None):
        """
        Initialize the Anthropic Claude client.

        Args:
            api_key: Anthropic API key for authentication
            model_name: Name of the Claude model to use (e.g., claude-sonnet-4-5-20250929)
            max_retries: Maximum number of retry attempts (default: from config)
        """
        self.config = get_config()
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries or self.config.performance.api_retry_attempts
        self.logger = get_logger(__name__)

        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)

        self.logger.info(f"Initialized AnthropicClient with model: {self.model_name}, max_retries: {self.max_retries}")

    @retry(
        stop=stop_after_attempt(3),  # Will be made dynamic in _call_api_with_retry
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _call_api_with_retry(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 4096
    ) -> Any:
        """
        Call Anthropic API with retry logic.

        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction
            temperature: Temperature for generation (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (default: 4096)

        Returns:
            API response object

        Raises:
            anthropic.RateLimitError: Rate limit exceeded
            anthropic.AuthenticationError: Authentication failed
            Exception: Other API errors
        """
        try:
            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # Call API with required max_tokens parameter
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_instruction if system_instruction else anthropic.NOT_GIVEN,
                messages=messages
            )

            return response

        except anthropic.RateLimitError as e:
            self.logger.warning("Rate limit hit, retrying...")
            raise
        except anthropic.AuthenticationError as e:
            self.logger.error("Authentication failed - check API key")
            raise
        except Exception as e:
            self.logger.error(f"API error: {e}")
            raise

    def _log_token_usage(self, response: Any, operation: str) -> None:
        """
        Extract and log token counts from API response.

        Args:
            response: API response object
            operation: Name of the operation (for logging)
        """
        try:
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                self.logger.info(
                    f"{operation} token usage",
                    extra={
                        "operation": operation,
                        "model": self.model_name,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.input_tokens + usage.output_tokens
                    }
                )
        except Exception as e:
            self.logger.warning(f"Could not extract token usage: {e}")

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Basic text generation.

        Args:
            prompt: The prompt to generate from
            system_instruction: Optional system instruction to guide the model
            temperature: Temperature for generation (default: from config)
            max_tokens: Maximum tokens to generate (default: from config)

        Returns:
            str: Generated text

        Raises:
            Exception: If generation fails
        """
        # Use config defaults if not provided
        if temperature is None:
            temperature = 1.0  # Default for general generation
        if max_tokens is None:
            max_tokens = self.config.performance.default_max_tokens

        try:
            self.logger.info(
                "Generating text with Claude",
                extra={
                    "prompt_length": len(prompt),
                    "has_system_instruction": system_instruction is not None,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )

            response = self._call_api_with_retry(
                prompt,
                system_instruction,
                temperature,
                max_tokens
            )

            # Log token usage
            self._log_token_usage(response, "generate")

            # Extract text from response
            generated_text = response.content[0].text

            self.logger.info(
                "Text generation complete",
                extra={"response_length": len(generated_text)}
            )

            return generated_text

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise

    def generate_structured(
        self,
        prompt: str,
        schema: dict,
        system_instruction: Optional[str] = None,
    ) -> dict:
        """
        Generate JSON matching a schema.

        Args:
            prompt: The prompt to generate from
            schema: JSON schema to validate against
            system_instruction: Optional system instruction

        Returns:
            dict: Parsed and validated JSON object

        Raises:
            ValueError: If response doesn't match schema
        """
        try:
            self.logger.info(
                "Generating structured output with Claude",
                extra={
                    "prompt_length": len(prompt),
                    "schema_keys": list(schema.keys()) if isinstance(schema, dict) else None
                }
            )

            # Add JSON instruction to prompt
            json_prompt = f"{prompt}\n\nRespond with valid JSON only. Use this format:\n```json\n{json.dumps(schema, indent=2)}\n```"

            response = self._call_api_with_retry(
                json_prompt,
                system_instruction,
                temperature=self.config.performance.validation_temperature,  # Lower temp for structured output
                max_tokens=self.config.performance.structured_output_max_tokens
            )

            # Log token usage
            self._log_token_usage(response, "generate_structured")

            # Parse JSON from response
            result = parse_json_object(response.content[0].text)

            # Basic schema validation (check required keys)
            if isinstance(schema, dict):
                for key in schema.keys():
                    if key not in result:
                        self.logger.warning(f"Missing key in response: {key}")

            self.logger.info("Structured generation complete")

            return result

        except Exception as e:
            self.logger.error(f"Structured generation failed: {e}")
            raise

    def analyze_risk(
        self,
        facts: List[dict],
        target_name: str
    ) -> List[dict]:
        """
        Analyze facts for potential risks.

        Identifies risks in categories:
        - legal: Lawsuits, regulatory actions, criminal charges
        - financial: Bankruptcy, fraud, payment defaults
        - reputational: Scandals, controversies, negative media
        - compliance: Regulatory violations, sanctions
        - behavioral: Job-hopping patterns, unexplained gaps

        Args:
            facts: List of fact dictionaries to analyze
            target_name: Name of the research target

        Returns:
            List[dict]: List of identified risks with severity and evidence
        """
        try:
            self.logger.info(
                "Analyzing risks with Claude",
                extra={
                    "target_name": target_name,
                    "facts_count": len(facts)
                }
            )

            # Build facts summary
            facts_text = "\n".join([f"- {fact.get('content', '')}" for fact in facts])

            prompt = f"""Analyze the following facts about "{target_name}" and identify potential risks.

Facts:
{facts_text}

For each risk, identify:
1. Severity: Low, Medium, High, or Critical
2. Category: legal, financial, reputational, compliance, or behavioral
3. Description: Clear explanation of the risk
4. Evidence: List of supporting facts
5. Confidence: Score from 0.0 to 1.0

Respond with JSON in this format:
```json
{{
    "risks": [
        {{
            "severity": "High",
            "category": "legal",
            "description": "Ongoing lawsuit regarding contract dispute",
            "evidence": ["Fact 1", "Fact 2"],
            "confidence": 0.85
        }}
    ]
}}
```"""

            system_instruction = """You are an expert risk analyst specializing in due diligence investigations.
Analyze facts objectively and identify genuine risk indicators. Be conservative with severity ratings -
only mark as Critical if there's clear evidence of serious issues."""

            response = self._call_api_with_retry(
                prompt,
                system_instruction,
                temperature=self.config.performance.risk_analysis_temperature,
                max_tokens=self.config.performance.default_max_tokens
            )

            # Log token usage
            self._log_token_usage(response, "analyze_risk")

            # Parse JSON from response
            result = parse_json_object(response.content[0].text)

            risks = result.get("risks", [])

            self.logger.info(
                "Risk analysis complete",
                extra={"risks_count": len(risks)}
            )

            return risks

        except Exception as e:
            self.logger.error(f"Risk analysis failed: {e}")
            # Return empty list on error to allow graceful degradation
            return []

    def generate_report(
        self,
        target_name: str,
        facts: List[dict],
        connections: List[dict],
        risks: List[dict],
        metadata: dict
    ) -> str:
        """
        Generate comprehensive markdown report.

        Args:
            target_name: Name of the research target
            facts: List of discovered facts
            connections: List of mapped connections
            risks: List of identified risks
            metadata: Research session metadata (iteration count, source diversity, etc.)

        Returns:
            str: Markdown-formatted report
        """
        try:
            self.logger.info(
                "Generating report with Claude",
                extra={
                    "target_name": target_name,
                    "facts_count": len(facts),
                    "connections_count": len(connections),
                    "risks_count": len(risks)
                }
            )

            # Build context
            facts_by_category = {}
            for fact in facts:
                category = fact.get("category", "other")
                if category not in facts_by_category:
                    facts_by_category[category] = []
                facts_by_category[category].append(fact)

            prompt = f"""Generate a comprehensive due diligence report for "{target_name}".

Use the following information:

FACTS ({len(facts)} total):
{json.dumps(facts_by_category, indent=2)}

CONNECTIONS ({len(connections)} total):
{json.dumps(connections, indent=2)}

RISKS ({len(risks)} total):
{json.dumps(risks, indent=2)}

METADATA:
{json.dumps(metadata, indent=2)}

Create a professional markdown report with these sections:
1. Executive Summary (3-4 sentences highlighting key findings)
2. Subject Overview (basic biographical/professional information)
3. Key Facts (organized by category with confidence indicators)
4. Risk Assessment (all identified risks with severity and evidence)
5. Network Analysis (important connections and relationships)
6. Timeline (chronological view of major events)
7. Source Summary (domains used, source diversity metrics)
8. Confidence Assessment (overall reliability of findings)
9. Recommendations (suggested follow-up investigations)

Use markdown formatting:
- Bold for key names and terms
- Tables for structured data
- Confidence indicators: 🟢 High (0.8+), 🟡 Medium (0.5-0.8), 🔴 Low (<0.5)
- Professional, objective tone"""

            system_instruction = """You are an expert due diligence analyst creating reports for
executive stakeholders. Write clearly, professionally, and objectively. Focus on facts and
evidence. Use confidence indicators consistently."""

            response = self._call_api_with_retry(
                prompt,
                system_instruction,
                temperature=self.config.performance.report_generation_temperature,
                max_tokens=self.config.performance.report_max_tokens  # Reports need more tokens
            )

            # Log token usage
            self._log_token_usage(response, "generate_report")

            report = response.content[0].text

            self.logger.info(
                "Report generation complete",
                extra={"report_length": len(report)}
            )

            return report

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise
