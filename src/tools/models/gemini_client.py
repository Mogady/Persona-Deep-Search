"""
Unified Gemini client supporting all Gemini models (Pro, Flash, etc.).

This module provides a flexible client that can work with any Gemini model
by accepting the model name as a parameter.
"""

import json
import re
from typing import List, Dict, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai
from google.api_core import exceptions

from src.utils.logger import get_logger


class GeminiClient:
    """
    Unified client for all Gemini models.

    Supports various Gemini models including:
    - gemini-2.0-pro-latest (complex reasoning, extraction, analysis)
    - gemini-2.5-flash-latest (fast queries, filtering)
    - Any other Gemini model variants
    """

    def __init__(self, api_key: str, model_name: str, max_retries: int = 3):
        """
        Initialize the Gemini client.

        Args:
            api_key: Google API key for authentication
            model_name: Name of the Gemini model to use
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.logger = get_logger(__name__)

        # Initialize Gemini client
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

        self.logger.info(f"Initialized GeminiClient with model: {self.model_name}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _call_api_with_retry(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Call Gemini API with retry logic.

        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction
            generation_config: Optional generation configuration

        Returns:
            API response object

        Raises:
            exceptions.ResourceExhausted: Rate limit exceeded
            exceptions.Unauthenticated: Authentication failed
            Exception: Other API errors
        """
        try:
            # Create model with system instruction if provided
            if system_instruction:
                model = genai.GenerativeModel(
                    self.model_name,
                    system_instruction=system_instruction
                )
            else:
                model = self.model

            # Generate content
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )

            return response

        except exceptions.ResourceExhausted as e:
            self.logger.warning("Rate limit hit, retrying...")
            raise
        except exceptions.Unauthenticated as e:
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
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                self.logger.info(
                    f"{operation} token usage",
                    extra={
                        "operation": operation,
                        "model": self.model_name,
                        "input_tokens": usage.prompt_token_count,
                        "output_tokens": usage.candidates_token_count,
                        "total_tokens": usage.total_token_count
                    }
                )
        except Exception as e:
            self.logger.warning(f"Could not extract token usage: {e}")

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Basic text generation.

        Args:
            prompt: The prompt to generate from
            system_instruction: Optional system instruction to guide the model
            temperature: Optional temperature override (0.0 to 1.0)

        Returns:
            str: Generated text

        Raises:
            Exception: If generation fails
        """
        try:
            self.logger.info(
                "Generating text",
                extra={
                    "prompt_length": len(prompt),
                    "has_system_instruction": system_instruction is not None,
                    "temperature": temperature
                }
            )

            # Build generation config if temperature specified
            generation_config = None
            if temperature is not None:
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature
                )

            response = self._call_api_with_retry(
                prompt,
                system_instruction,
                generation_config
            )

            # Log token usage
            self._log_token_usage(response, "generate")

            # Extract text from response
            generated_text = response.text

            self.logger.info(
                "Text generation complete",
                extra={"response_length": len(generated_text)}
            )

            return generated_text

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise

    def _parse_json_from_markdown(self, text: str) -> dict:
        """
        Parse JSON from markdown code blocks.

        Args:
            text: Text potentially containing JSON in markdown blocks

        Returns:
            dict: Parsed JSON object

        Raises:
            ValueError: If JSON parsing fails
        """
        # Try to find JSON in markdown code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)

        if matches:
            json_text = matches[0]
        else:
            # Try to parse the entire text as JSON
            json_text = text.strip()

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            self.logger.debug(f"Text to parse: {text[:500]}")
            raise ValueError(f"Invalid JSON response: {e}")

    def generate_structured(
        self,
        prompt: str,
        schema: dict,
        system_instruction: Optional[str] = None
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
                "Generating structured output",
                extra={
                    "prompt_length": len(prompt),
                    "schema_keys": list(schema.keys()) if isinstance(schema, dict) else None
                }
            )

            # Add JSON instruction to prompt
            json_prompt = f"{prompt}\n\nRespond with valid JSON only. Use this format:\n```json\n{json.dumps(schema, indent=2)}\n```"

            response = self._call_api_with_retry(json_prompt, system_instruction)

            # Log token usage
            self._log_token_usage(response, "generate_structured")

            # Parse JSON from response
            result = self._parse_json_from_markdown(response.text)

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

    def extract_entities(self, text: str) -> List[dict]:
        """
        Named entity recognition.

        Extracts entities including:
        - person: People's names
        - organization: Company/organization names
        - location: Geographic locations
        - date: Dates and time references
        - money: Monetary values

        Args:
            text: Text to extract entities from

        Returns:
            List[dict]: List of entities with type and text
        """
        try:
            self.logger.info(
                "Extracting entities",
                extra={"text_length": len(text)}
            )

            prompt = f"""Extract all named entities from the following text.
For each entity, identify its type (person, organization, location, date, money) and the exact text.

Text: {text}

Respond with JSON in this format:
```json
{{
    "entities": [
        {{"type": "person", "text": "John Doe"}},
        {{"type": "organization", "text": "Acme Corp"}},
        {{"type": "location", "text": "New York"}},
        {{"type": "date", "text": "2020"}},
        {{"type": "money", "text": "$1M"}}
    ]
}}
```"""

            response = self._call_api_with_retry(prompt)

            # Log token usage
            self._log_token_usage(response, "extract_entities")

            # Parse JSON from response
            result = self._parse_json_from_markdown(response.text)

            entities = result.get("entities", [])

            self.logger.info(
                "Entity extraction complete",
                extra={"entities_count": len(entities)}
            )

            return entities

        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            # Return empty list on error to allow graceful degradation
            return []

    def extract_facts(self, text: str, context: Optional[str] = None) -> List[dict]:
        """
        Extract structured facts with confidence scores.

        Each fact includes:
        - content: The fact statement
        - category: biographical, professional, financial, or behavioral
        - confidence: Score from 0.0 to 1.0
        - entities: List of related entities

        Args:
            text: Text to extract facts from
            context: Optional context about the research target

        Returns:
            List[dict]: List of fact dictionaries
        """
        try:
            self.logger.info(
                "Extracting facts",
                extra={
                    "text_length": len(text),
                    "has_context": context is not None
                }
            )

            context_str = f"\nContext: {context}" if context else ""

            prompt = f"""Extract all verifiable facts from the following text.
{context_str}

For each fact:
1. State the fact clearly
2. Categorize it as: biographical, professional, financial, or behavioral
3. Assign a confidence score (0.0 to 1.0) based on how definitive the statement is
4. List any entities (people, organizations, locations) mentioned in the fact

Text: {text}

Respond with JSON in this format:
```json
{{
    "facts": [
        {{
            "content": "John Doe was CEO of Acme Corp from 2015 to 2020",
            "category": "professional",
            "confidence": 0.9,
            "entities": ["John Doe", "Acme Corp", "2015", "2020"]
        }}
    ]
}}
```"""

            response = self._call_api_with_retry(prompt)

            # Log token usage
            self._log_token_usage(response, "extract_facts")

            # Parse JSON from response
            result = self._parse_json_from_markdown(response.text)

            facts = result.get("facts", [])

            # Validate fact structure
            validated_facts = []
            for fact in facts:
                if isinstance(fact, dict) and "content" in fact:
                    # Ensure all required fields exist
                    fact.setdefault("category", "biographical")
                    fact.setdefault("confidence", 0.5)
                    fact.setdefault("entities", [])

                    # Ensure confidence is in valid range
                    if not isinstance(fact["confidence"], (int, float)):
                        fact["confidence"] = 0.5
                    else:
                        fact["confidence"] = max(0.0, min(1.0, float(fact["confidence"])))

                    validated_facts.append(fact)

            self.logger.info(
                "Fact extraction complete",
                extra={"facts_count": len(validated_facts)}
            )

            return validated_facts

        except Exception as e:
            self.logger.error(f"Fact extraction failed: {e}")
            # Return empty list on error to allow graceful degradation
            return []

    def generate_search_queries(
        self,
        target_name: str,
        context: dict,
        num_queries: int = 5
    ) -> List[str]:
        """
        Generate search queries based on current knowledge.

        Uses a progressive strategy:
        - Iteration 1: Broad discovery
        - Iterations 2-3: Targeted investigation
        - Iterations 4-5: Deep connection mining
        - Iteration 6+: Risk assessment and gap filling

        Args:
            target_name: Name of the research target
            context: Research context including:
                - collected_facts: List of facts already collected
                - explored_topics: Set of topics already explored
                - current_iteration: Current search iteration number
            num_queries: Number of queries to generate (3-5)

        Returns:
            List[str]: List of diverse search queries
        """
        try:
            iteration = context.get("current_iteration", 1)
            collected_facts = context.get("collected_facts", [])
            explored_topics = context.get("explored_topics", set())

            self.logger.info(
                "Generating search queries",
                extra={
                    "target_name": target_name,
                    "iteration": iteration,
                    "facts_count": len(collected_facts),
                    "explored_topics_count": len(explored_topics)
                }
            )

            # Build context summary
            facts_summary = ""
            if collected_facts:
                facts_summary = "\n".join([f"- {fact.get('content', '')}" for fact in collected_facts[:10]])

            explored_str = ", ".join(list(explored_topics)[:10]) if explored_topics else "None"

            # Determine search strategy based on iteration
            if iteration == 1:
                strategy = "broad discovery - basic biographical, professional background, news mentions, social media"
            elif iteration <= 3:
                strategy = "targeted investigation - company affiliations, roles, locations, education"
            elif iteration <= 5:
                strategy = "deep connection mining - board memberships, partnerships, family connections, investments"
            elif iteration == 6:
                strategy = "risk assessment - legal issues, financial irregularities, reputational concerns"
            else:
                strategy = "gap filling - verify low-confidence facts, find additional sources, resolve contradictions"

            prompt = f"""Generate {num_queries} diverse search queries to research "{target_name}".

Current iteration: {iteration}
Strategy: {strategy}

Facts collected so far:
{facts_summary if facts_summary else "None yet"}

Topics already explored: {explored_str}

Requirements:
1. Generate {num_queries} unique search queries
2. Each query should follow the strategy for iteration {iteration}
3. Avoid repeating topics already explored
4. Queries should be specific and actionable
5. Vary the query structure (biographical, investigative, contextual)

Return ONLY a JSON array of query strings:
["query 1", "query 2", "query 3"]"""

            # Use higher temperature for creativity
            generation_config = genai.types.GenerationConfig(
                temperature=0.9
            )

            response = self._call_api_with_retry(prompt, generation_config=generation_config)

            # Log token usage
            self._log_token_usage(response, "generate_search_queries")

            # Parse queries from response
            response_text = response.text.strip()

            # Try to extract JSON array
            try:
                # Remove markdown code blocks if present
                if "```" in response_text:
                    start = response_text.find("[")
                    end = response_text.rfind("]") + 1
                    if start != -1 and end > start:
                        response_text = response_text[start:end]

                queries = json.loads(response_text)

                # Ensure we got a list
                if not isinstance(queries, list):
                    raise ValueError("Response is not a list")

                # Take only requested number of queries
                queries = queries[:num_queries]

                self.logger.info(
                    "Search queries generated",
                    extra={"queries_count": len(queries)}
                )

                return queries

            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"Failed to parse JSON, extracting queries manually: {e}")

                # Fallback: extract queries line by line
                lines = response_text.split("\n")
                queries = []
                for line in lines:
                    line = line.strip()
                    # Remove common prefixes
                    line = line.lstrip("- ").lstrip("* ").lstrip("1234567890. ")
                    # Remove quotes
                    line = line.strip('"').strip("'")
                    if line and len(line) > 10:  # Reasonable query length
                        queries.append(line)

                queries = queries[:num_queries]

                if queries:
                    self.logger.info(
                        "Search queries extracted manually",
                        extra={"queries_count": len(queries)}
                    )
                    return queries
                else:
                    # Ultimate fallback: basic queries
                    self.logger.warning("Using fallback queries")
                    return [
                        f"{target_name} biography",
                        f"{target_name} career history",
                        f"{target_name} news",
                    ][:num_queries]

        except Exception as e:
            self.logger.error(f"Query generation failed: {e}")
            # Return fallback queries on error
            return [
                f"{target_name} professional background",
                f"{target_name} company affiliations",
                f"{target_name} recent news",
            ][:num_queries]

    def filter_relevant_content(
        self,
        content: str,
        target_name: str,
        threshold: float = 0.7
    ) -> bool:
        """
        Quick relevance filtering.

        Determines if content is relevant to the research target.

        Args:
            content: Content to evaluate
            target_name: Name of research target
            threshold: Relevance threshold (0.0 to 1.0)

        Returns:
            bool: True if content is relevant, False otherwise
        """
        try:
            # Truncate very long content for speed
            max_content_length = 2000
            truncated_content = content[:max_content_length]

            self.logger.info(
                "Filtering content relevance",
                extra={
                    "content_length": len(content),
                    "target_name": target_name,
                    "threshold": threshold
                }
            )

            prompt = f"""Evaluate if this content is relevant to researching "{target_name}".

Content:
{truncated_content}

Is this content relevant? Consider:
1. Does it mention {target_name} or closely related entities?
2. Does it provide biographical, professional, financial, or behavioral information?
3. Is the information substantive (not just a passing mention)?

Respond with a single word: YES or NO"""

            # Use low temperature for consistent decisions
            generation_config = genai.types.GenerationConfig(
                temperature=0.1
            )

            response = self._call_api_with_retry(prompt, generation_config=generation_config)

            # Log token usage
            self._log_token_usage(response, "filter_relevant_content")

            # Parse response
            decision = response.text.strip().upper()

            is_relevant = "YES" in decision

            self.logger.info(
                "Content filtering complete",
                extra={
                    "is_relevant": is_relevant,
                    "decision": decision
                }
            )

            return is_relevant

        except Exception as e:
            self.logger.error(f"Content filtering failed: {e}")
            # Default to True on error to avoid losing potentially valuable content
            return True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def generate_embeddings(
        self,
        texts: List[str],
        task_type: str = "SEMANTIC_SIMILARITY"
    ) -> List[List[float]]:
        """
        Generate embeddings for texts using Gemini embedding model.

        Args:
            texts: List of texts to embed
            task_type: Type of task (SEMANTIC_SIMILARITY, RETRIEVAL_QUERY, etc.)

        Returns:
            List of embedding vectors (each 768 dimensions)

        Raises:
            Exception: If embedding generation fails
        """
        try:
            self.logger.info(f"Generating embeddings for {len(texts)} texts")

            # Use the embedding model (gemini-embedding-001)
            embeddings = []

            for text in texts:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type=task_type
                )
                embeddings.append(result['embedding'])

            self.logger.info(
                f"Generated {len(embeddings)} embeddings, "
                f"dimension: {len(embeddings[0]) if embeddings else 0}"
            )

            return embeddings

        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def extract_entities_advanced(
        self,
        text: str,
        temperature: float = 0.1
    ) -> Dict[str, List[str]]:
        """
        Extract named entities using Gemini's advanced understanding.
        More accurate than regex-based extraction.

        Args:
            text: Text to extract entities from
            temperature: Temperature for generation (default: 0.1 for consistency)

        Returns:
            Dict with entity lists (people, companies, locations)

        Raises:
            Exception: If entity extraction fails
        """
        try:
            self.logger.info("Extracting entities using Gemini NER")

            prompt = f"""Extract all named entities from the following text.

Text:
{text}

Instructions:
- Extract PEOPLE: Full names of individuals (normalize capitalization)
- Extract COMPANIES: Business organizations, corporations, startups
- Extract LOCATIONS: Cities, states, countries, regions

Return ONLY valid JSON in this exact format:
{{
  "people": ["Person Name 1", "Person Name 2"],
  "companies": ["Company 1", "Company 2"],
  "locations": ["Location 1", "Location 2"]
}}

Rules:
- Normalize names (e.g., "satya nadella" → "Satya Nadella")
- Handle abbreviations (e.g., "MSFT" → "Microsoft")
- Remove duplicates
- Return empty arrays if no entities found
- Do NOT include explanations, only JSON"""

            generation_config = genai.types.GenerationConfig(
                temperature=temperature
            )

            response = self._call_api_with_retry(prompt, generation_config=generation_config)

            # Log token usage
            self._log_token_usage(response, "extract_entities_advanced")

            # Parse JSON response
            entities = self._parse_json_from_markdown(response.text)

            # Validate structure
            if not isinstance(entities, dict):
                entities = {"people": [], "companies": [], "locations": []}

            # Ensure all keys exist
            for key in ["people", "companies", "locations"]:
                if key not in entities or not isinstance(entities[key], list):
                    entities[key] = []

            self.logger.info(
                f"Extracted entities: "
                f"{len(entities['people'])} people, "
                f"{len(entities['companies'])} companies, "
                f"{len(entities['locations'])} locations"
            )

            return entities

        except Exception as e:
            self.logger.error(f"Advanced entity extraction failed: {e}")
            # Return empty structure on error
            return {"people": [], "companies": [], "locations": []}
