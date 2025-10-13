"""
Unified Gemini client supporting all Gemini models.

This module provides a flexible client that can work with any Gemini model
by accepting the model name as a parameter.
"""

import json
from typing import List, Dict, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai
from google.api_core import exceptions

from src.utils.logger import get_logger
from src.utils.json_parser import parse_json_object
from src.utils.config import get_config


class GeminiClient:
    """
    Unified client for all Gemini models.

    Supports various Gemini models including:
    - gemini-2.0-pro-latest (complex reasoning, extraction, analysis)
    - gemini-2.5-flash-latest (fast queries, filtering)
    - Any other Gemini model variants
    """

    def __init__(self, api_key: str, model_name: str, max_retries: int = None):
        """
        Initialize the Gemini client.

        Args:
            api_key: Google API key for authentication
            model_name: Name of the Gemini model to use
            max_retries: Maximum number of retry attempts (default: from config)
        """
        self.config = get_config()
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries or self.config.performance.api_retry_attempts
        self.logger = get_logger(__name__)

        # Initialize Gemini client
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

        self.logger.debug(f"Initialized GeminiClient with model: {self.model_name}, max_retries: {self.max_retries}")

    @retry(
        stop=stop_after_attempt(3),  # Will be made dynamic in _call_api_with_retry
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
            self.logger.debug(
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

            # Extract text from response
            generated_text = response.text

            self.logger.debug(
                "Text generation complete",
                extra={"response_length": len(generated_text)}
            )

            return generated_text

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise

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
            self.logger.debug(f"Generating embeddings for {len(texts)} texts")

            # Use the embedding model (gemini-embedding-001)
            embeddings = []

            results = genai.embed_content(
                model="models/embedding-001",
                content=[texts],
                task_type=task_type
            )
            embeddings= [res for res in results["embedding"]]

            self.logger.debug(
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
        Extract named entities using advanced understanding.
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
            self.logger.debug("Extracting entities using advanced NER")

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

            # Parse JSON response (using unified parser)
            entities = parse_json_object(response.text)

            # Validate structure
            if not isinstance(entities, dict):
                entities = {"people": [], "companies": [], "locations": []}

            # Ensure all keys exist
            for key in ["people", "companies", "locations"]:
                if key not in entities or not isinstance(entities[key], list):
                    entities[key] = []

            self.logger.debug(
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
