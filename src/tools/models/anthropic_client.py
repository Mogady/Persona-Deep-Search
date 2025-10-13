"""
Anthropic client for complex reasoning and analysis tasks.

This module provides a robust client for Anthropic models with retry logic,
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
    Client for Anthropic models.

    Handles complex reasoning tasks including:
    - Risk assessment
    - Report generation
    - Complex analysis
    - Strategic reasoning
    """

    def __init__(self, api_key: str, model_name: str, max_retries: int = None):
        """
        Initialize the Anthropic client.

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

        self.logger.debug(f"Initialized AnthropicClient with model: {self.model_name}, max_retries: {self.max_retries}")

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
            self.logger.debug(
                "Generating text with Anthropic model",
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

            # Extract text from response
            generated_text = response.content[0].text

            self.logger.debug(
                "Text generation complete",
                extra={"response_length": len(generated_text)}
            )

            return generated_text

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise
