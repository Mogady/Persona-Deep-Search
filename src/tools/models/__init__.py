"""
AI Models module for the Deep Research Agent.

This module provides a unified interface for accessing AI models (Gemini and Anthropic)
through a factory pattern. Clients are created based on model names for maximum flexibility.
"""

from typing import Optional, Dict, Any

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """
    Factory for creating AI model clients.

    Supports:
    - Gemini models (Pro, Flash, etc.) via GeminiClient
    - Anthropic Claude models via AnthropicClient

    Clients are created based on model names passed as parameters.
    """

    _clients: Dict[str, Any] = {}

    @staticmethod
    def create_client(model_name: str):
        """
        Create or return cached client instance for a specific model.

        Args:
            model_name: Name of the model (e.g., 'gemini-2.0-pro-latest',
                       'gemini-2.5-flash-latest', 'claude-sonnet-4-5-20250929')

        Returns:
            Client instance (GeminiClient or AnthropicClient)

        Raises:
            ValueError: If model_name is not recognized
        """
        # Check if client already exists
        if model_name in ModelFactory._clients:
            return ModelFactory._clients[model_name]

        config = get_config()

        # Determine client type based on model name
        if "gemini" in model_name.lower():
            from .gemini_client import GeminiClient

            client = GeminiClient(
                api_key=config.ai_models.google_api_key,
                model_name=model_name
            )
            ModelFactory._clients[model_name] = client
            logger.info(f"Created new GeminiClient for model: {model_name}")
            return client

        elif "claude" in model_name.lower():
            from .anthropic_client import AnthropicClient

            client = AnthropicClient(
                api_key=config.ai_models.anthropic_api_key,
                model_name=model_name
            )
            ModelFactory._clients[model_name] = client
            logger.info(f"Created new AnthropicClient for model: {model_name}")
            return client

        else:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Model name must contain 'gemini' or 'claude'"
            )

    @staticmethod
    def get_optimal_model_for_task(task_type: str):
        """
        Return best model client for a specific task type.

        Task-to-Model Mapping:
        - Claude Sonnet (complex reasoning, risk analysis):
            - risk_analysis: Deep risk assessment
            - report_generation: Comprehensive reports
            - complex_reasoning: Advanced analysis

        - Gemini Pro (extraction, connections):
            - extraction: Fact extraction
            - entity_recognition: Named entity recognition
            - connection_mapping: Relationship mapping

        - Gemini Flash (fast operations):
            - query_generation: Search query creation
            - filtering: Content relevance filtering
            - preliminary_analysis: Quick analysis

        Args:
            task_type: Type of task to perform

        Returns:
            Client instance suitable for the task
        """
        config = get_config()

        task_mapping = {
            # Claude for complex reasoning and risk
            "risk_analysis": config.ai_models.claude_model,
            "report_generation": config.ai_models.claude_model,
            "complex_reasoning": config.ai_models.claude_model,

            # Gemini Pro for extraction and connections
            "extraction": config.ai_models.gemini_pro_model,
            "entity_recognition": config.ai_models.gemini_pro_model,
            "connection_mapping": config.ai_models.gemini_pro_model,

            # Gemini Flash for fast operations
            "query_generation": config.ai_models.gemini_flash_model,
            "filtering": config.ai_models.gemini_flash_model,
            "preliminary_analysis": config.ai_models.gemini_flash_model,
        }

        model_name = task_mapping.get(task_type)

        if model_name is None:
            logger.warning(
                f"Unknown task type '{task_type}', defaulting to Gemini Pro. "
                f"Valid task types: {', '.join(task_mapping.keys())}"
            )
            model_name = config.ai_models.gemini_pro_model

        logger.info(f"Selected model '{model_name}' for task: {task_type}")

        return ModelFactory.create_client(model_name)

    @staticmethod
    def reset_clients() -> None:
        """
        Reset all client instances.

        This is useful for testing or when you need to force recreation
        of clients (e.g., after configuration changes).
        """
        ModelFactory._clients = {}
        logger.info("Reset all model client instances")


# Import client classes for direct access
from .gemini_client import GeminiClient
from .anthropic_client import AnthropicClient

# Module exports
__all__ = [
    "GeminiClient",
    "AnthropicClient",
    "ModelFactory",
]
