"""
Configuration management for the Deep Research AI Agent.

Loads and validates configuration from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class AIModelConfig:
    """Configuration for AI model APIs."""
    # Required fields
    google_api_key: str
    anthropic_api_key: str

    # Gemini models (use full default capabilities)
    gemini_pro_model: str 
    gemini_flash_model: str

    # Claude models
    claude_model: str


@dataclass
class SearchConfig:
    """Configuration for search APIs."""
    serpapi_key: str
    brave_api_key: Optional[str] = None
    search_type: str = "web"
    max_results_per_query: int = 10
    max_concurrent_searches: int = 10


@dataclass
class DatabaseConfig:
    """Configuration for database connection."""
    database_url: str
    db_user: Optional[str] = None
    db_password: Optional[str] = None
    db_name: Optional[str] = None
    db_host: Optional[str] = None
    db_port: int = 5432
    cloud_sql_connection_name: Optional[str] = None




@dataclass
class ApplicationConfig:
    """General application settings."""
    max_search_iterations: int = 7
    research_timeout_minutes: int = 15
    log_level: str = "INFO"
    debug_mode: bool = False
    environment: str = "development"
    test_mode: bool = False
    mock_api_calls: bool = False


@dataclass
class GCPConfig:
    """Google Cloud Platform configuration."""
    project_id: Optional[str] = None
    region: str = "us-central1"
    use_secret_manager: bool = False
    secret_manager_project: Optional[str] = None
    storage_bucket: Optional[str] = None
    enable_gcs_storage: bool = False




@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_file_path: str = "logs/research.log"
    structured_logging: bool = True
    log_max_bytes: int = 10485760  # 10MB
    log_backup_count: int = 5
    enable_audit_log: bool = True
    audit_log_path: str = "logs/audit.log"
    anonymize_logs: bool = True


@dataclass
class PerformanceConfig:
    """Performance tuning settings."""
    extraction_batch_size: int = 10
    max_concurrent_llm_calls: int = 10
    max_concurrent_search_calls: int = 10
    api_request_timeout: int = 30
    api_retry_attempts: int = 3
    api_retry_backoff_multiplier: int = 2

    # Query generation limits
    min_queries_per_iteration: int = 4
    max_queries_per_iteration: int = 5

    # AI model temperatures
    query_generation_temperature: float = 0.7
    fact_extraction_temperature: float = 0.3
    validation_temperature: float = 0.2
    categorization_temperature: float = 0.1
    risk_analysis_temperature: float = 0.3
    connection_mapping_temperature: float = 0.3
    report_generation_temperature: float = 0.4

    # AI model max tokens
    default_max_tokens: int = 4096
    report_max_tokens: int = 8192
    structured_output_max_tokens: int = 4096


@dataclass
class ChainlitConfig:
    """Chainlit UI configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


@dataclass
class Config:
    """
    Main configuration class that aggregates all configuration sections.

    Load configuration from environment variables using the from_env() class method.
    """
    ai_models: AIModelConfig
    search: SearchConfig
    database: DatabaseConfig
    application: ApplicationConfig
    performance: PerformanceConfig

    @classmethod
    def from_env(cls) -> "Config":
        """
        Load configuration from environment variables.

        Returns:
            Config: Fully populated configuration object

        Raises:
            ValueError: If required environment variables are missing
        """
        # Load .env file if it exists
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment from {env_path}")

        # Validate required API keys
        required_keys = {
            "GOOGLE_API_KEY": "Google AI API key for Gemini",
            "ANTHROPIC_API_KEY": "Anthropic API key for Claude",
            "SERPAPI_KEY": "SerpApi key for search",
        }

        missing_keys = []
        for key, description in required_keys.items():
            if not os.getenv(key):
                missing_keys.append(f"{key} ({description})")

        if missing_keys:
            raise ValueError(
                f"Missing required environment variables:\n" +
                "\n".join(f"  - {key}" for key in missing_keys)
            )

        # AI Models Configuration (Multi-model: Claude + Gemini)
        ai_models = AIModelConfig(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            gemini_pro_model=os.getenv("GEMINI_PRO_MODEL"),
            gemini_flash_model=os.getenv("GEMINI_FLASH_MODEL"),
            claude_model=os.getenv("CLAUDE_MODEL"),
        )

        # Search Configuration
        search = SearchConfig(
            serpapi_key=os.getenv("SERPAPI_KEY"),
            brave_api_key=os.getenv("BRAVE_API_KEY"),
            search_type=os.getenv("SEARCH_TYPE"),
            max_results_per_query=int(os.getenv("MAX_SEARCH_RESULTS_PER_QUERY")),
            max_concurrent_searches=int(os.getenv("MAX_CONCURRENT_SEARCH_CALLS")),
        )

        # Database Configuration
        database = DatabaseConfig(
            database_url=os.getenv("DATABASE_UR"),
            db_user=os.getenv("DB_USER"),
            db_password=os.getenv("DB_PASSWORD"),
            db_name=os.getenv("DB_NAME"),
            db_host=os.getenv("DB_HOST"),
            db_port=int(os.getenv("DB_PORT")),
            cloud_sql_connection_name=os.getenv("CLOUD_SQL_CONNECTION_NAME"),
        )

        # Application Configuration
        application = ApplicationConfig(
            max_search_iterations=int(os.getenv("MAX_SEARCH_ITERATIONS")),
            research_timeout_minutes=int(os.getenv("RESEARCH_TIMEOUT_MINUTES")),
            log_level=os.getenv("LOG_LEVEL"),
            debug_mode=os.getenv("DEBUG_MODE").lower() == "true",
            environment=os.getenv("ENVIRONMENT"),
            test_mode=os.getenv("TEST_MODE").lower() == "true",
            mock_api_calls=os.getenv("MOCK_API_CALLS").lower() == "true",
        )

        # Performance Configuration
        performance = PerformanceConfig(
            extraction_batch_size=int(os.getenv("EXTRACTION_BATCH_SIZE")),
            max_concurrent_llm_calls=int(os.getenv("MAX_CONCURRENT_LLM_CALLS")),
            max_concurrent_search_calls=int(os.getenv("MAX_CONCURRENT_SEARCH_CALLS")),
            api_request_timeout=int(os.getenv("API_REQUEST_TIMEOUT")),
            api_retry_attempts=int(os.getenv("API_RETRY_ATTEMPTS")),
            api_retry_backoff_multiplier=int(os.getenv("API_RETRY_BACKOFF_MULTIPLIER")),
            # Query generation limits
            min_queries_per_iteration=int(os.getenv("MIN_QUERIES_PER_ITERATION", "3")),
            max_queries_per_iteration=int(os.getenv("MAX_QUERIES_PER_ITERATION", "5")),
            # AI model temperatures
            query_generation_temperature=float(os.getenv("QUERY_GENERATION_TEMPERATURE", "0.7")),
            fact_extraction_temperature=float(os.getenv("FACT_EXTRACTION_TEMPERATURE", "0.3")),
            validation_temperature=float(os.getenv("VALIDATION_TEMPERATURE", "0.2")),
            categorization_temperature=float(os.getenv("CATEGORIZATION_TEMPERATURE", "0.1")),
            risk_analysis_temperature=float(os.getenv("RISK_ANALYSIS_TEMPERATURE", "0.3")),
            connection_mapping_temperature=float(os.getenv("CONNECTION_MAPPING_TEMPERATURE", "0.3")),
            report_generation_temperature=float(os.getenv("REPORT_GENERATION_TEMPERATURE", "0.4")),
            # AI model max tokens
            default_max_tokens=int(os.getenv("DEFAULT_MAX_TOKENS", "4096")),
            report_max_tokens=int(os.getenv("REPORT_MAX_TOKENS", "8192")),
            structured_output_max_tokens=int(os.getenv("STRUCTURED_OUTPUT_MAX_TOKENS", "4096")),
        )



        logger.info("Configuration loaded successfully")

        return cls(
            ai_models=ai_models,
            search=search,
            database=database,
            application=application,
            performance=performance
        )

    def validate(self) -> None:
        """
        Validate the configuration for consistency and correctness.

        Raises:
            ValueError: If configuration is invalid
        """
        errors = []

        # Validate iterations
        if self.application.max_search_iterations < 1 or self.application.max_search_iterations > 10:
            errors.append("max_search_iterations must be between 1 and 10")

        # Validate timeout
        if self.application.research_timeout_minutes < 1:
            errors.append("research_timeout_minutes must be positive")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        logger.info("Configuration validated successfully")


# Global configuration instance (lazy loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.

    Loads configuration on first call and caches it.

    Returns:
        Config: The global configuration object
    """
    global _config
    if _config is None:
        _config = Config.from_env()
        _config.validate()
    return _config


def reload_config() -> Config:
    """
    Force reload of configuration from environment.

    Useful for testing or when environment changes.

    Returns:
        Config: Newly loaded configuration object
    """
    global _config
    _config = Config.from_env()
    _config.validate()
    return _config
