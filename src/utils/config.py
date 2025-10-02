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
    gemini_pro_model: str = "gemini-2.0-pro-latest"
    gemini_flash_model: str = "gemini-2.5-flash-latest"

    # Claude models
    claude_model: str = "claude-sonnet-4-5-20250929"


@dataclass
class SearchConfig:
    """Configuration for search APIs."""
    serpapi_key: str
    brave_api_key: Optional[str] = None
    firecrawl_api_key: Optional[str] = None
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
    max_facts_per_session: int = 100
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
    gcp: GCPConfig
    logging: LoggingConfig
    performance: PerformanceConfig
    chainlit: ChainlitConfig

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
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            gemini_pro_model=os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro"),
            gemini_flash_model=os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash"),
            claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929"),
        )

        # Search Configuration
        search = SearchConfig(
            serpapi_key=os.getenv("SERPAPI_KEY", ""),
            brave_api_key=os.getenv("BRAVE_API_KEY"),
            firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY"),
            search_type=os.getenv("SEARCH_TYPE", "web"),
            max_results_per_query=int(os.getenv("MAX_SEARCH_RESULTS_PER_QUERY", "10")),
            max_concurrent_searches=int(os.getenv("MAX_CONCURRENT_SEARCH_CALLS", "10")),
        )

        # Database Configuration
        database = DatabaseConfig(
            database_url=os.getenv("DATABASE_URL", "sqlite:///research_agent.db"),
            db_user=os.getenv("DB_USER"),
            db_password=os.getenv("DB_PASSWORD"),
            db_name=os.getenv("DB_NAME"),
            db_host=os.getenv("DB_HOST"),
            db_port=int(os.getenv("DB_PORT", "5432")),
            cloud_sql_connection_name=os.getenv("CLOUD_SQL_CONNECTION_NAME"),
        )

        # Application Configuration
        application = ApplicationConfig(
            max_search_iterations=int(os.getenv("MAX_SEARCH_ITERATIONS", "7")),
            max_facts_per_session=int(os.getenv("MAX_FACTS_PER_SESSION", "100")),
            research_timeout_minutes=int(os.getenv("RESEARCH_TIMEOUT_MINUTES", "15")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true",
            environment=os.getenv("ENVIRONMENT", "development"),
            test_mode=os.getenv("TEST_MODE", "false").lower() == "true",
            mock_api_calls=os.getenv("MOCK_API_CALLS", "false").lower() == "true",
        )

        # GCP Configuration
        gcp = GCPConfig(
            project_id=os.getenv("GCP_PROJECT_ID"),
            region=os.getenv("GCP_REGION", "us-central1"),
            use_secret_manager=os.getenv("USE_SECRET_MANAGER", "false").lower() == "true",
            secret_manager_project=os.getenv("SECRET_MANAGER_PROJECT"),
            storage_bucket=os.getenv("GCP_STORAGE_BUCKET"),
            enable_gcs_storage=os.getenv("ENABLE_GCS_STORAGE", "false").lower() == "true",
        )

        # Logging Configuration
        logging_config = LoggingConfig(
            log_file_path=os.getenv("LOG_FILE_PATH", "logs/research.log"),
            structured_logging=os.getenv("STRUCTURED_LOGGING", "true").lower() == "true",
            log_max_bytes=int(os.getenv("LOG_MAX_BYTES", "10485760")),
            log_backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
            enable_audit_log=os.getenv("ENABLE_AUDIT_LOG", "true").lower() == "true",
            audit_log_path=os.getenv("AUDIT_LOG_PATH", "logs/audit.log"),
            anonymize_logs=os.getenv("ANONYMIZE_LOGS", "true").lower() == "true",
        )

        # Performance Configuration
        performance = PerformanceConfig(
            extraction_batch_size=int(os.getenv("EXTRACTION_BATCH_SIZE", "10")),
            max_concurrent_llm_calls=int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "10")),
            max_concurrent_search_calls=int(os.getenv("MAX_CONCURRENT_SEARCH_CALLS", "10")),
            api_request_timeout=int(os.getenv("API_REQUEST_TIMEOUT", "30")),
            api_retry_attempts=int(os.getenv("API_RETRY_ATTEMPTS", "3")),
            api_retry_backoff_multiplier=int(os.getenv("API_RETRY_BACKOFF_MULTIPLIER", "2")),
        )

        # Chainlit Configuration
        chainlit = ChainlitConfig(
            host=os.getenv("CHAINLIT_HOST", "0.0.0.0"),
            port=int(os.getenv("CHAINLIT_PORT", "8000")),
            debug=os.getenv("CHAINLIT_DEBUG", "false").lower() == "true",
        )

        logger.info("Configuration loaded successfully")

        return cls(
            ai_models=ai_models,
            search=search,
            database=database,
            application=application,
            gcp=gcp,
            logging=logging_config,
            performance=performance,
            chainlit=chainlit,
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
