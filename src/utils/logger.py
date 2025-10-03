import sys
from loguru import logger
from src.utils.config import Config

# Flag to prevent multiple configurations
_logging_configured = False

def setup_logging(config: Config):
    """
    Configure the Loguru logger for the entire application.
    This should be called once at the application's entry point.
    """
    global _logging_configured
    if _logging_configured:
        return

    # Remove the default handler to prevent duplicate logs
    logger.remove()

    # Add a new handler (sink) with the level from the config
    logger.add(
        sys.stdout,
        level=config.application.log_level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )

    logger.info(f"Loguru logger configured with level {config.application.log_level.upper()}")
    _logging_configured = True

def get_logger(name: str):
    """
    Returns the pre-configured Loguru logger.
    The 'name' parameter is kept for API compatibility but is not used by Loguru in this setup.
    """
    return logger.bind(name=name) # Binding the name can be useful for context