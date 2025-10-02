"""
Structured logging utilities for the Deep Research AI Agent.

Provides JSON-formatted logging with correlation IDs, context tracking,
and specialized logging functions for different agent operations.
"""

import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler
import uuid
import re


class StructuredFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.

    Outputs log records as JSON with consistent fields for easy parsing
    and analysis in log aggregation systems.
    """

    def __init__(self, anonymize: bool = False):
        """
        Initialize the structured formatter.

        Args:
            anonymize: Whether to anonymize PII in logs
        """
        super().__init__()
        self.anonymize = anonymize

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            str: JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if present
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id

        # Add session ID if present
        if hasattr(record, "session_id"):
            log_data["session_id"] = record.session_id

        # Add agent name if present
        if hasattr(record, "agent_name"):
            log_data["agent_name"] = record.agent_name

        # Add custom fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Anonymize PII if enabled
        if self.anonymize:
            log_data["message"] = self._anonymize_text(log_data["message"])

        return json.dumps(log_data, default=str)

    def _anonymize_text(self, text: str) -> str:
        """
        Anonymize potential PII in text.

        Args:
            text: Text to anonymize

        Returns:
            str: Anonymized text
        """
        # Replace email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

        # Replace phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)

        # Replace social security numbers
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)

        # Replace credit card numbers (simplified)
        text = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[CC]', text)

        return text


class ContextLogger(logging.LoggerAdapter):
    """
    Logger adapter that adds contextual information to all log records.

    Automatically includes correlation IDs, session IDs, and other
    contextual information in log messages.
    """

    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        """
        Initialize the context logger.

        Args:
            logger: Base logger instance
            extra: Extra context to include in all logs
        """
        super().__init__(logger, extra or {})

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process log message to add context.

        Args:
            msg: Log message
            kwargs: Keyword arguments

        Returns:
            tuple: (message, kwargs) with added context
        """
        # Add extra fields to the log record
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        # Merge logger's extra with kwargs extra
        for key, value in self.extra.items():
            if key not in kwargs["extra"]:
                kwargs["extra"][key] = value

        return msg, kwargs


def setup_logger(
    name: str = "research_agent",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
    structured: bool = True,
    anonymize: bool = False,
) -> logging.Logger:
    """
    Setup and configure a logger instance.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, only console logging)
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        structured: Use structured (JSON) logging format
        anonymize: Anonymize PII in logs

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatter
    if structured:
        formatter = StructuredFormatter(anonymize=anonymize)
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (with rotation)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info(f"Logger '{name}' initialized at {log_level} level")

    return logger


def get_context_logger(
    name: str,
    correlation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> ContextLogger:
    """
    Get a logger with context information.

    Args:
        name: Logger name
        correlation_id: Correlation ID for tracking related operations
        session_id: Research session ID
        agent_name: Name of the agent node

    Returns:
        ContextLogger: Logger with context
    """
    logger = logging.getLogger(name)

    extra = {}
    if correlation_id:
        extra["correlation_id"] = correlation_id
    if session_id:
        extra["session_id"] = session_id
    if agent_name:
        extra["agent_name"] = agent_name

    return ContextLogger(logger, extra)


def log_agent_action(
    logger: logging.Logger,
    agent_name: str,
    action: str,
    session_id: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an agent action with structured context.

    Args:
        logger: Logger instance
        agent_name: Name of the agent
        action: Action being performed
        session_id: Research session ID
        details: Additional details about the action
    """
    extra_fields = {
        "agent_name": agent_name,
        "session_id": session_id,
        "action": action,
    }

    if details:
        extra_fields["details"] = details

    logger.info(
        f"Agent {agent_name}: {action}",
        extra={"extra_fields": extra_fields}
    )


def log_search(
    logger: logging.Logger,
    query: str,
    search_engine: str,
    results_count: int,
    session_id: str,
    iteration: int,
    execution_time_ms: Optional[int] = None,
) -> None:
    """
    Log a search operation with metrics.

    Args:
        logger: Logger instance
        query: Search query
        search_engine: Search engine used (tavily, serp, etc.)
        results_count: Number of results returned
        session_id: Research session ID
        iteration: Search iteration number
        execution_time_ms: Execution time in milliseconds
    """
    extra_fields = {
        "session_id": session_id,
        "operation": "search",
        "search_engine": search_engine,
        "results_count": results_count,
        "iteration": iteration,
        "query_length": len(query),
    }

    if execution_time_ms is not None:
        extra_fields["execution_time_ms"] = execution_time_ms

    logger.info(
        f"Search executed: '{query[:50]}...' ({results_count} results)",
        extra={"extra_fields": extra_fields}
    )


def log_fact_extraction(
    logger: logging.Logger,
    facts_count: int,
    source_url: str,
    session_id: str,
    model: str,
    confidence_scores: Optional[list] = None,
) -> None:
    """
    Log fact extraction operation.

    Args:
        logger: Logger instance
        facts_count: Number of facts extracted
        source_url: Source URL
        session_id: Research session ID
        model: Model used for extraction
        confidence_scores: List of confidence scores
    """
    extra_fields = {
        "session_id": session_id,
        "operation": "fact_extraction",
        "facts_count": facts_count,
        "source_url": source_url,
        "model": model,
    }

    if confidence_scores:
        extra_fields["avg_confidence"] = sum(confidence_scores) / len(confidence_scores)
        extra_fields["min_confidence"] = min(confidence_scores)
        extra_fields["max_confidence"] = max(confidence_scores)

    logger.info(
        f"Extracted {facts_count} facts from {source_url}",
        extra={"extra_fields": extra_fields}
    )


def log_validation(
    logger: logging.Logger,
    facts_validated: int,
    verified_count: int,
    rejected_count: int,
    session_id: str,
) -> None:
    """
    Log validation operation results.

    Args:
        logger: Logger instance
        facts_validated: Total facts validated
        verified_count: Number of verified facts
        rejected_count: Number of rejected facts
        session_id: Research session ID
    """
    extra_fields = {
        "session_id": session_id,
        "operation": "validation",
        "facts_validated": facts_validated,
        "verified_count": verified_count,
        "rejected_count": rejected_count,
        "verification_rate": verified_count / facts_validated if facts_validated > 0 else 0,
    }

    logger.info(
        f"Validated {facts_validated} facts: {verified_count} verified, {rejected_count} rejected",
        extra={"extra_fields": extra_fields}
    )


def log_risk_analysis(
    logger: logging.Logger,
    risks_found: int,
    critical_count: int,
    high_count: int,
    session_id: str,
) -> None:
    """
    Log risk analysis results.

    Args:
        logger: Logger instance
        risks_found: Total risks identified
        critical_count: Number of critical risks
        high_count: Number of high-severity risks
        session_id: Research session ID
    """
    extra_fields = {
        "session_id": session_id,
        "operation": "risk_analysis",
        "risks_found": risks_found,
        "critical_count": critical_count,
        "high_count": high_count,
    }

    logger.info(
        f"Risk analysis: {risks_found} risks found ({critical_count} critical, {high_count} high)",
        extra={"extra_fields": extra_fields}
    )


def log_error(
    logger: logging.Logger,
    error: Exception,
    session_id: str,
    operation: str,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an error with context.

    Args:
        logger: Logger instance
        error: Exception that occurred
        session_id: Research session ID
        operation: Operation that failed
        context: Additional context about the error
    """
    extra_fields = {
        "session_id": session_id,
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

    if context:
        extra_fields["context"] = context

    logger.error(
        f"Error in {operation}: {str(error)}",
        extra={"extra_fields": extra_fields},
        exc_info=True
    )


def log_api_call(
    logger: logging.Logger,
    api_name: str,
    endpoint: str,
    status_code: Optional[int] = None,
    response_time_ms: Optional[int] = None,
    session_id: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """
    Log an API call with metrics.

    Args:
        logger: Logger instance
        api_name: Name of the API (anthropic, google, serp, etc.)
        endpoint: API endpoint called
        status_code: HTTP status code
        response_time_ms: Response time in milliseconds
        session_id: Research session ID
        error: Error message if call failed
    """
    extra_fields = {
        "operation": "api_call",
        "api_name": api_name,
        "endpoint": endpoint,
    }

    if session_id:
        extra_fields["session_id"] = session_id
    if status_code:
        extra_fields["status_code"] = status_code
    if response_time_ms:
        extra_fields["response_time_ms"] = response_time_ms
    if error:
        extra_fields["error"] = error

    level = logging.INFO if not error else logging.ERROR
    message = f"API call to {api_name}: {endpoint}"
    if error:
        message += f" - ERROR: {error}"

    logger.log(level, message, extra={"extra_fields": extra_fields})


def generate_correlation_id() -> str:
    """
    Generate a unique correlation ID for tracking related operations.

    Returns:
        str: UUID correlation ID
    """
    return str(uuid.uuid4())


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance by name.

    This is a convenience function that returns a standard Python logger.
    For context-aware logging, use get_context_logger() instead.

    Args:
        name: Logger name (typically __name__)

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)
