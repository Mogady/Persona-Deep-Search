"""
Simple rate limiter based on concurrent request count.

Provides semaphore-based rate limiting for API calls to prevent
overwhelming external services.
"""

from typing import Optional
from contextlib import  contextmanager
import logging
from threading import Semaphore

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple rate limiter based on concurrent request count.

    Uses semaphore to limit the number of concurrent requests.
    Thread-safe for synchronous operations.
    """

    def __init__(self, max_concurrent: int = 10, name: str = "rate_limiter"):
        """
        Initialize the rate limiter.

        Args:
            max_concurrent: Maximum number of concurrent requests allowed
            name: Name for logging purposes
        """
        self.max_concurrent = max_concurrent
        self.name = name
        self.semaphore = Semaphore(max_concurrent)
        self._active_requests = 0

        logger.info(
            f"Initialized {self.name} with max_concurrent={max_concurrent}"
        )

    @contextmanager
    def acquire(self):
        """
        Context manager for rate-limited operations.

        Usage:
            with rate_limiter.acquire():
                # Make API call here
                result = api_call()

        Yields:
            None
        """
        acquired = False
        try:
            # Acquire semaphore
            self.semaphore.acquire()
            acquired = True
            self._active_requests += 1

            logger.debug(
                f"{self.name}: Acquired slot ({self._active_requests}/{self.max_concurrent} active)"
            )

            yield

        finally:
            if acquired:
                self._active_requests -= 1
                self.semaphore.release()

                logger.debug(
                    f"{self.name}: Released slot ({self._active_requests}/{self.max_concurrent} active)"
                )

    def available_slots(self) -> int:
        """
        Get number of available slots.

        Returns:
            int: Number of available request slots
        """
        return self.max_concurrent - self._active_requests

    def is_available(self) -> bool:
        """
        Check if rate limiter has available slots.

        Returns:
            bool: True if slots are available
        """
        return self.available_slots() > 0

    def __repr__(self) -> str:
        return (
            f"<RateLimiter(name='{self.name}', "
            f"max_concurrent={self.max_concurrent}, "
            f"active={self._active_requests})>"
        )


# Global rate limiters (initialized on first use)
_search_rate_limiter: Optional[RateLimiter] = None
_llm_rate_limiter: Optional[RateLimiter] = None


def get_search_rate_limiter(max_concurrent: int = 10) -> RateLimiter:
    """
    Get or create the global search API rate limiter.

    Args:
        max_concurrent: Maximum concurrent search requests (default: 10)

    Returns:
        RateLimiter: Global search rate limiter instance
    """
    global _search_rate_limiter

    if _search_rate_limiter is None:
        _search_rate_limiter = RateLimiter(
            max_concurrent=max_concurrent,
            name="search_api"
        )
        logger.info(f"Created global search rate limiter: {_search_rate_limiter}")

    return _search_rate_limiter


def get_llm_rate_limiter(max_concurrent: int = 10) -> RateLimiter:
    """
    Get or create the global LLM API rate limiter.

    Args:
        max_concurrent: Maximum concurrent LLM requests (default: 10)

    Returns:
        RateLimiter: Global LLM rate limiter instance
    """
    global _llm_rate_limiter

    if _llm_rate_limiter is None:
        _llm_rate_limiter = RateLimiter(
            max_concurrent=max_concurrent,
            name="llm_api"
        )
        logger.info(f"Created global LLM rate limiter: {_llm_rate_limiter}")

    return _llm_rate_limiter


def reset_rate_limiters() -> None:
    """
    Reset all global rate limiters.

    Useful for testing or when you need to reinitialize with different settings.
    """
    global _search_rate_limiter, _llm_rate_limiter

    _search_rate_limiter = None
    _llm_rate_limiter = None

    logger.info("Reset all global rate limiters")
