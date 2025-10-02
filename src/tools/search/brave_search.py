"""
Brave Search API integration using direct REST API.

Optional secondary search engine for diversity and coverage.
"""

import requests
from typing import List, Optional
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.rate_limiter import get_search_rate_limiter
from .models import SearchResult


class BraveSearch:
    """
    Direct Brave Search API integration.

    Secondary search engine for additional coverage and diversity.
    """

    BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str, rate_limiter=None):
        """
        Initialize Brave Search client.

        Args:
            api_key: Brave Search API key
            rate_limiter: Optional rate limiter (defaults to global search limiter)
        """
        self.api_key = api_key
        self.rate_limiter = rate_limiter or get_search_rate_limiter(max_concurrent=10)
        self.logger = get_logger(__name__)
        self.cache = {}

        self.logger.info("Initialized BraveSearch with rate limiting")

    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Execute search via Brave Search API.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List[SearchResult]: Parsed search results
        """
        # Check cache
        cache_key = f"{query}:{max_results}"
        if cache_key in self.cache:
            self.logger.debug(f"Cache hit for query: {query}")
            return self.cache[cache_key]

        # Use rate limiter
        with self.rate_limiter.acquire():
            self.logger.info(f"Searching Brave: '{query}' (max_results={max_results})")

            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key
            }

            params = {
                "q": query,
                "count": max_results,
            }

            try:
                response = requests.get(
                    self.BRAVE_API_URL,
                    headers=headers,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                # Parse results
                results = self._parse_brave_response(data)

                # Cache
                self.cache[cache_key] = results

                self.logger.info(f"Found {len(results)} results from Brave")

                return results

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Brave Search request failed: {e}")
                raise

    def _parse_brave_response(self, data: dict) -> List[SearchResult]:
        """Parse Brave Search JSON response."""
        results = []

        for idx, item in enumerate(data.get("web", {}).get("results", [])):
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("description", ""),
                score=float(idx),
                source_domain=self._extract_domain(item.get("url", "")),
                published_date=self._parse_date(item.get("age")),
                search_engine="brave",
                metadata={}
            )
            results.append(result)

        return results

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            return parsed.netloc.replace("www.", "")
        except:
            return ""

    def _parse_date(self, age_str: Optional[str]) -> Optional[datetime]:
        """Parse Brave age string to datetime."""
        # Brave returns age like "2 days ago" - simple implementation
        return None  # Can enhance later if needed
