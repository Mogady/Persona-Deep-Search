"""
Search orchestration with fallback strategy.

Coordinates between SerpApi (primary) and Brave Search (optional fallback).
"""

from typing import List, Optional
from src.utils.config import get_config
from src.utils.logger import get_logger

from .models import SearchResult
from .serp_api_search import SerpApiSearch
from .brave_search import BraveSearch


class SearchOrchestrator:
    """
    Orchestrates search across multiple engines with fallback.

    Strategy:
    1. Primary: SerpApi (comprehensive Google results)
    2. Fallback: Brave Search (if configured and needed)
    """

    def __init__(self, config=None):
        """
        Initialize search orchestrator.

        Args:
            config: Optional config object (defaults to global config)
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__)

        # Initialize SerpApi (primary)
        self.serp_search = SerpApiSearch(
            api_key=self.config.search.serpapi_key
        )

        # Initialize Brave Search if API key available (optional)
        if self.config.search.brave_api_key:
            self.brave_search = BraveSearch(
                api_key=self.config.search.brave_api_key
            )
            self.logger.info("Brave Search enabled as fallback")
        else:
            self.brave_search = None
            self.logger.info("Brave Search not configured (SerpApi only)")

    def search(
        self,
        query: str,
        max_results: int = 10,
        use_fallback: bool = True
    ) -> List[SearchResult]:
        """
        Execute search with optional fallback.

        Args:
            query: Search query
            max_results: Maximum results to return
            use_fallback: Whether to use Brave as fallback if SerpApi returns few results

        Returns:
            List[SearchResult]: Combined and deduplicated results
        """
        self.logger.info(f"Orchestrating search for: '{query}'")

        # 1. Try SerpApi first (primary)
        try:
            results = self.serp_search.search(query, max_results=max_results)

            # If we got enough quality results, return them
            if len(results) >= 3 or not use_fallback or not self.brave_search:
                self.logger.info(f"Returning {len(results)} results from SerpApi")
                return results

            # 2. Supplement with Brave Search if available
            self.logger.info(
                f"Only {len(results)} results from SerpApi, "
                f"supplementing with Brave Search"
            )

            brave_results = self.brave_search.search(
                query,
                max_results=max_results - len(results)
            )

            # Combine and deduplicate
            combined = results + brave_results
            deduplicated = self.serp_search._deduplicate_results(combined)

            self.logger.info(
                f"Returning {len(deduplicated)} combined results "
                f"({len(results)} SerpApi + {len(brave_results)} Brave)"
            )

            return deduplicated

        except Exception as e:
            self.logger.error(f"Search failed: {e}")

            # Fallback to Brave if SerpApi fails completely
            if self.brave_search and use_fallback:
                self.logger.warning("SerpApi failed, falling back to Brave Search")
                return self.brave_search.search(query, max_results=max_results)

            raise

    def batch_search(self, queries: List[str], max_results: int = 10) -> List[SearchResult]:
        """
        Execute multiple searches.

        Args:
            queries: List of search queries
            max_results: Maximum results per query

        Returns:
            List[SearchResult]: All results combined and deduplicated
        """
        return self.serp_search.batch_search(queries, max_results=max_results)

    def calculate_source_diversity(self, results: List[SearchResult]) -> float:
        """
        Calculate source diversity score.

        Args:
            results: List of search results

        Returns:
            float: Diversity score (0.0 to 1.0)
        """
        return self.serp_search._calculate_source_diversity(results)


# Export main classes and models
__all__ = [
    "SearchOrchestrator",
    "SearchResult",
    "SerpApiSearch",
    "BraveSearch",
]
