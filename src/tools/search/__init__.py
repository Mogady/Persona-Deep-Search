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
            self.logger.debug("Brave Search enabled as fallback")
        else:
            self.brave_search = None
            self.logger.debug("Brave Search not configured (SerpApi only)")

    def search(
        self,
        query: str,
        max_results: int = 10,
        use_fallback: bool = True
    ) -> List[SearchResult]:
        """
        Execute search with automatic fallback.

        Strategy:
        1. Primary: Try SerpApi (Google Search)
        2. Supplement: If SerpApi returns < 3 results, supplement with Brave
        3. Fallback: If SerpApi fails completely, use Brave as fallback
        4. Fail: If both fail, return empty list (graceful degradation)

        Args:
            query: Search query
            max_results: Maximum results to return
            use_fallback: Whether to use Brave as fallback

        Returns:
            List[SearchResult]: Combined and deduplicated results
        """
        self.logger.debug(f"Orchestrating search for: '{query}'")

        # 1. Try SerpApi first (primary)
        try:
            results = self.serp_search.search(query, max_results=max_results)

            # If we got enough quality results, return them
            if len(results) >= 3 or not use_fallback or not self.brave_search:
                self.logger.debug(f"SerpApi returned {len(results)} results")
                return results

            # 2. Supplement with Brave Search if we have few results
            self.logger.debug(
                f"Only {len(results)} results from SerpApi, "
                f"supplementing with Brave Search"
            )

            try:
                brave_results = self.brave_search.search(
                    query,
                    max_results=max_results - len(results)
                )

                # Combine and deduplicate
                combined = results + brave_results
                deduplicated = self.serp_search._deduplicate_results(combined)

                self.logger.debug(
                    f"Returning {len(deduplicated)} combined results "
                    f"({len(results)} SerpApi + {len(brave_results)} Brave)"
                )

                return deduplicated

            except Exception as brave_error:
                self.logger.warning(f"Brave Search supplementation failed: {brave_error}")
                # Return SerpApi results even if supplementation failed
                return results

        except Exception as e:
            self.logger.error(f"SerpApi failed: {e}", exc_info=True)

            # 3. Fallback to Brave if SerpApi fails completely
            if self.brave_search and use_fallback:
                try:
                    self.logger.warning("SerpApi failed completely, falling back to Brave Search")
                    brave_results = self.brave_search.search(query, max_results=max_results)
                    self.logger.debug(f"Brave Search fallback returned {len(brave_results)} results")
                    return brave_results
                except Exception as brave_error:
                    self.logger.error(f"Brave Search fallback also failed: {brave_error}", exc_info=True)

            # 4. Both failed - return empty list (graceful degradation)
            self.logger.error(f"All search providers failed for query: '{query}'")
            return []

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
