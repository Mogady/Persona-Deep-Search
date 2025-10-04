"""
SerpApi search integration using direct REST API.

Provides comprehensive search results with full control over
response parsing and feature extraction.
"""

import requests
from typing import List, Dict, Optional, Any
from datetime import datetime
from urllib.parse import urlparse
from tenacity import retry, stop_after_attempt, wait_exponential

from dateutil import parser
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.rate_limiter import get_search_rate_limiter
from .models import SearchResult


class SerpApiSearch:
    """
    Direct SerpApi integration for Google search.

    Uses REST API for full control over result parsing and
    extraction of advanced features (knowledge graph, rich snippets).
    """

    SERPAPI_BASE_URL = "https://serpapi.com/search"

    def __init__(self, api_key: str, rate_limiter=None):
        """
        Initialize SerpApi search client.

        Args:
            api_key: SerpApi API key
            rate_limiter: Optional rate limiter (defaults to global search limiter)
        """
        self.api_key = api_key
        self.rate_limiter = rate_limiter or get_search_rate_limiter(max_concurrent=10)
        self.logger = get_logger(__name__)

        self.logger.debug("Initialized SerpApiSearch with rate limiting")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def search(
        self,
        query: str,
        max_results: int = 10,
        search_type: str = "web"
    ) -> List[SearchResult]:
        """
        Execute Google search via SerpApi.

        Args:
            query: Search query
            max_results: Maximum number of results (default: 10)
            search_type: Type of search ("web", "news", "scholar")

        Returns:
            List[SearchResult]: Parsed search results
        """

        # Use rate limiter
        with self.rate_limiter.acquire():
            self.logger.debug(f"Searching SerpApi: '{query}' (max_results={max_results})")

            # Build request parameters
            params = {
                "q": query,
                "api_key": self.api_key,
                "num": max_results,
                "engine": "google",
            }

            # Add search type specific params
            if search_type == "news":
                params["tbm"] = "nws"
            elif search_type == "scholar":
                params["engine"] = "google_scholar"

            try:
                # Make API request
                response = requests.get(
                    self.SERPAPI_BASE_URL,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                # Parse results
                results = self._parse_serp_response(data, query)


                self.logger.debug(f"Found {len(results)} results for '{query}'")

                return results

            except requests.exceptions.RequestException as e:
                self.logger.error(f"SerpApi request failed: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Error parsing SerpApi response: {e}")
                raise

    def batch_search(self, queries: List[str], max_results: int = 10) -> List[SearchResult]:
        """
        Execute multiple searches efficiently.

        Args:
            queries: List of search queries
            max_results: Maximum results per query

        Returns:
            List[SearchResult]: All results combined (deduplicated)
        """
        self.logger.debug(f"Batch searching {len(queries)} queries")

        all_results = []
        for query in queries:
            try:
                results = self.search(query, max_results=max_results)
                all_results.extend(results)
            except Exception as e:
                self.logger.error(f"Failed to search '{query}': {e}")
                continue

        # Deduplicate
        deduplicated = self._deduplicate_results(all_results)

        self.logger.debug(
            f"Batch search complete: {len(all_results)} total, "
            f"{len(deduplicated)} after deduplication"
        )

        return deduplicated

    def _parse_serp_response(self, data: Dict[str, Any], query: str) -> List[SearchResult]:
        """
        Parse SerpApi JSON response into SearchResult objects.

        Extracts:
        - Organic results
        - Knowledge graph (if available)
        - People also ask (if available)
        - Rich snippets

        Args:
            data: SerpApi JSON response
            query: Original search query

        Returns:
            List[SearchResult]: Parsed results
        """
        results = []

        # 1. Extract organic results
        for idx, item in enumerate(data.get("organic_results", [])):
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                content=item.get("snippet", ""),
                score=float(idx),  # Position as score (0 is best)
                source_domain=self._extract_domain(item.get("link", "")),
                published_date=self._parse_date(item.get("date")),
                search_engine="serpapi",
                metadata={
                    "position": item.get("position"),
                    "rich_snippet": item.get("rich_snippet"),
                    "sitelinks": item.get("sitelinks"),
                }
            )
            results.append(result)

        # 2. Extract knowledge graph (high priority)
        if "knowledge_graph" in data:
            kg = data["knowledge_graph"]
            kg_result = SearchResult(
                title=kg.get("title", ""),
                url=kg.get("website", "") or kg.get("source", {}).get("link", ""),
                content=kg.get("description", ""),
                score=0.0,  # Highest priority
                source_domain=self._extract_domain(
                    kg.get("website", "") or kg.get("source", {}).get("link", "")
                ),
                published_date=None,
                search_engine="serpapi_knowledge_graph",
                metadata={
                    "type": kg.get("type"),
                    "kgmid": kg.get("kgmid"),
                }
            )
            results.insert(0, kg_result)  # Add at beginning (high priority)

        # 3. Extract "People Also Ask" (good for related info)
        for paa in data.get("related_questions", [])[:3]:  # Take top 3
            paa_result = SearchResult(
                title=paa.get("question", ""),
                url=paa.get("link", ""),
                content=paa.get("snippet", ""),
                score=99.0,  # Lower priority
                source_domain=self._extract_domain(paa.get("link", "")),
                published_date=None,
                search_engine="serpapi_related_question",
                metadata={"question": paa.get("question")}
            )
            results.append(paa_result)

        return results

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate URLs and near-duplicate content.

        Args:
            results: List of search results

        Returns:
            List[SearchResult]: Deduplicated results
        """
        seen_urls = set()
        seen_content = set()
        deduplicated = []

        for result in results:
            # Check URL
            if result.url in seen_urls:
                continue

            # Check content similarity (simple - first 100 chars)
            content_sig = result.content[:100].lower().strip()
            if content_sig in seen_content and content_sig:
                continue

            seen_urls.add(result.url)
            seen_content.add(content_sig)
            deduplicated.append(result)

        return deduplicated

    def _calculate_source_diversity(self, results: List[SearchResult]) -> float:
        """
        Calculate source diversity score based on unique domains.

        Args:
            results: List of search results

        Returns:
            float: Diversity score from 0.0 to 1.0
        """
        if not results:
            return 0.0

        unique_domains = set(r.source_domain for r in results if r.source_domain)

        # Normalize: 1.0 if all different domains, lower if duplicates
        diversity = len(unique_domains) / len(results)

        return min(1.0, diversity)

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.replace("www.", "")
        except:
            return ""

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None

        try:
            return parser.parse(date_str)
        except:
            return None
