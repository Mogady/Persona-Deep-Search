# 📋 AGENT 3 PROMPT - FINAL VERSION (Direct API Implementation)

## 🎯 Mission
Build robust search tools using **direct API calls** to SerpApi (primary) and Brave Search (optional secondary). **No LangChain wrappers** - full control over API responses for better feature extraction.

## 🔧 Environment Setup
- **Python Environment**: Use existing `.venv` in project root
- **Package Installation**: `source .venv/bin/activate && pip install <package>`
- **Configuration**: Import from `src.utils.config` (completed by Agent 1)
- **Logging**: Import from `src.utils.logger` (completed by Agent 1)
- **Rate Limiting**: Import from `src.utils.rate_limiter` (available - use `get_search_rate_limiter()`)

## 📦 Deliverables

### 1. `src/tools/search/serp_api_search.py` - **REQUIRED**

**Implementation using direct REST API calls:**

```python
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
        self.cache = {}  # Simple in-memory cache

        self.logger.info("Initialized SerpApiSearch with rate limiting")

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
        # Check cache first
        cache_key = f"{query}:{max_results}:{search_type}"
        if cache_key in self.cache:
            self.logger.debug(f"Cache hit for query: {query}")
            return self.cache[cache_key]

        # Use rate limiter
        with self.rate_limiter.acquire():
            self.logger.info(f"Searching SerpApi: '{query}' (max_results={max_results})")

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

                # Cache results
                self.cache[cache_key] = results

                self.logger.info(f"Found {len(results)} results for '{query}'")

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
        self.logger.info(f"Batch searching {len(queries)} queries")

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

        self.logger.info(
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

        # Simple date parsing - enhance as needed
        try:
            from dateutil import parser
            return parser.parse(date_str)
        except:
            return None
```

---

### 2. `src/tools/search/brave_search.py` - **OPTIONAL**

**Implementation using Brave Search REST API:**

```python
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
```

---

### 3. `src/tools/search/models.py` - **REQUIRED**

**Data models:**

```python
"""
Data models for search results.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class SearchResult:
    """
    Structured search result from any search engine.

    Attributes:
        title: Result title
        url: Result URL
        content: Snippet/description
        score: Relevance score (lower is better, 0 = most relevant)
        source_domain: Domain name (e.g., "example.com")
        published_date: Publication date if available
        search_engine: Source engine ("serpapi", "brave", etc.)
        metadata: Additional engine-specific data
    """
    title: str
    url: str
    content: str
    score: float
    source_domain: str
    published_date: Optional[datetime] = None
    search_engine: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"<SearchResult(title='{self.title[:50]}...', "
            f"url='{self.url}', engine={self.search_engine})>"
        )
```

---

### 4. `src/tools/search/__init__.py` - **REQUIRED**

**SearchOrchestrator with hybrid strategy:**

```python
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
```

---

## 📦 Required Packages

**Add to requirements.txt:**
```txt
requests>=2.31.0
python-dateutil>=2.8.2
tenacity>=8.2.3  # Already included
```

**NO need for:**
- ❌ `langchain`
- ❌ `langchain-community`
- ❌ `google-search-results` (we're using direct REST API)

---

## ✅ Completion Checklist

**Must Have:**
- [ ] `src/tools/search/models.py` - SearchResult dataclass
- [ ] `src/tools/search/serp_api_search.py` - Direct SerpApi REST implementation
- [ ] `src/tools/search/__init__.py` - SearchOrchestrator with fallback
- [ ] Rate limiting integrated (max 10 concurrent)
- [ ] In-memory caching (simple dict)
- [ ] Deduplication (URL and content)
- [ ] Source diversity calculation
- [ ] Knowledge graph extraction
- [ ] Rich snippets extraction
- [ ] Error handling with retry logic
- [ ] Integration with config, logger, rate limiter
- [ ] All integration tests passing

**Nice to Have:**
- [ ] `src/tools/search/brave_search.py` - Brave Search API implementation
- [ ] Hybrid fallback strategy working

---

## 🧪 Testing Example

```python
from src.utils.config import get_config
from src.tools.search import SearchOrchestrator

# Initialize
config = get_config()
orchestrator = SearchOrchestrator(config)

# Test basic search
results = orchestrator.search("John Doe CEO", max_results=10)

# Verify structure
assert all(isinstance(r, SearchResult) for r in results)
assert len(results) <= 10

# Check for knowledge graph extraction
kg_results = [r for r in results if "knowledge_graph" in r.search_engine]
print(f"Found {len(kg_results)} knowledge graph results")

# Test batch search
queries = ["query 1", "query 2", "query 3"]
all_results = orchestrator.batch_search(queries)

# Test source diversity
diversity = orchestrator.calculate_source_diversity(all_results)
print(f"Source diversity: {diversity:.2f}")
assert 0.0 <= diversity <= 1.0

# Test caching (second call should be instant)
import time
start = time.time()
results2 = orchestrator.search("John Doe CEO", max_results=10)
elapsed = time.time() - start
assert elapsed < 0.1  # Should be cached
```

---

## 🚀 Why Direct API > LangChain

1. ✅ **Full control** over SerpApi features (knowledge graph, rich snippets, people also ask)
2. ✅ **Exact SearchResult mapping** - no fighting with LangChain's Document structure
3. ✅ **Better caching** - our custom in-memory dict
4. ✅ **Seamless rate limiting** - integrates perfectly with our RateLimiter
5. ✅ **Lighter weight** - just `requests` library, no LangChain overhead
6. ✅ **Easier debugging** - see exactly what API returns
7. ✅ **More flexible** - can add custom parsing for any SerpApi feature

---

## 🔑 API Documentation

- **SerpApi Docs**: https://serpapi.com/search-api
- **Brave Search Docs**: https://api.search.brave.com/app/documentation/web-search/get-started

---

**Ready to dispatch Agent 3!** 🚀
