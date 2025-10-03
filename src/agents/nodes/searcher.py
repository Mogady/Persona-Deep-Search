"""
Search Executor Node - Executes search queries and collects results.

This node executes all planned queries using the SearchOrchestrator and manages
the search history.
"""

from typing import Dict, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from src.tools.search import SearchOrchestrator, SearchResult
from src.utils.logger import get_logger
from src.utils.config import Config
from src.utils.rate_limiter import get_search_rate_limiter
from src.database.repository import ResearchRepository


class SearchExecutorNode:
    """
    Executes search queries and collects results.
    Uses SearchOrchestrator (SerpApi primary, Brave fallback) with RateLimiter.
    """

    def __init__(self, config: Config, repository: ResearchRepository):
        """
        Initialize the search executor with SearchOrchestrator and RateLimiter.

        Args:
            config: Configuration object with all settings
            repository: Database repository for persistence
        """
        self.config = config
        self.repository = repository
        self.orchestrator = SearchOrchestrator(config)
        self.logger = get_logger(__name__)

        self.max_results_per_query = config.search.max_results_per_query
        self.max_concurrent_searches = config.performance.max_concurrent_search_calls

        # Initialize rate limiter for search API
        self.rate_limiter = get_search_rate_limiter(max_concurrent=self.max_concurrent_searches)


    def execute(self, state: Dict) -> Dict:
        """
        Execute all queries in state['next_queries'].

        Args:
            state: Contains next_queries: List[str]

        Returns:
            Updated state with:
                - raw_search_results: List[SearchResult] (for next node)
                - search_history: Updated with new queries
                - current_iteration: Incremented by 1
        """
        next_queries = state.get("next_queries", [])
        current_iteration = state.get("current_iteration", 1)
        search_history = state.get("search_history", [])

        if not next_queries:
            self.logger.warning("No queries to execute")
            state["raw_search_results"] = []
            return state

        self.logger.info(
            f"Executing {len(next_queries)} queries for iteration {current_iteration}"
        )

        # Execute queries (with parallel execution)
        start_time = time.time()
        results = self._execute_queries(next_queries)
        execution_time = (time.time() - start_time) * 1000  # Convert to ms

        self.logger.info(f"Retrieved {len(results)} total results in {execution_time:.0f}ms")

        # Deduplicate results
        deduplicated_results = self._deduplicate_results(results)
        self.logger.info(
            f"Deduplicated to {len(deduplicated_results)} unique results"
        )

        # Calculate metrics
        source_diversity = self._calculate_source_diversity(deduplicated_results)
        relevance_score = self._calculate_relevance_score(deduplicated_results)

        # Update search history
        search_history = self._update_search_history(
            search_history,
            next_queries,
            len(deduplicated_results),
            relevance_score,
            current_iteration
        )

        # Save search queries to database
        session_id = state.get("session_id")
        if session_id:
            try:
                for query in next_queries:
                    self.repository.save_search_query(
                        session_id=session_id,
                        query=query,
                        iteration=current_iteration,
                        search_engine="serpapi",
                        results_count=len(deduplicated_results) // len(next_queries) if next_queries else 0,
                        relevance_score=relevance_score,
                        execution_time_ms=int(execution_time)
                    )
                self.logger.info(f"Saved {len(next_queries)} search queries to database")

            except Exception as e:
                self.logger.error(f"Failed to save search queries to database: {e}", exc_info=True)
                # Don't fail the workflow if DB save fails

        # Update state
        state["raw_search_results"] = deduplicated_results
        state["search_history"] = search_history
        state["current_iteration"] = current_iteration + 1

        # Track explored topics
        if "explored_topics" not in state:
            state["explored_topics"] = set()

        # Add query keywords to explored topics
        for query in next_queries:
            keywords = self._extract_keywords(query)
            state["explored_topics"].update(keywords)

        self.logger.info(
            f"Search execution complete: {len(deduplicated_results)} results, "
            f"diversity={source_diversity:.2f}, relevance={relevance_score:.2f}"
        )

        return state

    def _execute_queries(self, queries: List[str]) -> List[SearchResult]:
        """
        Execute queries with parallel execution for speed.

        Args:
            queries: List of search queries

        Returns:
            All search results combined
        """
        all_results = []

        # Try parallel execution first
        try:
            with ThreadPoolExecutor(max_workers=self.max_concurrent_searches) as executor:
                future_to_query = {
                    executor.submit(self._execute_single_query, query): query
                    for query in queries
                }

                for future in as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        results = future.result(timeout=60)
                        all_results.extend(results)
                        self.logger.info(f"Query '{query}' returned {len(results)} results")
                    except Exception as e:
                        self.logger.error(f"Query '{query}' failed: {e}")
                        # Continue with other queries

        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}, falling back to serial")
            # Fallback to serial execution
            for query in queries:
                try:
                    results = self._execute_single_query(query)
                    all_results.extend(results)
                except Exception as e:
                    self.logger.error(f"Query '{query}' failed: {e}")
                    continue

        return all_results

    def _execute_single_query(self, query: str) -> List[SearchResult]:
        """
        Execute a single search query with rate limiting.

        Args:
            query: Search query string

        Returns:
            List of search results
        """
        try:
            self.logger.debug(f"Executing query: {query}")

            # Use rate limiter to control concurrent API calls
            with self.rate_limiter.acquire():
                results = self.orchestrator.search(query, max_results=self.max_results_per_query)

            return results
        except Exception as e:
            self.logger.error(f"Error executing query '{query}': {e}")
            return []

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate URLs and near-duplicate content.

        Args:
            results: List of search results

        Returns:
            Deduplicated results
        """
        seen_urls = set()
        seen_content_sigs = set()
        deduplicated = []

        for result in results:
            # Check URL
            if result.url in seen_urls:
                continue

            # Check content similarity (first 100 chars as signature)
            content_sig = result.content[:100].lower().strip()

            if content_sig and content_sig in seen_content_sigs:
                continue

            # Add to deduplicated list
            seen_urls.add(result.url)
            if content_sig:
                seen_content_sigs.add(content_sig)

            deduplicated.append(result)

        return deduplicated

    def _calculate_source_diversity(self, results: List[SearchResult]) -> float:
        """
        Calculate source diversity score based on unique domains.

        Args:
            results: List of search results

        Returns:
            Diversity score from 0.0 to 1.0
        """
        if not results:
            return 0.0

        # Get unique domains
        unique_domains = set(r.source_domain for r in results if r.source_domain)

        # Calculate diversity (ratio of unique domains to total results)
        diversity = len(unique_domains) / len(results)

        return min(1.0, diversity)

    def _calculate_relevance_score(self, results: List[SearchResult]) -> float:
        """
        Calculate average relevance score with dynamic normalization.
        Adapts to different search engine scoring systems.

        Args:
            results: List of search results

        Returns:
            Average relevance score from 0.0 to 1.0
        """
        if not results:
            return 0.0

        # Extract scores
        scores = [r.score for r in results]

        # Use dynamic max score from actual results
        max_score = max(scores) if scores else 1.0

        # Avoid division by zero
        if max_score == 0:
            # All scores are 0 (perfect relevance)
            return 1.0

        # Normalize using actual max
        # Lower score = more relevant (search engine convention)
        total_score = 0.0
        for result in results:
            normalized = 1.0 - (result.score / max_score)
            total_score += normalized

        avg_score = total_score / len(results)

        self.logger.debug(
            f"Relevance scoring: max_score={max_score}, "
            f"avg_relevance={avg_score:.3f}"
        )

        return avg_score

    def _update_search_history(
        self,
        search_history: List[Dict],
        queries: List[str],
        results_count: int,
        relevance_score: float,
        iteration: int
    ) -> List[Dict]:
        """
        Update search history with new queries.

        Args:
            search_history: Current search history
            queries: Executed queries
            results_count: Total results count
            relevance_score: Average relevance score
            iteration: Current iteration

        Returns:
            Updated search history
        """
        timestamp = datetime.now()

        for query in queries:
            search_entry = {
                "query": query,
                "iteration": iteration,
                "timestamp": timestamp,
                "results_count": results_count,
                "relevance_score": relevance_score
            }
            search_history.append(search_entry)

        return search_history

    def _extract_keywords(self, query: str) -> set:
        """
        Extract keywords from query for topic tracking.

        Args:
            query: Search query

        Returns:
            Set of keywords
        """
        # Remove quotes and common words
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'about'
        }

        # Clean query
        cleaned = query.replace('"', '').lower()

        # Extract words (3+ chars)
        words = [
            w.strip() for w in cleaned.split()
            if len(w.strip()) >= 3 and w.strip() not in stopwords
        ]

        return set(words)
