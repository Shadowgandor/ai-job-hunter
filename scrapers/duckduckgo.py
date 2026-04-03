"""
Web search-based job scraper using DuckDuckGo.

Searches across multiple job sites (LinkedIn, Glassdoor, Magnet.me, etc.)
without needing individual scrapers for each platform.
"""

import logging
import time

from scrapers import Job

logger = logging.getLogger(__name__)

# Site-specific search prefixes to find jobs on specific platforms
SITE_FILTERS = [
    "site:linkedin.com/jobs",
    "site:magnet.me",
    "site:werkenbij.nl OR site:werkeninai.nl",
    "site:glassdoor.nl/Vacature",
    "",  # Open web search (no site filter)
]


def scrape_duckduckgo(
    queries: list[str],
    max_results_per_query: int = 10,
    delay_seconds: float = 3.0,
) -> list[Job]:
    """
    Search DuckDuckGo for job listings across multiple sites.
    
    Requires: pip install duckduckgo-search
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning(
            "duckduckgo-search not installed. "
            "Run: pip install duckduckgo-search"
        )
        return []

    all_jobs: list[Job] = []
    seen_urls: set[str] = set()

    # Build search queries: combine job queries with site filters
    search_pairs = []
    for query in queries:
        for site_filter in SITE_FILTERS:
            search_query = f"{query} Nederland Utrecht"
            if site_filter:
                search_query = f"{site_filter} {query} Nederland"
            search_pairs.append((search_query, site_filter or "web"))

    # Limit total searches to avoid rate limiting
    search_pairs = search_pairs[:30]

    for search_query, source_label in search_pairs:
        logger.info(f"DuckDuckGo: '{search_query}'")
        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(
                        search_query,
                        region="nl-nl",
                        max_results=max_results_per_query,
                    )
                )
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed for '{search_query}': {e}")
            time.sleep(delay_seconds)
            continue

        for result in results:
            url = result.get("href", "")
            title = result.get("title", "")
            snippet = result.get("body", "")

            # Skip non-job URLs
            if not url or not title:
                continue
            if any(skip in url for skip in ["/salary", "/review", "/company", "/about"]):
                continue

            if url not in seen_urls:
                seen_urls.add(url)
                all_jobs.append(
                    Job(
                        title=title,
                        company=_extract_company(title, url),
                        location=_extract_location(snippet),
                        url=url,
                        snippet=snippet[:500],
                        source=f"ddg:{source_label}",
                    )
                )

        time.sleep(delay_seconds)

    logger.info(f"DuckDuckGo: found {len(all_jobs)} unique results")
    return all_jobs


def _extract_company(title: str, url: str) -> str:
    """Try to extract company name from title or URL."""
    # Common pattern: "Job Title - Company | Platform"
    if " | " in title:
        parts = title.split(" | ")
        if len(parts) >= 2:
            # The company is often the second-to-last part
            return parts[-2].strip() if len(parts) > 2 else parts[-1].strip()
    if " - " in title:
        parts = title.rsplit(" - ", 1)
        return parts[-1].strip()
    if " bij " in title.lower():
        parts = title.lower().split(" bij ", 1)
        return parts[-1].strip().title()
    return ""


def _extract_location(snippet: str) -> str:
    """Try to extract location from snippet."""
    dutch_cities = [
        "Utrecht", "Amsterdam", "Nieuwegein", "Amersfoort", "Hilversum",
        "Zeist", "De Bilt", "Maarssen", "Woerden", "Bunnik",
        "Rotterdam", "Den Haag", "Eindhoven", "Almere",
    ]
    snippet_lower = snippet.lower()
    for city in dutch_cities:
        if city.lower() in snippet_lower:
            return city
    return ""
