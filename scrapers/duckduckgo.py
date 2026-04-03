"""
Web search-based job scraper using DuckDuckGo.

Searches across multiple job sites (LinkedIn, Glassdoor, Magnet.me, etc.)
without needing individual scrapers for each platform.
"""

import logging
import random
import re
import time

from scrapers import Job

logger = logging.getLogger(__name__)

# Domains that are definitely job-related and Dutch/relevant
ALLOWED_DOMAINS = {
    "linkedin.com", "nl.linkedin.com",
    "glassdoor.nl", "glassdoor.com",
    "indeed.nl", "nl.indeed.com",
    "magnet.me",
    "werkenbij.nl", "werkeninai.nl", "aivacatures.com",
    "wellfound.com",
    "yer.nl",
    "nationalevacaturebank.nl",
    "intermediair.nl",
    "builtin.com",
    "werkenvoornederland.nl",
}

# Domains to always skip
BLOCKED_DOMAINS = {
    "zhihu.com", "baidu.com", "csdn.net",
    "gov.br", "receita.fazenda.gov.br",
    "youtube.com", "reddit.com",
    "wikipedia.org", "facebook.com",
}

# URL path patterns that indicate non-job pages
SKIP_URL_PATTERNS = [
    "/salary", "/review", "/company/", "/about",
    "/blog/", "/guide/", "/press", "/legal/",
]


def scrape_duckduckgo(
    queries: list[str],
    max_results_per_query: int = 8,
    delay_seconds: float = 3.0,
) -> list[Job]:
    """Search DuckDuckGo for job listings across multiple sites."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("Neither ddgs nor duckduckgo-search installed.")
            return []

    all_jobs: list[Job] = []
    seen_urls: set[str] = set()

    search_pairs = _build_search_queries(queries)

    for search_query, source_label in search_pairs:
        logger.info(f"DuckDuckGo: '{search_query}'")
        try:
            ddgs = DDGS()
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

            if not url or not title:
                continue

            if not _is_allowed_url(url):
                logger.debug(f"Skipping non-job URL: {url}")
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


def _build_search_queries(queries: list[str]) -> list[tuple[str, str]]:
    """Build effective search queries, forcing Dutch context."""
    pairs: list[tuple[str, str]] = []

    # Site-specific searches on key job boards
    site_searches = [
        ("site:nl.linkedin.com/jobs", "linkedin"),
        ("site:magnet.me vacature", "magnet"),
        ("site:glassdoor.nl vacature", "glassdoor"),
        ("site:nl.indeed.com vacature", "indeed"),
        ("site:werkeninai.nl", "werkeninai"),
    ]
    for query in queries[:5]:
        for site_filter, label in site_searches:
            pairs.append((f"{site_filter} {query}", label))

    # Open web search with strong Dutch anchoring
    for query in queries[:6]:
        pairs.append((f"{query} vacature Nederland Utrecht 2026", "web-nl"))

    # Shuffle for variety across daily runs, then cap
    random.shuffle(pairs)
    return pairs[:20]


def _is_allowed_url(url: str) -> bool:
    """Check if URL is from a known job site or a .nl domain."""
    url_lower = url.lower()

    for blocked in BLOCKED_DOMAINS:
        if blocked in url_lower:
            return False

    for pattern in SKIP_URL_PATTERNS:
        if pattern in url_lower:
            return False

    for domain in ALLOWED_DOMAINS:
        if domain in url_lower:
            return True

    # Allow any .nl domain (likely Dutch job-related)
    domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url_lower)
    if domain_match:
        domain = domain_match.group(1)
        if domain.endswith(".nl"):
            return True

    return False


def _extract_company(title: str, url: str) -> str:
    """Try to extract company name from title or URL."""
    if " bij " in title.lower():
        parts = title.lower().split(" bij ", 1)
        company = parts[-1].strip()
        if " | " in company:
            company = company.split(" | ")[0].strip()
        return company.title()
    if " | " in title:
        parts = title.split(" | ")
        if len(parts) >= 2:
            return parts[-2].strip() if len(parts) > 2 else parts[-1].strip()
    if " - " in title:
        parts = title.rsplit(" - ", 1)
        candidate = parts[-1].strip()
        if len(candidate) < 50 and candidate not in {"Indeed", "Glassdoor", "LinkedIn"}:
            return candidate
    return ""


def _extract_location(snippet: str) -> str:
    """Try to extract location from snippet."""
    dutch_cities = [
        "Utrecht", "Amsterdam", "Nieuwegein", "Amersfoort", "Hilversum",
        "Zeist", "De Bilt", "Maarssen", "Woerden", "Bunnik",
        "Rotterdam", "Den Haag", "Eindhoven", "Almere",
        "Hoofddorp", "Amstelveen", "Rijswijk", "Leidsche Rijn",
    ]
    snippet_lower = snippet.lower()
    for city in dutch_cities:
        if city.lower() in snippet_lower:
            return city
    return ""
