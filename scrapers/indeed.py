"""
Indeed.nl job scraper using RSS feeds.

NOTE: Indeed frequently blocks RSS requests from cloud/datacenter IPs
(returns 403). This scraper will gracefully degrade — if RSS fails,
jobs are still picked up via the DuckDuckGo scraper's site:indeed.nl queries.
"""

import xml.etree.ElementTree as ET
import logging
import re
import time
from urllib.parse import quote_plus

import requests

from scrapers import Job

logger = logging.getLogger(__name__)

INDEED_RSS_URL = "https://nl.indeed.com/rss"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


def fetch_indeed_rss(query: str, location: str, radius_km: int = 40) -> list[Job]:
    """Fetch jobs from a single Indeed.nl RSS feed."""
    params = {
        "q": query,
        "l": location,
        "radius": str(radius_km),
        "sort": "date",
    }
    url = f"{INDEED_RSS_URL}?{'&'.join(f'{k}={quote_plus(v)}' for k, v in params.items())}"

    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        if resp.status_code == 403:
            # Expected from datacenter IPs — don't spam warnings
            logger.debug(f"Indeed RSS blocked (403) for '{query}' in '{location}'")
            return []
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.debug(f"Indeed RSS failed for '{query}' in '{location}': {e}")
        return []

    return _parse_rss(resp.text, query)


def _parse_rss(xml_text: str, query: str) -> list[Job]:
    """Parse Indeed RSS XML into Job objects."""
    jobs: list[Job] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning(f"Failed to parse Indeed RSS XML: {e}")
        return []

    channel = root.find("channel")
    if channel is None:
        return []

    for item in channel.findall("item"):
        title = _text(item, "title")
        link = _text(item, "link")
        description = _text(item, "description")

        company = ""
        if " - " in title:
            parts = title.rsplit(" - ", 1)
            title = parts[0].strip()
            company = parts[1].strip()

        location = ""
        if description:
            loc_match = re.match(r"^([A-Za-z\s\-]+(?:\([^)]+\))?)\s*[-–]", description)
            if loc_match:
                location = loc_match.group(1).strip()

        if title and link:
            jobs.append(
                Job(
                    title=title,
                    company=company,
                    location=location,
                    url=link,
                    snippet=_clean_html(description or ""),
                    source=f"indeed-rss:{query}",
                )
            )

    return jobs


def _text(element: ET.Element, tag: str) -> str:
    child = element.find(tag)
    return (child.text or "").strip() if child is not None else ""


def _clean_html(text: str) -> str:
    """Remove HTML tags from text."""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()[:500]


def scrape_indeed(
    queries: list[str],
    locations: list[str],
    radius_km: int = 40,
    delay_seconds: float = 2.0,
) -> list[Job]:
    """
    Fetch jobs from Indeed.nl across multiple queries and locations.
    Gracefully handles 403 blocks from datacenter IPs.
    """
    all_jobs: list[Job] = []
    seen_urls: set[str] = set()
    blocked = False

    for query in queries:
        if blocked:
            break
        for location in locations:
            jobs = fetch_indeed_rss(query, location, radius_km)

            if not jobs and not blocked:
                # After first empty result, check if we're being blocked
                # If the first query returns nothing, likely all will fail
                blocked = True
                logger.info(
                    "Indeed RSS appears blocked from this IP. "
                    "Indeed jobs will be picked up via DuckDuckGo instead."
                )
                break

            for job in jobs:
                if job.url not in seen_urls:
                    seen_urls.add(job.url)
                    all_jobs.append(job)

            time.sleep(delay_seconds)

    logger.info(f"Indeed RSS: found {len(all_jobs)} unique jobs")
    return all_jobs
