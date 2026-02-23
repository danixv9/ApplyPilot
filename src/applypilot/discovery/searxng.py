"""SearXNG-based job discovery: aggregates Google, Bing, DuckDuckGo via local instance.

Queries a self-hosted SearXNG instance's JSON API for job postings,
extracts URLs from search results, and stores them in the ApplyPilot database.
No browser needed — pure HTTP.
"""

import logging
import sqlite3
import time
from datetime import datetime, timezone
from urllib.parse import quote_plus, urlparse

import requests

from applypilot import config
from applypilot.database import get_connection, init_db, store_jobs

log = logging.getLogger(__name__)

DEFAULT_SEARXNG_URL = "http://localhost:8888"

# Domains we actually want job links from
JOB_DOMAINS = {
    "linkedin.com", "indeed.com", "glassdoor.com", "ziprecruiter.com",
    "lever.co", "greenhouse.io", "boards.greenhouse.io",
    "jobs.lever.co", "myworkdayjobs.com", "wd3.myworkdayjobs.com",
    "wd5.myworkdayjobs.com", "icims.com", "smartrecruiters.com",
    "bamboohr.com", "ashbyhq.com", "wellfound.com", "angel.co",
    "dice.com", "remoteok.com", "weworkremotely.com",
    "himalayas.app", "remotive.com", "flexjobs.com",
    "builtin.com", "otta.com", "arc.dev", "nodesk.co",
    "workingnomads.com", "remoteco.com", "justremote.co",
    "careers-page.com", "jobvite.com", "taleo.net",
    "breezy.hr", "recruitee.com", "workable.com",
}

# Domains to skip entirely
SKIP_DOMAINS = {
    "youtube.com", "reddit.com", "quora.com", "medium.com",
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "tiktok.com", "wikipedia.org", "stackoverflow.com",
    "github.com", "turing.com", "developers.turing.com",
    "dataannotation.tech", "toptal.com", "upwork.com", "fiverr.com",
}


def _is_job_url(url: str) -> bool:
    """Check if a URL is likely a job posting (not a search results page)."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")

    if any(skip in domain for skip in SKIP_DOMAINS):
        return False

    # LinkedIn job view pages
    if "linkedin.com" in domain and "/jobs/view/" in url:
        return True

    # Indeed job pages
    if "indeed.com" in domain and ("/viewjob" in url or "/rc/clk" in url):
        return True

    # Workday job pages
    if "myworkdayjobs.com" in domain and "/job/" in url:
        return True

    # Greenhouse
    if "greenhouse.io" in domain and "/jobs/" in url:
        return True

    # Lever
    if "lever.co" in domain and len(parsed.path.strip("/").split("/")) >= 2:
        return True

    # Generic job board pages (not search/listing pages)
    path = parsed.path.lower()
    job_indicators = ["/job/", "/jobs/", "/position/", "/career/",
                      "/opening/", "/apply/", "/vacancy/"]
    listing_indicators = ["/search", "/results", "/browse", "/categories",
                          "/q-", "/l-"]

    if any(ind in path for ind in listing_indicators):
        return False

    if any(ind in path for ind in job_indicators):
        return True

    # Accept any URL from known job domains
    if any(jd in domain for jd in JOB_DOMAINS):
        return True

    return False


def _extract_site(url: str) -> str:
    """Extract a clean site name from a URL."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")

    if "linkedin.com" in domain:
        return "linkedin"
    if "indeed.com" in domain:
        return "indeed"
    if "glassdoor.com" in domain:
        return "glassdoor"
    if "myworkdayjobs.com" in domain:
        # Extract employer from subdomain
        parts = domain.split(".")
        if len(parts) >= 3:
            return parts[0].replace("-", " ").title()
        return "Workday"
    if "greenhouse.io" in domain or "boards.greenhouse.io" in domain:
        return "Greenhouse"
    if "lever.co" in domain:
        return "Lever"

    # Clean domain name
    return domain.split(".")[0].title()


def search_searxng(query: str, base_url: str = DEFAULT_SEARXNG_URL,
                   max_pages: int = 3) -> list[dict]:
    """Search SearXNG and return job results.

    Args:
        query: Search query string.
        base_url: SearXNG instance URL.
        max_pages: Number of result pages to fetch.

    Returns:
        List of job dicts with url, title, description.
    """
    jobs = []
    seen_urls = set()

    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "format": "json",
            "categories": "general",
            "pageno": page,
        }

        try:
            resp = requests.get(f"{base_url}/search", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning("SearXNG query failed (page %d): %s", page, e)
            break

        results = data.get("results", [])
        if not results:
            break

        for r in results:
            url = r.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            if not _is_job_url(url):
                continue

            title = r.get("title", "").strip()
            content = r.get("content", "").strip()

            if title:
                jobs.append({
                    "url": url,
                    "title": title,
                    "description": content[:500] if content else "",
                    "salary": "",
                    "location": "Remote",
                })

        # Be polite
        if page < max_pages:
            time.sleep(1)

    return jobs


def run_searxng_discovery(base_url: str = DEFAULT_SEARXNG_URL,
                          tiers: list[int] | None = None,
                          max_pages: int = 2) -> int:
    """Run full SearXNG discovery using configured search queries.

    Args:
        base_url: SearXNG instance URL.
        tiers: Which query tiers to run (default: [1, 2]).
        max_pages: Pages per query (default: 2).

    Returns:
        Total new jobs discovered.
    """
    init_db()
    conn = get_connection()
    search_config = config.load_search_config()

    queries = search_config.get("queries", [])
    if tiers is None:
        tiers = [1, 2]

    # Filter by tier
    active_queries = [q for q in queries if q.get("tier", 3) in tiers]

    total_new = 0
    total_dupes = 0

    log.info("SearXNG discovery: %d queries (tiers %s), %d pages each",
             len(active_queries), tiers, max_pages)

    for i, q_config in enumerate(active_queries, 1):
        query = q_config["query"]
        log.info("[%d/%d] Searching: %s", i, len(active_queries), query)

        jobs = search_searxng(query, base_url=base_url, max_pages=max_pages)

        if jobs:
            # Group by site for storage
            by_site: dict[str, list[dict]] = {}
            for job in jobs:
                site = _extract_site(job["url"])
                by_site.setdefault(site, []).append(job)

            for site, site_jobs in by_site.items():
                new, dupes = store_jobs(conn, site_jobs, site, "searxng")
                total_new += new
                total_dupes += dupes

            log.info("  -> %d results, %d job URLs found", len(jobs), len(jobs))
        else:
            log.info("  -> 0 results")

        # Rate limit between queries
        if i < len(active_queries):
            time.sleep(0.5)

    conn.commit()
    log.info("SearXNG discovery complete: %d new, %d duplicates",
             total_new, total_dupes)
    return total_new
