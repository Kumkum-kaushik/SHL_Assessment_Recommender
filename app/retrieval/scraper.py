"""
SHL catalog scraper using BeautifulSoup.

Scrapes assessment data from the SHL website and falls back to
curated seed data when scraping is unavailable or blocked.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SEED_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "seed_assessments.json"

# SHL product catalog entry points
CATALOG_URLS = [
    "https://www.shl.com/solutions/products/assessments/",
    "https://www.shl.com/solutions/products/",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# URL patterns that indicate product/assessment pages
PRODUCT_URL_PATTERNS = [
    "/solutions/products/assessments/",
    "/solutions/products/product/",
]

# URL fragments to skip (not product pages)
SKIP_URL_FRAGMENTS = [
    "/blog/", "/news/", "/events/", "/about/", "/careers/",
    "/contact/", "/partners/", "/legal/", "/privacy/",
    "#", "mailto:", "javascript:",
]


def scrape_catalog() -> list[dict]:
    """
    Attempt to scrape the SHL catalog. Returns list of assessment dicts.
    Returns empty list on failure — caller should fall back to seed data.
    """
    visited = set()
    assessments = []

    for start_url in CATALOG_URLS:
        try:
            logger.info("Scraping catalog from: %s", start_url)
            product_links = _get_product_links(start_url, visited)
            logger.info("Found %d product links", len(product_links))

            for url in product_links:
                if url in visited:
                    continue
                visited.add(url)
                assessment = _scrape_product_page(url)
                if assessment:
                    assessments.append(assessment)
                    logger.info("Scraped: %s", assessment["name"])
                time.sleep(1.0)  # Respectful crawl delay

        except Exception as exc:
            logger.warning("Failed to scrape %s: %s", start_url, exc)

    logger.info("Scraper collected %d assessments", len(assessments))
    return assessments


def _get_product_links(url: str, visited: set) -> list[str]:
    """Fetch a catalog page and extract product page links."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("GET %s failed: %s", url, exc)
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    links = []

    for tag in soup.find_all("a", href=True):
        href: str = tag["href"]

        # Normalise relative URLs
        if href.startswith("/"):
            href = "https://www.shl.com" + href
        if not href.startswith("http"):
            continue

        # Skip non-SHL and blacklisted fragments
        if "shl.com" not in href:
            continue
        if any(skip in href for skip in SKIP_URL_FRAGMENTS):
            continue
        if href in visited:
            continue

        # Only follow assessment/product URLs
        if any(pat in href for pat in PRODUCT_URL_PATTERNS):
            links.append(href)

    return list(dict.fromkeys(links))  # deduplicate, preserve order


def _scrape_product_page(url: str) -> Optional[dict]:
    """Scrape a single SHL product page and return structured data."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("GET %s failed: %s", url, exc)
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    name = _extract_title(soup)
    if not name or len(name) < 3:
        return None

    description = _extract_description(soup)
    duration = _extract_duration(soup, description)
    test_type = _infer_test_type(url, name, description)

    return {
        "name": name,
        "url": url,
        "description": description,
        "skills_measured": [],   # Hard to parse reliably; enriched via seed data
        "test_type": test_type,
        "duration": duration,
        "remote_testing": True,
        "adaptive": "adaptive" in description.lower() or "adaptive" in url.lower(),
        "suitable_for": [],
    }


def _extract_title(soup: BeautifulSoup) -> str:
    """Extract product title from page."""
    # Try common title selectors in order of preference
    for selector in [
        "h1.hero__title", "h1.product-hero__title", "h1.page-title",
        "h1", ".product-name", ".hero h1",
    ]:
        el = soup.select_one(selector)
        if el and el.get_text(strip=True):
            return el.get_text(strip=True)
    return ""


def _extract_description(soup: BeautifulSoup) -> str:
    """Extract the main product description."""
    for selector in [
        ".hero__description", ".product-hero__description",
        ".product-description", ".intro-text", "meta[name='description']",
    ]:
        el = soup.select_one(selector)
        if el:
            text = el.get("content") or el.get_text(separator=" ", strip=True)
            if text and len(text) > 30:
                return text[:800]

    # Fallback: first substantial paragraph
    for p in soup.find_all("p"):
        text = p.get_text(separator=" ", strip=True)
        if len(text) > 80:
            return text[:800]

    return ""


def _extract_duration(soup: BeautifulSoup, description: str) -> str:
    """Try to extract assessment duration from page content."""
    import re

    combined = description + " " + soup.get_text(separator=" ", strip=True)
    match = re.search(
        r"(\d+[\s-]*(?:to|-)\s*\d+\s*(?:min|minutes?)|"
        r"\d+\s*(?:min|minutes?))",
        combined,
        re.IGNORECASE,
    )
    return match.group(0).strip() if match else "Varies"


def _infer_test_type(url: str, name: str, description: str) -> str:
    """Infer test type from URL, name, and description."""
    combined = f"{url} {name} {description}".lower()

    if any(k in combined for k in ["personality", "opq", "motivation", "mq", "remoteworkq", "adept", "dependability"]):
        return "Personality"
    if any(k in combined for k in ["cognitive", "numerical", "verbal", "inductive", "deductive", "verify", "graduate"]):
        return "Cognitive"
    if any(k in combined for k in ["coding", "technical", "java", "python", "sql", "javascript"]):
        return "Technical Skills"
    if any(k in combined for k in ["situational", "sjt", "judgment"]):
        return "Situational Judgment"
    if any(k in combined for k in ["video", "virtual interview"]):
        return "Video Interview"
    if any(k in combined for k in ["sales", "customer", "contact center"]):
        return "Job-Focused"
    return "Assessment"


def load_seed_data() -> list[dict]:
    """Load curated seed assessments from data/seed_assessments.json."""
    try:
        with open(SEED_DATA_PATH, encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded %d seed assessments", len(data))
        return data
    except Exception as exc:
        logger.error("Failed to load seed data: %s", exc)
        return []


def load_or_scrape(use_seed: bool = False) -> list[dict]:
    """
    Main entry point for data loading.
    1. If use_seed=True, skip scraping and use seed data.
    2. Otherwise try scraping; fall back to seed data on failure.
    """
    if use_seed:
        logger.info("Using seed data (scraping skipped)")
        return load_seed_data()

    logger.info("Attempting to scrape SHL catalog...")
    scraped = scrape_catalog()

    if scraped:
        # Merge with seed data — seed data wins for fields not captured by scraper
        seed_by_url = {a["url"]: a for a in load_seed_data()}
        merged = []
        for item in scraped:
            seed = seed_by_url.get(item["url"], {})
            merged.append({**seed, **{k: v for k, v in item.items() if v}})
        return merged

    logger.warning("Scraping returned no results — falling back to seed data")
    return load_seed_data()
