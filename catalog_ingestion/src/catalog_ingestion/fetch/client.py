"""HTTP fetcher with caching, rate-limiting, robots.txt compliance, and optional Playwright."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from catalog_ingestion.fetch.cache import CachedPage, DiskCache, content_hash
from catalog_ingestion.fetch.robots import get_crawl_delay, is_allowed

logger = logging.getLogger(__name__)

PARSER_VERSION = "1.0"

_WAF_INDICATOR_STATUS = 202
_WAF_HEADER_KEY = "x-amzn-waf-action"


class WafChallengeError(RuntimeError):
    """Raised when the site returns an AWS WAF JS challenge instead of real content."""


class RobotsDisallowedError(RuntimeError):
    """Raised when robots.txt disallows fetching a URL."""


class FetchError(RuntimeError):
    """General fetch error after all retries exhausted."""


class FetchedPage:
    def __init__(self, url: str, html: str, http_status: int, from_cache: bool = False) -> None:
        self.url = url
        self.html = html
        self.http_status = http_status
        self.from_cache = from_cache
        self.content_hash = content_hash(html)
        self.fetched_at = datetime.utcnow()


class HttpxFetcher:
    """Simple HTTP fetcher using httpx. Works for CDN-cached pages (e.g. index.php).

    Most catalog content pages (content.php, preview_*.php) are protected by AWS WAF
    JavaScript challenge. Those pages return HTTP 202 with a JS challenge page instead
    of real content. Use PlaywrightFetcher for those pages.
    """

    def __init__(
        self,
        *,
        user_agent: str,
        base_url: str,
        crawl_delay: float = 120.0,
        timeout: float = 60.0,
        retries: int = 3,
        cache: DiskCache | None = None,
        dry_run: bool = False,
    ) -> None:
        self.user_agent = user_agent
        self.base_url = base_url
        self.crawl_delay = crawl_delay
        self.timeout = timeout
        self.retries = retries
        self.cache = cache
        self.dry_run = dry_run
        self._last_fetch_time: float = 0.0

    def fetch(self, url: str, *, skip_robots: bool = False) -> FetchedPage:
        if not skip_robots and not is_allowed(url, self.user_agent, self.base_url):
            raise RobotsDisallowedError(f"robots.txt disallows: {url}")

        if self.cache:
            cached = self.cache.get(url)
            if cached:
                logger.debug("Cache hit: %s", url)
                return FetchedPage(
                    url=cached.url,
                    html=cached.html,
                    http_status=cached.http_status,
                    from_cache=True,
                )

        if self.dry_run:
            logger.info("[dry-run] Would fetch: %s", url)
            return FetchedPage(url=url, html="", http_status=0)

        self._rate_limit()
        return self._fetch_with_retry(url)

    def _rate_limit(self) -> None:
        robots_delay = get_crawl_delay(self.base_url, self.user_agent)
        delay = max(self.crawl_delay, robots_delay or 0.0)
        elapsed = time.monotonic() - self._last_fetch_time
        if elapsed < delay:
            sleep_for = delay - elapsed
            logger.debug("Rate limiting: sleeping %.1fs", sleep_for)
            time.sleep(sleep_for)

    def _fetch_with_retry(self, url: str) -> FetchedPage:
        import httpx

        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                with httpx.Client(
                    timeout=self.timeout,
                    follow_redirects=True,
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                    },
                ) as client:
                    resp = client.get(url)
                    self._last_fetch_time = time.monotonic()

                    if (
                        resp.status_code == _WAF_INDICATOR_STATUS
                        and _WAF_HEADER_KEY in resp.headers
                    ):
                        raise WafChallengeError(
                            f"AWS WAF JavaScript challenge on {url}. "
                            "Set FETCHER_BACKEND=playwright to use a headless browser."
                        )

                    html = resp.text
                    page = FetchedPage(url=url, html=html, http_status=resp.status_code)
                    if self.cache:
                        self.cache.put(
                            CachedPage(
                                url=url,
                                html=html,
                                http_status=resp.status_code,
                                fetched_at=page.fetched_at,
                                content_hash=page.content_hash,
                            )
                        )
                    logger.info("Fetched %s [%d]", url, resp.status_code)
                    return page

            except WafChallengeError:
                raise
            except Exception as exc:
                last_exc = exc
                wait = 2**attempt
                logger.warning("Fetch attempt %d/%d failed for %s: %s (retry in %ds)",
                               attempt + 1, self.retries, url, exc, wait)
                time.sleep(wait)

        raise FetchError(f"All {self.retries} retries failed for {url}") from last_exc


class PlaywrightFetcher:
    """Headless Chromium fetcher via Playwright.

    Executes the AWS WAF JavaScript challenge and returns real catalog HTML.
    Playwright must be installed: pip install playwright && playwright install chromium
    """

    def __init__(
        self,
        *,
        user_agent: str,
        base_url: str,
        crawl_delay: float = 120.0,
        timeout: float = 60.0,
        cache: DiskCache | None = None,
        dry_run: bool = False,
        browser_executable: str = "",
    ) -> None:
        self.user_agent = user_agent
        self.base_url = base_url
        self.crawl_delay = crawl_delay
        self.timeout = timeout
        self.cache = cache
        self.dry_run = dry_run
        self.browser_executable = browser_executable or None
        self._last_fetch_time: float = 0.0
        self._browser = None
        self._playwright = None

    def __enter__(self) -> PlaywrightFetcher:
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        launch_kwargs: dict = {"headless": True}
        if self.browser_executable:
            launch_kwargs["executable_path"] = self.browser_executable
        self._browser = self._playwright.chromium.launch(**launch_kwargs)
        logger.info("Playwright Chromium started")
        return self

    def __exit__(self, *_) -> None:
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        logger.info("Playwright Chromium stopped")

    def fetch(self, url: str, *, skip_robots: bool = False) -> FetchedPage:
        if not skip_robots and not is_allowed(url, self.user_agent, self.base_url):
            raise RobotsDisallowedError(f"robots.txt disallows: {url}")

        if self.cache:
            cached = self.cache.get(url)
            if cached:
                logger.debug("Cache hit: %s", url)
                return FetchedPage(url=cached.url, html=cached.html,
                                   http_status=cached.http_status, from_cache=True)

        if self.dry_run:
            logger.info("[dry-run] Would fetch (Playwright): %s", url)
            return FetchedPage(url=url, html="", http_status=0)

        if self._browser is None:
            raise RuntimeError("PlaywrightFetcher must be used as a context manager")

        self._rate_limit()

        context = self._browser.new_context(
            user_agent=self.user_agent,
            java_script_enabled=True,
        )
        try:
            page = context.new_page()
            resp = page.goto(url, timeout=int(self.timeout * 1000), wait_until="networkidle")
            self._last_fetch_time = time.monotonic()
            status = resp.status if resp else 0
            html = page.content()
            page.close()
        finally:
            context.close()

        fetched = FetchedPage(url=url, html=html, http_status=status)
        if self.cache:
            self.cache.put(
                CachedPage(
                    url=url,
                    html=html,
                    http_status=status,
                    fetched_at=fetched.fetched_at,
                    content_hash=fetched.content_hash,
                )
            )
        logger.info("Playwright fetched %s [%d]", url, status)
        return fetched

    def _rate_limit(self) -> None:
        robots_delay = get_crawl_delay(self.base_url, self.user_agent)
        delay = max(self.crawl_delay, robots_delay or 0.0)
        elapsed = time.monotonic() - self._last_fetch_time
        if elapsed < delay:
            sleep_for = delay - elapsed
            logger.debug("Rate limiting: sleeping %.1fs", sleep_for)
            time.sleep(sleep_for)


def build_fetcher(
    *,
    backend: str,
    user_agent: str,
    base_url: str,
    crawl_delay: float,
    cache_dir: Path | None,
    dry_run: bool = False,
    browser_executable: str = "",
) -> HttpxFetcher | PlaywrightFetcher:
    cache = DiskCache(cache_dir) if cache_dir else None
    if backend == "playwright":
        return PlaywrightFetcher(
            user_agent=user_agent,
            base_url=base_url,
            crawl_delay=crawl_delay,
            cache=cache,
            dry_run=dry_run,
            browser_executable=browser_executable,
        )
    return HttpxFetcher(
        user_agent=user_agent,
        base_url=base_url,
        crawl_delay=crawl_delay,
        cache=cache,
        dry_run=dry_run,
    )
