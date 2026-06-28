"""robots.txt checking for the catalog fetcher."""

from __future__ import annotations

import logging
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

logger = logging.getLogger(__name__)

_parsers: dict[str, RobotFileParser] = {}


def _get_parser(base_url: str, user_agent: str) -> RobotFileParser:
    if base_url not in _parsers:
        parsed = urlparse(base_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
            logger.info("Loaded robots.txt from %s", robots_url)
        except Exception as exc:
            logger.warning("Could not read robots.txt from %s: %s", robots_url, exc)
        _parsers[base_url] = rp
    return _parsers[base_url]


def is_allowed(url: str, user_agent: str, base_url: str) -> bool:
    try:
        rp = _get_parser(base_url, user_agent)
        return rp.can_fetch(user_agent, url)
    except Exception as exc:
        logger.warning("robots.txt check failed for %s: %s — assuming allowed", url, exc)
        return True


def get_crawl_delay(base_url: str, user_agent: str) -> float | None:
    try:
        rp = _get_parser(base_url, user_agent)
        return rp.crawl_delay(user_agent)
    except Exception:
        return None
