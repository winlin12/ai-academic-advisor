"""Polite fetch of Purdue Banner course-detail pages.

READ THIS BEFORE RAISING THE RATE. ``selfservice.mypurdue.purdue.edu/robots.txt`` is a blanket
``User-agent: * / Disallow: /``. There is no documented public API for prerequisites — Banner
is the only machine-readable source Purdue publishes — so this crawler exists as a deliberate,
low-volume, project-owner decision, not an unattended service. Everything here is built to keep
it that way:

  * a real, contactable User-Agent (no spoofing a browser),
  * a mandatory crawl delay between requests (``--delay``, default 5s), never zero,
  * a persistent on-disk page cache so a re-run re-fetches nothing (``--cache-dir``),
  * single-threaded by construction — there is no concurrency knob to turn up.

If Purdue offers a data feed or asks this to stop, this is the one file to change. The parser
(``parser.py``) is deliberately network-free so it can be pointed at an official source instead
without touching any parsing logic.
"""

from __future__ import annotations

import hashlib
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

BASE = "https://selfservice.mypurdue.purdue.edu/prod/bwckctlg.p_disp_course_detail"
USER_AGENT = (
    "PurdueAcademicPlannerBot/1.0 (student hobby project; not affiliated with Purdue; "
    "contact: bingdaddycompany@gmail.com)"
)


class BannerFetcher:
    def __init__(
        self,
        *,
        term_code: str,
        cache_dir: Path,
        delay_s: float = 5.0,
        timeout_s: float = 45.0,
    ):
        self.term_code = term_code
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay_s = max(delay_s, 1.0)  # a zero delay is never allowed against this host
        self.timeout_s = timeout_s
        self._last_fetch = 0.0

    def _cache_path(self, subject: str, number: str) -> Path:
        key = f"{self.term_code}_{subject}_{number}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:20]
        return self.cache_dir / f"{key}_{digest}.html"

    def fetch(self, subject: str, number: str) -> tuple[str, bool]:
        """Return (html, from_cache). A cache hit spends no request and no delay."""
        cache_path = self._cache_path(subject, number)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8", errors="replace"), True

        # Rate limit only around a REAL fetch. Never against the cache.
        elapsed = time.monotonic() - self._last_fetch
        if elapsed < self.delay_s:
            time.sleep(self.delay_s - elapsed)

        params = urllib.parse.urlencode({
            "cat_term_in": self.term_code,
            "subj_code_in": subject,
            "crse_numb_in": number,
        })
        request = urllib.request.Request(
            f"{BASE}?{params}", headers={"User-Agent": USER_AGENT, "Accept": "text/html"}
        )
        self._last_fetch = time.monotonic()
        with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        cache_path.write_text(body, encoding="utf-8")
        return body, False
