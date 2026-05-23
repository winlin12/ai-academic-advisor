#!/usr/bin/env python3
"""Download catalog PDFs from a Purdue catalog index page using curl.

This script:
1. Fetches the index page HTML with curl.
2. Finds PDF links on that page.
3. Optionally follows one hop into linked pages to find additional PDF links.
4. Downloads each discovered PDF with curl.

Example:
    python purdueio/curl_catalog_pdfs.py \
      --index-url "https://catalog.purdue.edu/content.php?catoid=18&navoid=23275" \
      --output-dir purdueio/data/catalog_pdfs \
      --one-hop
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urljoin, urlparse


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

SKIP_LINK_TEXT = {
    "add to my favorites",
    "share this page",
    "print",
    "help",
}

PDF_URL_RE = re.compile(r"\.pdf(?:$|[?#])", re.IGNORECASE)
YEAR_RE = re.compile(r"20\d{2}\s*[-–]\s*\d{2}")


@dataclass(frozen=True)
class AnchorLink:
    text: str
    url: str
    source_page: str


class AnchorParser(HTMLParser):
    """Extract anchor links and visible text from HTML."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[AnchorLink] = []
        self._current_href: str | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = {k.lower(): v for k, v in attrs}
        href = attr_map.get("href")
        if not href:
            return
        self._current_href = href.strip()
        self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_href is None:
            return
        self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a":
            return
        if self._current_href is None:
            return

        raw_text = "".join(self._text_parts)
        text = " ".join(raw_text.split())
        absolute_url = urljoin(self.base_url, self._current_href)
        self.links.append(AnchorLink(text=text, url=absolute_url, source_page=self.base_url))

        self._current_href = None
        self._text_parts = []


@dataclass(frozen=True)
class PdfCandidate:
    url: str
    source_page: str
    link_text: str


def run_curl_text(
    url: str,
    user_agent: str,
    timeout: int,
    referer: str | None,
    cookie: str | None,
) -> str:
    cmd = [
        "curl",
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--compressed",
        "--max-time",
        str(timeout),
        "--user-agent",
        user_agent,
    ]
    if referer:
        cmd.extend(["--referer", referer])
    if cookie:
        cmd.extend(["--cookie", cookie])
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=False)
    if result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"curl failed for {url}: {stderr_text}")

    stdout_bytes = result.stdout
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return stdout_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    # Should be unreachable because latin-1 can decode any byte sequence.
    return stdout_bytes.decode("utf-8", errors="replace")


def parse_anchors(html: str, base_url: str) -> list[AnchorLink]:
    parser = AnchorParser(base_url)
    parser.feed(html)
    return parser.links


def is_pdf_url(url: str) -> bool:
    return PDF_URL_RE.search(url) is not None


def should_follow_one_hop(link: AnchorLink, host: str) -> bool:
    parsed = urlparse(link.url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc != host:
        return False
    if is_pdf_url(link.url):
        return False

    text = link.text.strip().lower()
    if not text:
        return False
    if text in SKIP_LINK_TEXT:
        return False

    # Prioritize known catalog-ish links to avoid crawling the whole site nav.
    if YEAR_RE.search(link.text):
        return True
    keywords = (
        "catalog",
        "college",
        "school",
        "courses",
        "academic calendar",
        "codo",
        "major change",
        "honors",
        "exploratory",
        "polytechnic",
        "provost",
        "libraries",
        "indianapolis",
    )
    return any(keyword in text for keyword in keywords)


def discover_pdf_links(
    index_url: str,
    user_agent: str,
    timeout: int,
    one_hop: bool,
    sleep_seconds: float,
    referer: str | None,
    cookie: str | None,
) -> list[PdfCandidate]:
    index_html = run_curl_text(index_url, user_agent, timeout, referer, cookie)
    root_links = parse_anchors(index_html, index_url)

    host = urlparse(index_url).netloc
    discovered: dict[str, PdfCandidate] = {}

    def add_pdf(url: str, source_page: str, link_text: str) -> None:
        if not is_pdf_url(url):
            return
        if url in discovered:
            return
        discovered[url] = PdfCandidate(url=url, source_page=source_page, link_text=link_text)

    for link in root_links:
        add_pdf(link.url, index_url, link.text)

    if one_hop:
        hop_targets = [link for link in root_links if should_follow_one_hop(link, host)]
        visited_pages = {index_url}
        for target in hop_targets:
            if target.url in visited_pages:
                continue
            visited_pages.add(target.url)
            try:
                page_html = run_curl_text(
                    target.url,
                    user_agent,
                    timeout,
                    referer or index_url,
                    cookie,
                )
            except RuntimeError as exc:
                print(f"[warn] {exc}", file=sys.stderr)
                continue

            for inner in parse_anchors(page_html, target.url):
                add_pdf(inner.url, target.url, inner.text or target.text)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return sorted(discovered.values(), key=lambda item: item.url.lower())


def slugify(value: str) -> str:
    text = re.sub(r"\s+", " ", value.strip())
    text = re.sub(r"[^A-Za-z0-9._ -]", "", text)
    text = text.replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    return text.strip("._-")


def pick_filename(candidate: PdfCandidate, existing: set[str], index: int) -> str:
    parsed = urlparse(candidate.url)
    path_name = unquote(Path(parsed.path).name)

    if path_name.lower().endswith(".pdf"):
        base_name = slugify(path_name[:-4]) or f"catalog_pdf_{index:04d}"
    else:
        text_name = slugify(candidate.link_text)
        base_name = text_name or f"catalog_pdf_{index:04d}"

    filename = f"{base_name}.pdf"
    suffix = 2
    while filename in existing:
        filename = f"{base_name}__{suffix}.pdf"
        suffix += 1

    existing.add(filename)
    return filename


def download_pdfs(
    pdfs: Iterable[PdfCandidate],
    output_dir: Path,
    user_agent: str,
    timeout: int,
    referer: str | None,
    cookie: str | None,
    dry_run: bool,
) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_names: set[str] = set()
    manifest: list[dict[str, str]] = []

    for idx, candidate in enumerate(pdfs, start=1):
        filename = pick_filename(candidate, existing_names, idx)
        destination = output_dir / filename

        row = {
            "url": candidate.url,
            "saved_path": str(destination),
            "source_page": candidate.source_page,
            "link_text": candidate.link_text,
        }
        manifest.append(row)

        if dry_run:
            print(f"[dry-run] {candidate.url} -> {destination}")
            continue

        cmd = [
            "curl",
            "--location",
            "--fail",
            "--show-error",
            "--silent",
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "--max-time",
            str(timeout),
            "--user-agent",
            user_agent,
        ]
        if referer:
            cmd.extend(["--referer", referer])
        if cookie:
            cmd.extend(["--cookie", cookie])
        cmd.extend(["--output", str(destination), candidate.url])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[error] failed download: {candidate.url}", file=sys.stderr)
            if result.stderr.strip():
                print(f"        {result.stderr.strip()}", file=sys.stderr)
            continue

        print(f"[ok] {destination.name}")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover and download Purdue catalog PDFs from a catalog index page using curl."
        )
    )
    parser.add_argument(
        "--index-url",
        required=True,
        help="Catalog index page URL (for example the 'Catalog PDF & Archived Catalogs' page).",
    )
    parser.add_argument(
        "--output-dir",
        default="purdueio/data/catalog_pdfs",
        help="Directory where downloaded PDFs are saved.",
    )
    parser.add_argument(
        "--manifest",
        default="purdueio/data/catalog_pdfs/manifest.json",
        help="Path to write a JSON manifest of discovered/downloaded PDFs.",
    )
    parser.add_argument(
        "--one-hop",
        action="store_true",
        help="Follow one hop into catalog section pages to find additional PDF links.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.25,
        help="Delay between one-hop page fetches (default: 0.25).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Curl timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent header used by curl requests.",
    )
    parser.add_argument(
        "--referer",
        default=None,
        help="Optional referer header for curl requests.",
    )
    parser.add_argument(
        "--cookie",
        default=None,
        help="Optional cookie string for curl requests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover links and print intended downloads without downloading files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    index_url = args.index_url.strip()
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)

    try:
        pdf_candidates = discover_pdf_links(
            index_url=index_url,
            user_agent=args.user_agent,
            timeout=args.timeout,
            one_hop=args.one_hop,
            sleep_seconds=args.sleep_seconds,
            referer=args.referer or index_url,
            cookie=args.cookie,
        )
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if not pdf_candidates:
        print("[warn] no PDF links discovered")
        return 2

    print(f"Discovered {len(pdf_candidates)} PDF link(s)")
    manifest = download_pdfs(
        pdfs=pdf_candidates,
        output_dir=output_dir,
        user_agent=args.user_agent,
        timeout=args.timeout,
        referer=args.referer or index_url,
        cookie=args.cookie,
        dry_run=args.dry_run,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
