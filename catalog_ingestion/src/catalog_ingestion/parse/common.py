"""Shared HTML parsing utilities."""

from __future__ import annotations

import html as html_module
import re
from typing import Iterable

from bs4 import BeautifulSoup, Tag

BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl",
    "dt", "fieldset", "figcaption", "figure", "footer", "form",
    "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main",
    "nav", "ol", "p", "pre", "section", "table", "td", "th", "tr", "ul",
}
IGNORED_TAGS = {"script", "style", "svg", "noscript"}

CREDITS_RE = re.compile(
    r"\b(\d+(?:\.\d+)?(?:\s*(?:-|to)\s*\d+(?:\.\d+)?)?)\s*(?:credits?|cr\.)\b",
    re.IGNORECASE,
)
COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,6})\s+(\d{3,5}[A-Z]?)\b")


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    value = html_module.unescape(value)
    value = value.replace("\xa0", " ")
    return " ".join(value.split()).strip()


def soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def text_lines(html: str) -> list[str]:
    """Extract non-empty lines of text from HTML, preserving block-level breaks."""
    s = soup(html)
    for tag in s.find_all(IGNORED_TAGS):
        tag.decompose()

    lines: list[str] = []
    _collect(s, lines)
    result: list[str] = []
    prev = ""
    for line in lines:
        cleaned = clean_text(line)
        if cleaned and cleaned != prev:
            result.append(cleaned)
            prev = cleaned
    return result


def _collect(node, lines: list[str]) -> None:
    if isinstance(node, str):
        lines.append(node)
        return
    if not isinstance(node, Tag):
        return
    tag_name = node.name.lower() if node.name else ""
    if tag_name in IGNORED_TAGS:
        return
    if tag_name in BLOCK_TAGS:
        lines.append("\n")
    for child in node.children:
        _collect(child, lines)
    if tag_name in BLOCK_TAGS:
        lines.append("\n")


def extract_credits(text: str) -> tuple[float | None, float | None, str | None]:
    """Return (min, max, raw) credit hours from a text string."""
    match = CREDITS_RE.search(text)
    if not match:
        return None, None, None
    raw = match.group(0)
    num_str = match.group(1)
    range_match = re.match(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)", num_str)
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2)), raw
    try:
        val = float(num_str.strip())
        return val, val, raw
    except ValueError:
        return None, None, raw


def extract_course_refs(text: str) -> list[tuple[str, str]]:
    """Return list of (subject_code, course_number) tuples found in text."""
    return [(m.group(1), m.group(2)) for m in COURSE_CODE_RE.finditer(text.upper())]
