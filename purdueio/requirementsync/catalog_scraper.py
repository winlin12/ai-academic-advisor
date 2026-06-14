from __future__ import annotations

import hashlib
import html
import json
import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://catalog.purdue.edu/"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}
IGNORED_TAGS = {"script", "style", "svg", "noscript"}
CATALOG_YEAR_RE = re.compile(r"\b(20\d{2})\s*[-–]\s*(\d{2}|\d{4})\b")
COURSE_RE = re.compile(r"\b([A-Z]{2,6})\s*(?:-| )?\s*(\d{3,5}[A-Z]?)\b")
CREDITS_RE = re.compile(
    r"\b(\d+(?:\.\d+)?(?:\s*(?:-|to)\s*\d+(?:\.\d+)?)?)\s*(?:credits?|cr\.)\b",
    re.IGNORECASE,
)
STOP_MARKERS = (
    "print degree planner",
    "the mypurdueplan",
    "university catalog",
    "help",
    "portfolio",
    "facebook",
    "twitter",
)
HEADING_KEYWORDS = (
    "requirement",
    "core",
    "elective",
    "concentration",
    "major",
    "college",
    "university",
    "foundation",
    "select",
    "choose",
    "credit",
)
GRADUATE_DEGREES = {"MS", "MA", "MBA", "MFA", "PHD", "EDD", "GRAD CERT"}
UNDERGRAD_DEGREES = (
    "BA",
    "BFA",
    "BLA",
    "BM",
    "BS",
    "BSC",
    "BSE",
    "BSN",
    "BSPH",
    "DVM",
    "PHARMD",
)


@dataclass(frozen=True)
class AnchorLink:
    text: str
    url: str


@dataclass
class CatalogPage:
    url: str
    html: str

    @property
    def source_hash(self) -> str:
        return hashlib.sha256(self.html.encode("utf-8", errors="replace")).hexdigest()


@dataclass
class ParsedProgram:
    catalog_year: int
    source_url: str
    source_hash: str
    school: str | None
    program_title: str
    degree_code: str | None
    variant: str | None
    parser_status: str
    blocks: list[dict[str, object]] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)

    def to_raw_json(self) -> dict[str, object]:
        return {
            "source_url": self.source_url,
            "source_hash": self.source_hash,
            "school": self.school,
            "program_title": self.program_title,
            "degree_code": self.degree_code,
            "variant": self.variant,
            "parser_status": self.parser_status,
            "blocks": self.blocks,
            "raw_lines": self.raw_lines,
        }


class AnchorParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[AnchorLink] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_map = {name.lower(): value for name, value in attrs}
        href = attrs_map.get("href")
        if not href:
            return
        self._href = href.strip()
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._href is None:
            return
        text = clean_text(" ".join(self._text))
        self.links.append(AnchorLink(text=text, url=urljoin(self.base_url, self._href)))
        self._href = None
        self._text = []


class TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def lines(self) -> list[str]:
        text = html.unescape("".join(self.parts))
        lines = [clean_text(line) for line in text.splitlines()]
        return [line for line in lines if line]


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    value = html.unescape(value)
    value = value.replace("\xa0", " ")
    return " ".join(value.split()).strip()


def normalize_course_code(subject: str, number: str) -> str:
    return f"{subject}{number}".replace(" ", "").replace("-", "").upper()


def extract_course_refs(text: str) -> list[dict[str, str | None]]:
    refs: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for match in COURSE_RE.finditer(text.upper()):
        subject = match.group(1)
        number = match.group(2)
        code = normalize_course_code(subject, number)
        if code in seen:
            continue
        seen.add(code)
        refs.append(
            {
                "course_code_text": f"{subject} {number}",
                "normalized_code": code,
                "credits_text": extract_credits_text(text),
                "raw_text": text,
            }
        )
    return refs


def extract_credits_text(text: str) -> str | None:
    match = CREDITS_RE.search(text)
    if not match:
        return None
    return match.group(0)


def parse_anchors(page: CatalogPage) -> list[AnchorLink]:
    parser = AnchorParser(page.url)
    parser.feed(page.html)
    return parser.links


def extract_text_lines(html_text: str) -> list[str]:
    parser = TextParser()
    parser.feed(html_text)
    return dedupe_adjacent(parser.lines())


def dedupe_adjacent(lines: Iterable[str]) -> list[str]:
    result: list[str] = []
    previous = ""
    for line in lines:
        if line != previous:
            result.append(line)
        previous = line
    return result


class CatalogFetcher:
    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: int = 60,
        retries: int = 2,
        sleep_seconds: float = 0.2,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self.retries = retries
        self.sleep_seconds = sleep_seconds

    def fetch(self, url: str) -> CatalogPage:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                request = Request(
                    url,
                    headers={
                        "Accept": "text/html,application/xhtml+xml",
                        "User-Agent": self.user_agent,
                    },
                )
                with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                    body = response.read()
                return CatalogPage(url=url, html=decode_html(body))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.sleep_seconds)
        raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def decode_html(body: bytes) -> str:
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return body.decode(encoding)
        except UnicodeDecodeError:
            continue
    return body.decode("utf-8", errors="replace")


def discover_catalog_entrypoint(
    fetcher: CatalogFetcher,
    *,
    base_url: str,
    catalog_year: str,
) -> tuple[str, int, str]:
    page = fetcher.fetch(base_url)
    links = parse_anchors(page)
    candidates: list[tuple[int, str, str]] = []

    for link in links:
        label = link.text
        match = CATALOG_YEAR_RE.search(label)
        if not match:
            continue
        start_year = int(match.group(1))
        if catalog_year != "current" and str(start_year) != str(catalog_year):
            continue
        if urlparse(link.url).netloc != urlparse(base_url).netloc:
            continue
        candidates.append((start_year, link.url, label))

    if candidates:
        start_year, url, label = max(candidates, key=lambda item: item[0])
        return url, start_year, label

    if catalog_year == "current":
        current_year = max((year for year, _, _ in candidates), default=0)
        if current_year:
            return base_url, current_year, f"{current_year}"

    try:
        requested_year = int(catalog_year)
    except ValueError:
        requested_year = 0
    return base_url, requested_year, catalog_year


def discover_program_urls(
    fetcher: CatalogFetcher,
    *,
    catalog_url: str,
    max_pages: int = 120,
) -> list[str]:
    parsed_catalog = urlparse(catalog_url)
    expected_catoid = first_query_value(catalog_url, "catoid")
    queue = [catalog_url]
    visited: set[str] = set()
    program_urls: dict[str, str] = {}

    while queue and len(visited) < max_pages:
        page_url = queue.pop(0)
        if page_url in visited:
            continue
        visited.add(page_url)

        page = fetcher.fetch(page_url)
        for link in parse_anchors(page):
            normalized = strip_fragment(link.url)
            parsed = urlparse(normalized)
            if parsed.netloc != parsed_catalog.netloc:
                continue
            if expected_catoid and first_query_value(normalized, "catoid") != expected_catoid:
                continue
            if parsed.path.endswith("/preview_program.php"):
                program_urls[normalized] = link.text
                continue
            if should_crawl_catalog_link(normalized, link.text):
                if normalized not in visited and normalized not in queue:
                    queue.append(normalized)

    return sorted(program_urls)


def first_query_value(url: str, key: str) -> str | None:
    values = parse_qs(urlparse(url).query).get(key)
    return values[0] if values else None


def strip_fragment(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl()


def should_crawl_catalog_link(url: str, text: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.path.endswith("/preview_program.php"):
        return True
    if not parsed.path.endswith(("/index.php", "/content.php")):
        return False
    lowered = text.lower()
    if any(token in lowered for token in ("login", "help", "print", "portfolio")):
        return False
    return any(
        token in lowered
        for token in ("program", "degree", "undergraduate", "college", "school", "catalog")
    )


def parse_program_page(page: CatalogPage, *, catalog_year: int) -> ParsedProgram | None:
    lines = extract_text_lines(page.html)
    full_text = "\n".join(lines)
    raw_title = find_program_heading(lines, page.html)
    program_title, degree_code, variant = split_program_title(raw_title)

    if not program_title:
        return None

    if not is_undergraduate_program(program_title, degree_code, full_text):
        return None

    school = find_school(lines)
    requirement_lines = slice_requirement_lines(lines)
    blocks = parse_requirement_blocks(requirement_lines)
    status = "parsed" if blocks else "no_requirements_found"

    return ParsedProgram(
        catalog_year=catalog_year,
        source_url=page.url,
        source_hash=page.source_hash,
        school=school,
        program_title=program_title,
        degree_code=degree_code,
        variant=variant,
        parser_status=status,
        blocks=blocks,
        raw_lines=requirement_lines[:400],
    )


def find_program_heading(lines: list[str], html_text: str) -> str:
    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", html_text, flags=re.IGNORECASE | re.DOTALL)
    if h1_match:
        title = clean_text(re.sub(r"<[^>]+>", " ", h1_match.group(1)))
        if title:
            return normalize_program_heading(title)

    for line in lines[:80]:
        if line.lower().startswith("program:"):
            return normalize_program_heading(line)

    title_match = re.search(
        r"<title[^>]*>(.*?)</title>",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if title_match:
        return normalize_program_heading(clean_text(title_match.group(1)))

    return ""


def normalize_program_heading(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"^Program:\s*", "", value, flags=re.IGNORECASE)
    value = re.split(r"\s+-\s+Purdue\s+University", value, maxsplit=1)[0]
    return value.strip(" -")


def split_program_title(value: str) -> tuple[str, str | None, str | None]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return "", None, None
    if len(parts) == 1:
        return parts[0], None, None

    degree = parts[-1]
    title = ", ".join(parts[:-1])
    variant = None
    variant_match = re.search(r"\(([^)]+)\)\s*$", title)
    if variant_match:
        variant = variant_match.group(1).strip()
        title = title[: variant_match.start()].strip()
    return title, degree, variant


def is_undergraduate_program(title: str, degree_code: str | None, full_text: str) -> bool:
    degree = clean_text(degree_code).upper().replace(".", "")
    if degree in GRADUATE_DEGREES:
        return False
    if degree.startswith(UNDERGRAD_DEGREES):
        return True
    if "BACHELOR" in title.upper() or "BACHELOR" in degree:
        return True
    text_prefix = full_text[:4000].upper()
    return "UNDERGRADUATE" in text_prefix and "GRADUATE PROGRAM" not in text_prefix


def find_school(lines: list[str]) -> str | None:
    for line in lines[:160]:
        match = re.search(r"\b((?:College|School|Department) of [A-Za-z &,-]+)", line)
        if match:
            return clean_text(match.group(1)).strip(" ,")
    return None


def slice_requirement_lines(lines: list[str]) -> list[str]:
    start = 0
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if any(
            marker in lowered
            for marker in ("degree requirements", "program requirements", "major requirements")
        ):
            start = idx
            break

    selected: list[str] = []
    for line in lines[start:]:
        lowered = line.lower()
        if selected and any(marker in lowered for marker in STOP_MARKERS):
            break
        selected.append(line)
    return selected


def parse_requirement_blocks(lines: list[str]) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    current = new_block("Requirements", 1)

    for line in lines:
        if is_heading_line(line):
            if current["rules"] or current["title"] != "Requirements":
                blocks.append(current)
            current = new_block(line, len(blocks) + 1)
            continue

        refs = extract_course_refs(line)
        if not refs:
            continue
        current["rules"].append(build_rule(line, refs, len(current["rules"]) + 1))

    if current["rules"] or current["title"] != "Requirements":
        blocks.append(current)

    for idx, block in enumerate(blocks, start=1):
        block["sort_order"] = idx
    return blocks


def new_block(title: str, sort_order: int) -> dict[str, object]:
    return {
        "title": title,
        "sort_order": sort_order,
        "credits_text": extract_credits_text(title),
        "rules": [],
        "raw_text": title,
    }


def is_heading_line(line: str) -> bool:
    if extract_course_refs(line):
        return False
    lowered = line.lower()
    if len(line) > 100 or line.endswith("."):
        return False
    return any(keyword in lowered for keyword in HEADING_KEYWORDS)


def build_rule(
    raw_text: str,
    refs: list[dict[str, str | None]],
    sort_order: int,
) -> dict[str, object]:
    lowered = raw_text.lower()
    is_choice = len(refs) > 1 and (" or " in lowered or "choose" in lowered or "select" in lowered)
    rule_type = "choice_group" if is_choice else "all_of" if len(refs) > 1 else "course"

    if is_choice:
        options = [
            {
                "option_index": idx,
                "label": ref["course_code_text"],
                "courses": [ref],
            }
            for idx, ref in enumerate(refs, start=1)
        ]
    else:
        options = [
            {
                "option_index": 1,
                "label": None,
                "courses": refs,
            }
        ]

    return {
        "sort_order": sort_order,
        "rule_type": rule_type,
        "choose_count": 1 if is_choice else None,
        "raw_text": raw_text,
        "options": options,
    }


def parsed_programs_to_json(programs: list[ParsedProgram]) -> str:
    return json.dumps([program.to_raw_json() for program in programs], indent=2)
