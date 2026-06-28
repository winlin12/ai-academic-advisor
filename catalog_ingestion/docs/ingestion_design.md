# Catalog Ingestion — Design Notes

## Phase 0 Research Findings

### Source: catalog.purdue.edu

**Catalog system:** Acalog CMS (hosted by Digarc)
**Catalog years discovered:**

| Label     | catoid | Status   |
|-----------|--------|----------|
| 2026-2027 | 19     | Current  |
| 2025-2026 | 18     | Archived |
| 2024-2025 | 17     | Archived |
| 2023-2024 | 16     | Archived |
| 2022-2023 | 15     | Archived |
| 2021-2022 | 14     | Archived |
| 2020-2021 | 13     | Archived |
| 2019-2020 | 10     | Archived |
| 2018-2019 | 9      | Archived |
| 2017-2018 | 8      | Archived |
| 2016-2017 | 7      | Archived |
| 2015-2016 | 6      | Archived |
| 2014-2015 | 4      | Archived |

**Key URL patterns:**
- Catalog home: `/index.php?catoid=N`
- Content pages: `/content.php?catoid=N&navoid=XXXXX`
- Individual course: `/preview_course_nopop.php?catoid=N&coid=XXXXX`
- Individual program: `/preview_program.php?catoid=N&poid=XXXXX`

**Key navoids for catoid=19 (2026-2027):**
- `navoid=25468` — Courses filter page
- `navoid=25420` — Course listing
- `navoid=25586` — Graduate and Undergraduate Programs Lists
- `navoid=25588` — Academic Colleges
- `navoid=25582` — Academic Regulations
- `navoid=25693` — Undergraduate Requirements

**robots.txt findings:**
- All bots: `Crawl-delay: 120`
- Disallowed: `/portfolio.php`, `/portfolio_nopop.php`, `/ajax/`, `/search_advanced.php`
- All catalog content pages are allowed

**AWS WAF finding:**
`content.php`, `preview_program.php`, and `preview_course_nopop.php` return HTTP 202 with an
AWS WAF JavaScript challenge when accessed without a real browser. Only `index.php` is served
directly from CDN (cache-hit, no WAF check).

**Implication:** The fetcher MUST use a headless browser (Playwright) to access catalog
content pages. Using `FETCHER_BACKEND=playwright` in `.env` is required for all scraping
beyond catalog year discovery.

This is NOT bypassing a bot check — Playwright runs the actual JavaScript challenge correctly.

---

### Source: PurdueAPI (Purdue.io)

**What PurdueAPI provides:**
- Courses: subject, number, title, credit_hours, description
- Subjects: code, name
- Terms: code, name, start/end dates
- Classes, Sections, Meetings (scheduling data)
- Campuses, Buildings, Rooms
- Instructors

**What PurdueAPI does NOT provide (our gap):**
- Degree programs
- Degree requirements (major/minor/certificate requirements)
- Prerequisite text or structure
- Requirement groups or elective lists
- Plan of study / course sequences
- College-level requirements
- Program notes, grade requirements, residency requirements
- Catalog year versioning (PurdueAPI has terms, not catalog years)

**Schema concepts reused from PurdueAPI:**
- `Subjects` table structure (code, name) → our `subjects` table
- `Courses` table structure → our `courses` table
- Stable UUID approach for deduplication
- OData paging pattern for import (`$top`, `@odata.nextLink`)

**How we use PurdueAPI:**
1. Optional import into `purdueapi_*_staging` tables
2. Cross-reference course titles/credit_hours against scraped catalog data
3. Mismatch reports for data quality validation
4. NOT used as a replacement for catalog scraping — we scrape catalog independently

---

## Architecture Decisions

### Catalog Year Versioning
Every scraped entity (course, program, requirement) is tied to a specific `catalog_year_id`.
A `CS 18000` record in 2026-2027 is a separate database row from `CS 18000` in 2025-2026,
even if identical. This preserves the historical record and supports "what were requirements
when I enrolled?" queries.

### Source Page Preservation
Every fetched HTML page is stored in `source_pages` with:
- The full raw HTML (for reprocessing without refetching)
- A content hash (to detect unchanged pages)
- Timestamp and HTTP status

This is the audit trail. Parser bugs can be fixed and data reprocessed from cached HTML.

### Prerequisite Parsing
Raw prerequisite text is ALWAYS stored, regardless of parse success.
The `parse_confidence` field:
- `high` — parsed successfully, structure is unambiguous
- `medium` — parsed, but some heuristics applied
- `low` — could not parse; raw_text has the information; human review needed

The JSON structure follows:
```json
{"type": "AND|OR|COURSE|UNKNOWN", "children": [...], "course": "SUBJ NUM"}
```

### Requirement Group Hierarchy
Requirement pages often have nested sections. The `requirement_groups` table
supports a self-referencing `parent_group_id` for nesting.

A requirement group says "take some or all of these courses." The `requirement_options`
table holds individual course entries with `is_selective_option=true` for elective lists.

### Idempotency
All ingest functions use upsert logic:
- Courses: keyed on `(catalog_year_id, course_code)`
- Programs: keyed on `(catalog_year_id, name, degree_type)`
- Source pages: keyed on `(url, content_hash)`

Re-running ingestion is safe and only updates changed records.

---

## Fetcher Design

### HttpxFetcher (limited use)
Simple HTTP client. Works ONLY for:
- `index.php` (CDN-cached, no WAF)
- Any page added to CDN cache

Use for: catalog year discovery only.

### PlaywrightFetcher (recommended)
Headless Chromium via Playwright. Solves AWS WAF JS challenge.
Use for: all catalog content pages.

Setup:
```bash
pip install playwright
playwright install chromium
```

Set `FETCHER_BACKEND=playwright` in `.env`.

### Rate Limiting
The robots.txt specifies `Crawl-delay: 120` for all bots.
Default `CRAWL_DELAY_SECONDS=120` respects this.
Lower values may be used with cached pages or during development.

---

## Data Flow

```
index.php
  → discover_catalog_years()
      → catalog_years table

content.php?navoid=N (programs list)
  → discover_program_links()
      → preview_program.php?poid=N
          → parse_program_page() + parse_requirement_sections()
              → ingest_program() + _ingest_requirement_group()
                  → programs, requirement_groups, requirement_options, program_notes

content.php?navoid=N (courses filter)
  → discover_subjects_from_filter_page()
      → build_subject_filter_url() (per subject)
          → extract_course_links_from_listing()
              → preview_course_nopop.php?coid=N
                  → parse_course_page()
                      → ingest_course() + _upsert_prerequisite_rule()
                          → courses, prerequisite_rules, source_pages
```

---

## Phase 1 Implementation Findings (live-verified 2026-06-26)

These were confirmed against the live catoid=19 (2026-2027) catalog.

### Deployment: containerized, no host sudo required
The host (Ubuntu 26.04) cannot `apt-get install` Chromium's system libraries without
interactive sudo. Solution: run ingestion in a container built on
`mcr.microsoft.com/playwright/python:v1.60.0-noble` (Chromium + all libs preinstalled),
with Postgres as a sibling service, orchestrated by rootless **podman** via the
`docker compose` provider. Bring-up:
```
systemctl --user enable --now podman.socket   # one-time: enable rootless API socket
podman compose up -d postgres
podman compose build ingestion
podman compose run --rm ingestion db-init
podman compose run --rm ingestion discover-years
podman compose run --rm ingestion sync-courses --year 2026-2027 --subject CS
```
The `playwright` pip version MUST match the base-image tag (1.60.0) so the bundled
browser build is compatible.

### Corrected Acalog navigation (catoid=19)
- **Programs list**: `navoid=25484` ("Undergraduate Programs List", 959 program links).
  The old value 25586 is the *hub* page and has zero direct links.
- **Courses filter**: `navoid=25468`. Subject dropdown is
  `<select id="courseprefix" name="filter[27]">`; filter a subject via
  `filter%5B27%5D=CS&filter%5Bcpage%5D=N` (100 courses/page, paginated).

### Link encoding
Catalog hrefs are HTML-escaped (`?catoid=19&amp;poid=...`). They MUST be unescaped
before `parse_qs`/fetching, or the query param keys become `amp;poid` and the fetched
URL is malformed. Course/program references inside requirement lists are NOT plain
hrefs — they are `<a onclick="showCourse('<catoid>','<coid>',...)">` calls. The **coid
is the 2nd argument** and is captured into `requirement_options.coid` for stable linking.

### Program requirement structure (Acalog)
Requirements render as a flat series of `div.acalog-core` blocks; the heading tag
(h2/h3/h4) encodes nesting, which we rebuild into `requirement_groups.parent_group_id`.
Courses are `<li class="acalog-course">`. The "Sample 4-Year Plan" is captured as
`requirement_type=plan_of_study`; GPA/licensure/policy/disclaimer narrative blocks are
preserved as groups with `raw_text` (never discarded).

### Prerequisites are NOT in the Purdue catalog
**Key data-source finding:** Purdue's Acalog course pages (preview_course_nopop.php)
contain only credit hours, description, and learning outcomes — **no prerequisite,
corequisite, restriction, or minimum-grade text** (verified on CS 25100 Data Structures
and others). Purdue manages prerequisites in Banner / myPurduePlan, not the catalog.
The parser therefore correctly stores `prerequisites_raw = NULL` rather than inventing
structure (per spec: never hallucinate prerequisites). When a course page *does* carry a
labeled "Prerequisites:" section, `parse_course_page` extracts it and
`parse_prerequisite_text` builds the AND/OR tree. Sourcing prerequisites for Purdue will
require the scheduling system or a supplementary source — a documented gap, not a bug.

## Known Limitations (First Pass)

1. **navoid values for archived years** — only catoid=19 navoids are confirmed from live research.
   Older years may use different navoids. The system will discover them from the programs list page.

2. **Requirement parser accuracy** — degree requirement pages are highly variable.
   The first-pass parser stores everything as raw_text; progressive improvement is expected.

3. **Credit hours from requirement options** — credit hours per course in a requirement list
   are not always explicitly stated. The parser captures what it can.

4. **Cross-listings and aliases** — the `course_aliases` table is prepared but not populated
   by the first-pass scraper. Manual curation or a second scraping pass is needed.

5. **Graduate programs** — not filtered out; they will be scraped and stored.
   The `degree_type` and `program_type` fields distinguish them from undergraduate programs.
