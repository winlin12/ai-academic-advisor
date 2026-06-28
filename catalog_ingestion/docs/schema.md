# Catalog Ingestion — Database Schema

PostgreSQL schema for the `catalog_ingestion` database.

## Tables

### `catalog_years`
One row per catalog year. catoid is the Acalog CMS identifier.

| Column       | Type    | Notes                        |
|--------------|---------|------------------------------|
| id           | UUID PK |                              |
| label        | text    | e.g. `2026-2027`             |
| catoid       | int     | Unique. From URL `?catoid=N` |
| start_year   | int     | e.g. 2026                    |
| end_year     | int     | e.g. 2027                    |
| is_archived  | bool    |                              |
| catalog_url  | text    | index.php?catoid=N           |
| created_at   | datetime|                              |
| updated_at   | datetime|                              |

---

### `source_pages`
Raw HTML snapshot of every fetched catalog page. Audit trail.

| Column          | Type    | Notes                           |
|-----------------|---------|---------------------------------|
| id              | UUID PK |                                 |
| catalog_year_id | UUID FK | → catalog_years                 |
| url             | text    | Full URL                        |
| page_type       | text    | course, program, listing, index |
| http_status     | int     |                                 |
| content_hash    | text    | SHA-256 of raw_html             |
| fetched_at      | datetime|                                 |
| raw_html        | text    | Full HTML                       |
| raw_text        | text    | Plain text (optional)           |
| parser_version  | text    |                                 |

Unique constraint on `(url, content_hash)`.

---

### `scrape_runs`
Audit log of ingestion runs.

| Column               | Type    | Notes                   |
|----------------------|---------|-------------------------|
| id                   | UUID PK |                         |
| catalog_year_id      | UUID FK |                         |
| started_at           | datetime|                         |
| finished_at          | datetime|                         |
| status               | text    | running, completed, failed |
| target_catalog_years | jsonb   | list of year labels     |
| pages_attempted      | int     |                         |
| pages_succeeded      | int     |                         |
| pages_failed         | int     |                         |
| parser_version       | text    |                         |
| notes                | text    |                         |

---

### `scrape_errors`
Per-URL errors within a scrape run.

| Column        | Type    |
|---------------|---------|
| id            | UUID PK |
| scrape_run_id | UUID FK |
| url           | text    |
| error_type    | text    |
| error_message | text    |
| created_at    | datetime|

---

### `colleges`
One row per college/school per catalog year.

| Column          | Type    |
|-----------------|---------|
| id              | UUID PK |
| catalog_year_id | UUID FK |
| name            | text    |
| source_page_id  | UUID FK |

Unique on `(catalog_year_id, name)`.

---

### `departments`
One row per department per catalog year.

| Column          | Type    |
|-----------------|---------|
| id              | UUID PK |
| catalog_year_id | UUID FK |
| college_id      | UUID FK |
| name            | text    |
| source_page_id  | UUID FK |

---

### `subjects`
Course subject codes (e.g. CS, MA, ECE).

| Column          | Type    |
|-----------------|---------|
| id              | UUID PK |
| catalog_year_id | UUID FK |
| code            | text    |
| name            | text    |
| source_page_id  | UUID FK |

Unique on `(catalog_year_id, code)`.

---

### `courses`
One row per course per catalog year. Key: `(catalog_year_id, course_code)`.

| Column           | Type    | Notes                      |
|------------------|---------|----------------------------|
| id               | UUID PK |                            |
| catalog_year_id  | UUID FK |                            |
| subject_id       | UUID FK |                            |
| subject_code     | text    | e.g. CS                    |
| course_number    | text    | e.g. 18000                 |
| course_code      | text    | e.g. CS 18000              |
| coid             | int     | Acalog coid parameter      |
| title            | text    |                            |
| description      | text    |                            |
| credit_hours_min | float   |                            |
| credit_hours_max | float   |                            |
| credit_hours_raw | text    | Original credit text       |
| prerequisites_raw| text    | Raw prerequisite text      |
| corequisites_raw | text    |                            |
| restrictions_raw | text    |                            |
| attributes_raw   | text    |                            |
| source_page_id   | UUID FK |                            |

---

### `course_aliases`
Cross-listings, old/new code mappings.

| Column     | Type    |
|------------|---------|
| id         | UUID PK |
| course_id  | UUID FK |
| alias_code | text    |
| reason     | text    |

---

### `prerequisite_rules`
Parsed prerequisite structure. Raw text is always stored.

| Column           | Type    | Notes                        |
|------------------|---------|------------------------------|
| id               | UUID PK |                              |
| course_id        | UUID FK |                              |
| raw_text         | text    | Never null                   |
| parsed_json      | jsonb   | AND/OR tree; null if failed  |
| parse_confidence | text    | high, medium, low            |
| parser_notes     | text    | Warnings/errors from parser  |

---

### `programs`
One row per program per catalog year. Key: `(catalog_year_id, name, degree_type)`.

| Column           | Type    | Notes                          |
|------------------|---------|--------------------------------|
| id               | UUID PK |                                |
| catalog_year_id  | UUID FK |                                |
| college_id       | UUID FK |                                |
| department_id    | UUID FK |                                |
| name             | text    | e.g. Computer Science: MI      |
| degree_type      | text    | BS, BA, MS, PHD, Minor, etc.   |
| program_type     | text    | undergraduate_major, graduate, |
|                  |         | minor, certificate, tsap, etc. |
| campus           | text    | West Lafayette, Indianapolis   |
| total_credits_raw| text    |                                |
| total_credits_min| float   |                                |
| description      | text    |                                |
| poid             | int     | Acalog poid parameter          |
| source_page_id   | UUID FK |                                |

---

### `requirement_groups`
Sections within a program's degree requirements. Self-referencing for nesting.

| Column           | Type    | Notes                          |
|------------------|---------|--------------------------------|
| id               | UUID PK |                                |
| program_id       | UUID FK |                                |
| parent_group_id  | UUID FK | Self-ref; null for top-level   |
| name             | text    | Section heading                |
| requirement_type | text    | major, college, university,    |
|                  |         | gen_ed, core, elective, etc.   |
| credits_min      | float   |                                |
| credits_max      | float   |                                |
| raw_text         | text    | Full original section text     |
| display_order    | int     |                                |
| source_page_id   | UUID FK |                                |

---

### `requirement_options`
Individual course entries within a requirement group.

| Column               | Type    | Notes                          |
|----------------------|---------|--------------------------------|
| id                   | UUID PK |                                |
| requirement_group_id | UUID FK |                                |
| course_id            | UUID FK | Resolved FK; null if not found |
| course_code_raw      | text    | As printed in catalog          |
| option_text          | text    | Full line from catalog         |
| credits              | float   |                                |
| minimum_grade        | text    |                                |
| is_required          | bool    | false = selective/elective     |
| is_selective_option  | bool    | true = "choose from this list" |
| display_order        | int     |                                |

---

### `program_notes`
Unstructured notes from program pages (GPA requirements, residency, etc.).

| Column         | Type    | Notes                          |
|----------------|---------|--------------------------------|
| id             | UUID PK |                                |
| program_id     | UUID FK |                                |
| note_text      | text    |                                |
| note_type      | text    | parser_warning, gpa, residency |
| source_page_id | UUID FK |                                |

---

### PurdueAPI Staging Tables

Three staging tables for PurdueAPI import (Phase 5):
- `purdueapi_courses_staging`
- `purdueapi_subjects_staging`
- `purdueapi_terms_staging`

These hold raw OData JSON and normalized fields for cross-referencing with scraped data.
They are TRUNCATED and reimported each time `import-purdueapi` runs.

---

## Relationship Overview

```
catalog_years
  ├── source_pages
  ├── scrape_runs → scrape_errors
  ├── colleges → departments
  ├── subjects → courses
  │              ├── course_aliases
  │              └── prerequisite_rules
  └── programs
       ├── requirement_groups (self-ref tree)
       │    └── requirement_options → courses
       └── program_notes
```
