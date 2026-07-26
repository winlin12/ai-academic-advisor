-- ============================================================================
-- 003_course_prerequisites — the prerequisite edges Purdue publishes nowhere else.
--
-- Neither the Acalog catalog nor the PurdueIO API carries prerequisites (TODO.md
-- §2.5). They exist only on Banner course-detail pages, whose robots.txt
-- disallows crawling — so this table is populated by a deliberate, low-volume,
-- cached crawl (services/prerequisites/sync.py), not an unattended service.
--
-- BOTH representations are stored on purpose:
--   * parsed_tree  — the full AND/OR/COURSE tree, keeping minimum grades and
--                    "may be taken concurrently". Nothing the source stated is lost.
--   * prereq_groups — a flat AND-of-OR-groups the deterministic planner can
--                    evaluate: [["CS 25000"],["CS 25100","CS 25300"]] = "CS 25000
--                    AND (one of CS 25100 / CS 25300)". NULL when the tree could
--                    not be flattened honestly (a nested AND inside an OR), so a
--                    reader never confuses "too complex to flatten" with "no prereq".
--   * raw_text     — the cleaned human expression, always kept, so a low-confidence
--                    parse can be re-checked by eye without re-fetching.
--
-- Lives in the purdueio `advisor` schema alongside the crawled degree requirements
-- and the observed offerings (002), following the same convention: numbered
-- migration + a sync-runs table for provenance.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS advisor;

CREATE TABLE IF NOT EXISTS advisor.prereq_sync_runs (
    id             uuid PRIMARY KEY,
    term_code      text        NOT NULL,   -- Banner term the pages were read for
    status         text        NOT NULL,   -- 'running' | 'completed' | 'failed'
    started_at     timestamptz NOT NULL DEFAULT now(),
    completed_at   timestamptz,
    courses_seen   integer     NOT NULL DEFAULT 0,
    courses_parsed integer     NOT NULL DEFAULT 0,   -- high/medium confidence
    from_cache     integer     NOT NULL DEFAULT 0,
    errors         jsonb       NOT NULL DEFAULT '[]'::jsonb,
    metadata       jsonb       NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS advisor.course_prerequisites (
    course_code   text PRIMARY KEY,        -- canonical 'CS 38100'
    subject       text NOT NULL,
    number        text NOT NULL,
    term_code     text NOT NULL,           -- the term the prereqs were read from
    has_prereqs   boolean NOT NULL,        -- false = page had no prerequisite block (a real fact)
    raw_text      text,                    -- cleaned expression, always kept
    parsed_tree   jsonb,                   -- AND/OR/COURSE tree, NULL if unparseable
    prereq_groups jsonb,                   -- flat AND-of-ORs, NULL if unflattenable
    confidence    text NOT NULL,           -- 'high' | 'medium' | 'low' | 'none'
    notes         jsonb NOT NULL DEFAULT '[]'::jsonb,
    sync_run_id   uuid REFERENCES advisor.prereq_sync_runs(id) ON DELETE SET NULL,
    updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_course_prerequisites_subject ON advisor.course_prerequisites (subject);
CREATE INDEX IF NOT EXISTS ix_course_prerequisites_conf    ON advisor.course_prerequisites (confidence);

INSERT INTO advisor.schema_migrations (version, applied_at)
VALUES ('003_course_prerequisites', now())
ON CONFLICT (version) DO NOTHING;
