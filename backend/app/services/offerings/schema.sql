-- ============================================================================
-- 002_course_offerings — when is a course actually taught?
--
-- WHY THIS EXISTS. Purdue's Acalog catalog publishes neither prerequisites nor
-- term offerings (TODO.md §2.5), so until now `planner_catalog._course_from_row`
-- hard-coded every course as offered every term. That is not a small
-- simplification: term availability is most of what makes course sequencing
-- non-trivial, and a planner that believes everything is always available will
-- happily produce a schedule a student cannot register for.
--
-- PurdueIO (api.purdue.io) does not publish offerings as a field either — but it
-- publishes Classes, and a Class is a (Course, Term, Campus) triple that existed.
-- So offerings are OBSERVED, not declared: "CS 47100 is a fall course" is an
-- inference from having seen it every fall and never in a spring, across N terms
-- of history. That distinction is why `course_offering_patterns` records the
-- observation COUNTS and not just booleans — a pattern derived from one term is
-- a guess, and the planner deserves to know which it is holding.
--
-- Applies to the purdueio database's `advisor` schema, alongside the degree
-- requirement tables (whose convention this follows: numbered migration + a
-- sync-runs table for provenance).
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS advisor;

CREATE TABLE IF NOT EXISTS advisor.offering_sync_runs (
    id             uuid PRIMARY KEY,
    source         text        NOT NULL,   -- 'purdueio_odata' | 'purdueio_local'
    status         text        NOT NULL,   -- 'running' | 'completed' | 'failed'
    started_at     timestamptz NOT NULL DEFAULT now(),
    completed_at   timestamptz,
    terms_synced   integer     NOT NULL DEFAULT 0,
    offerings_seen integer     NOT NULL DEFAULT 0,
    errors         jsonb       NOT NULL DEFAULT '[]'::jsonb,
    metadata       jsonb       NOT NULL DEFAULT '{}'::jsonb
);

-- One row per (course, term) that actually ran. This is raw observation; never
-- edit it by hand, because `course_offering_patterns` is derived from it and any
-- manual correction here would be silently overwritten by the next sync.
CREATE TABLE IF NOT EXISTS advisor.course_offerings (
    course_code   text    NOT NULL,        -- canonical 'CS 25100'
    subject       text    NOT NULL,        -- 'CS'
    number        text    NOT NULL,        -- '25100'
    term_code     text    NOT NULL,        -- Banner code, e.g. '202710'
    term_name     text,                    -- 'Fall 2026'
    season        text    NOT NULL,        -- 'fall' | 'spring' | 'summer' | 'winter'
    calendar_year integer NOT NULL,
    class_count   integer NOT NULL DEFAULT 0,
    sync_run_id   uuid REFERENCES advisor.offering_sync_runs(id) ON DELETE SET NULL,
    PRIMARY KEY (course_code, term_code)
);

CREATE INDEX IF NOT EXISTS ix_course_offerings_code   ON advisor.course_offerings (course_code);
CREATE INDEX IF NOT EXISTS ix_course_offerings_season ON advisor.course_offerings (season);
CREATE INDEX IF NOT EXISTS ix_course_offerings_term   ON advisor.course_offerings (term_code);

-- Derived, and rebuilt wholesale by the sync. `*_terms` are the observation
-- counts behind each boolean: `offered_fall = true, fall_terms = 1` means one
-- sighting, which is a hint; `fall_terms = 9` is a pattern. `terms_observed` is
-- the denominator — a course with 1 total observation says nothing about the
-- terms it was NOT seen in, and callers must not read absence as evidence.
CREATE TABLE IF NOT EXISTS advisor.course_offering_patterns (
    course_code     text PRIMARY KEY,
    subject         text    NOT NULL,
    number          text    NOT NULL,
    offered_fall    boolean NOT NULL DEFAULT false,
    offered_spring  boolean NOT NULL DEFAULT false,
    offered_summer  boolean NOT NULL DEFAULT false,
    offered_winter  boolean NOT NULL DEFAULT false,
    fall_terms      integer NOT NULL DEFAULT 0,
    spring_terms    integer NOT NULL DEFAULT 0,
    summer_terms    integer NOT NULL DEFAULT 0,
    winter_terms    integer NOT NULL DEFAULT 0,
    terms_observed  integer NOT NULL DEFAULT 0,
    first_term_code text,
    last_term_code  text,
    updated_at      timestamptz NOT NULL DEFAULT now()
);

-- Convenience view in the shape the planner wants: a text[] of schedulable terms.
-- Winter is excluded here on purpose — it is a ~38-class intersession, not a term
-- a degree plan is built from. Summer IS included in the array; whether the
-- planner may schedule into it is a separate decision (settings.planner_include_summer),
-- and conflating "the course is taught then" with "we will plan you into it"
-- would make the offerings data lie.
CREATE OR REPLACE VIEW advisor.course_planner_terms AS
SELECT
    course_code,
    subject,
    number,
    ARRAY_REMOVE(ARRAY[
        CASE WHEN offered_fall   THEN 'fall'   END,
        CASE WHEN offered_spring THEN 'spring' END,
        CASE WHEN offered_summer THEN 'summer' END
    ], NULL) AS offered_terms,
    terms_observed,
    last_term_code
FROM advisor.course_offering_patterns;

INSERT INTO advisor.schema_migrations (version, applied_at)
VALUES ('002_course_offerings', now())
ON CONFLICT (version) DO NOTHING;
