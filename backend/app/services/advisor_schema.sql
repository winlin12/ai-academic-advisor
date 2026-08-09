-- ============================================================================
-- advisor schema, inside the catalog Postgres.
--
-- WHY IT IS HERE AND NOT IN `purdueio`. services/offerings/schema.sql and
-- services/prerequisites/schema.sql create these same tables in the PurdueIO
-- database's `advisor` schema, which is where the SYNC jobs write them. That
-- database is a separate container that is not part of this app's compose file
-- (settings.purdueio_database_url addresses it by raw Docker IP), so the web
-- app cannot assume it is reachable — and the app needs prerequisites and term
-- offerings on every single plan it builds. A plan built without them is not a
-- degraded plan, it is a wrong one: `planner_catalog._course_from_row` used to
-- hard-code every course as offered every term with no prereq edges at all,
-- which is exactly the "will happily produce a schedule a student cannot
-- register for" failure 002_course_offerings was written about.
--
-- So the serving side keeps its own copy in the database it definitely has.
-- Column shapes are deliberately IDENTICAL to the two migrations above so the
-- sync jobs can target either database and so `plan_context` renders the same
-- rows the model_eval harness measured against (harness/mock_db.py TABLES).
--
-- `course_workload` has no upstream at all — it is an advisor-assigned 1-5
-- estimate used only for tie-breaking inside the deterministic planner, and it
-- is recorded here rather than hard-coded so it is inspectable and editable
-- like every other fact the planner reads.
--
-- Idempotent: safe to re-run, and run at API startup (main.lifespan).
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS advisor;

CREATE TABLE IF NOT EXISTS advisor.schema_migrations (
    version    text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

-- Mirrors advisor.course_prerequisites in services/prerequisites/schema.sql.
CREATE TABLE IF NOT EXISTS advisor.course_prerequisites (
    course_code   text PRIMARY KEY,        -- canonical 'CS 38100'
    subject       text NOT NULL,
    number        text NOT NULL,
    term_code     text NOT NULL,           -- the term the prereqs were read from
    has_prereqs   boolean NOT NULL,        -- false = no prerequisite block (a real fact)
    raw_text      text,                    -- cleaned expression, always kept
    parsed_tree   jsonb,                   -- AND/OR/COURSE tree, NULL if unparseable
    prereq_groups jsonb,                   -- flat AND-of-ORs, NULL if unflattenable
    coreq_codes   jsonb NOT NULL DEFAULT '[]'::jsonb,
    confidence    text NOT NULL,           -- 'high' | 'medium' | 'low' | 'none'
    notes         jsonb NOT NULL DEFAULT '[]'::jsonb,
    updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_course_prerequisites_subject
    ON advisor.course_prerequisites (subject);

-- Mirrors advisor.course_offering_patterns in services/offerings/schema.sql.
-- `*_terms` are the observation counts behind each boolean, and `terms_observed`
-- is the denominator: a course with 1 total observation says nothing about the
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

-- The shape the planner wants: a text[] of schedulable terms. Winter is excluded
-- (a ~38-class intersession, not a term a degree plan is built from). Summer IS
-- in the array; whether the planner may schedule into it is a separate decision
-- (settings.planner_include_summer), and conflating "taught then" with "we will
-- plan you into it" would make the offerings data lie.
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

-- Advisor-assigned, no upstream source. 1 (light) to 5 (heavy).
CREATE TABLE IF NOT EXISTS advisor.course_workload (
    course_code    text PRIMARY KEY,
    workload_score integer NOT NULL CHECK (workload_score BETWEEN 1 AND 5),
    updated_at     timestamptz NOT NULL DEFAULT now()
);

INSERT INTO advisor.schema_migrations (version, applied_at)
VALUES ('001_advisor_serving_tables', now())
ON CONFLICT (version) DO NOTHING;
