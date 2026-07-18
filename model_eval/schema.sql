-- ============================================================================
-- STUB SCHEMA — REPLACE / EXTEND BEFORE TRUSTING RESULTS
--
-- This is a SQLite adaptation of the REAL Postgres schema in
-- backend/app/data/db_schema.md (catalog_ingestion database), simplified:
--   * UUID PKs -> INTEGER PKs (SQLite convention; models find these easier and
--     the difference is not what we're measuring)
--   * scrape/audit tables (source_pages, scrape_runs, ...) dropped — no
--     advising question ever needs them
--   * jsonb -> TEXT
--
-- The SEED DATA at the bottom is FAKE-BUT-PLAUSIBLE. It exists so gold SQL
-- queries return rows. Edit it to match real Purdue data before you trust
-- SQL_CORRECT numbers on multi-hop questions.
--
-- This exact file's text is embedded verbatim in the static prompt block, so
-- editing it invalidates prior results (the harness records a hash of the
-- static block with every run — hashes won't match across edits, by design).
-- ============================================================================

CREATE TABLE catalog_years (
    id          INTEGER PRIMARY KEY,
    label       TEXT NOT NULL,        -- e.g. '2026-2027'
    start_year  INTEGER,
    end_year    INTEGER,
    is_archived INTEGER DEFAULT 0     -- boolean 0/1
);

CREATE TABLE subjects (
    id              INTEGER PRIMARY KEY,
    catalog_year_id INTEGER REFERENCES catalog_years(id),
    code            TEXT NOT NULL,    -- e.g. 'CS'
    name            TEXT              -- e.g. 'Computer Science'
);

CREATE TABLE courses (
    id                 INTEGER PRIMARY KEY,
    catalog_year_id    INTEGER REFERENCES catalog_years(id),
    subject_id         INTEGER REFERENCES subjects(id),
    subject_code       TEXT NOT NULL,     -- e.g. 'CS'
    course_number      TEXT NOT NULL,     -- e.g. '25100'
    course_code        TEXT NOT NULL,     -- e.g. 'CS 25100' (canonical: space-separated)
    title              TEXT,
    description        TEXT,
    credit_hours_min   REAL,
    credit_hours_max   REAL,
    prerequisites_raw  TEXT,              -- raw catalog prerequisite text, may be NULL
    corequisites_raw   TEXT,
    restrictions_raw   TEXT
);

CREATE TABLE course_aliases (
    id          INTEGER PRIMARY KEY,
    course_id   INTEGER REFERENCES courses(id),
    alias_code  TEXT NOT NULL,        -- cross-listing or old code, e.g. 'CS 251'
    reason      TEXT
);

CREATE TABLE prerequisite_rules (
    id               INTEGER PRIMARY KEY,
    course_id        INTEGER REFERENCES courses(id),
    raw_text         TEXT NOT NULL,   -- never null
    parsed_json      TEXT,            -- AND/OR tree as JSON; NULL if parse failed
    parse_confidence TEXT             -- 'high' | 'medium' | 'low'
);

CREATE TABLE programs (
    id                INTEGER PRIMARY KEY,
    catalog_year_id   INTEGER REFERENCES catalog_years(id),
    name              TEXT NOT NULL,   -- e.g. 'Computer Science'
    degree_type       TEXT,            -- 'BS', 'BA', 'MS', 'Minor', ...
    program_type      TEXT,            -- 'undergraduate_major', 'minor', ...
    campus            TEXT,
    total_credits_min REAL,
    description       TEXT
);

CREATE TABLE requirement_groups (
    id               INTEGER PRIMARY KEY,
    program_id       INTEGER REFERENCES programs(id),
    parent_group_id  INTEGER REFERENCES requirement_groups(id),  -- NULL = top level
    name             TEXT,             -- section heading, e.g. 'Computer Science Core'
    requirement_type TEXT,             -- 'core', 'elective', 'gen_ed', 'college', ...
    credits_min      REAL,
    credits_max      REAL,
    raw_text         TEXT
);

CREATE TABLE requirement_options (
    id                   INTEGER PRIMARY KEY,
    requirement_group_id INTEGER REFERENCES requirement_groups(id),
    course_id            INTEGER REFERENCES courses(id),  -- NULL if unresolved
    course_code_raw      TEXT,          -- as printed in the catalog
    option_text          TEXT,
    credits              REAL,
    minimum_grade        TEXT,
    is_required          INTEGER,       -- boolean 0/1; 0 = selective/elective
    is_selective_option  INTEGER        -- boolean 0/1; 1 = 'choose from this list'
);

CREATE TABLE program_notes (
    id         INTEGER PRIMARY KEY,
    program_id INTEGER REFERENCES programs(id),
    note_text  TEXT,
    note_type  TEXT                    -- 'gpa', 'residency', 'parser_warning', ...
);

-- ============================================================================
-- SEED DATA (FAKE — edit before trusting multi-hop SQL_CORRECT results)
-- ============================================================================

INSERT INTO catalog_years (id, label, start_year, end_year) VALUES
    (1, '2026-2027', 2026, 2027);

INSERT INTO subjects (id, catalog_year_id, code, name) VALUES
    (1, 1, 'CS',   'Computer Science'),
    (2, 1, 'MA',   'Mathematics'),
    (3, 1, 'STAT', 'Statistics'),
    (4, 1, 'PHYS', 'Physics');

INSERT INTO courses (id, catalog_year_id, subject_id, subject_code, course_number,
                     course_code, title, description, credit_hours_min,
                     credit_hours_max, prerequisites_raw) VALUES
    (1,  1, 1, 'CS', '18000', 'CS 18000', 'Problem Solving And Object-Oriented Programming',
     'Introduction to Java and object-oriented programming.', 4, 4, NULL),
    (2,  1, 1, 'CS', '18200', 'CS 18200', 'Foundations Of Computer Science',
     'Discrete math foundations: logic, sets, proofs, graphs.', 3, 3,
     'CS 18000 and (MA 16100 or MA 16500)'),
    (3,  1, 1, 'CS', '24000', 'CS 24000', 'Programming In C',
     'C programming, pointers, memory management.', 3, 3, 'CS 18000'),
    (4,  1, 1, 'CS', '25000', 'CS 25000', 'Computer Architecture',
     'Digital logic, assembly, processor organization.', 4, 4, 'CS 18200 and CS 24000'),
    (5,  1, 1, 'CS', '25100', 'CS 25100', 'Data Structures And Algorithms',
     'Lists, trees, hashing, graphs, algorithm analysis.', 3, 3, 'CS 18200 and CS 24000'),
    (6,  1, 1, 'CS', '25200', 'CS 25200', 'Systems Programming',
     'UNIX systems programming, processes, threads, sockets.', 4, 4,
     'CS 25000 and CS 25100'),
    (7,  1, 1, 'CS', '35400', 'CS 35400', 'Operating Systems',
     'Processes, scheduling, memory management, file systems, concurrency.', 3, 3,
     'CS 25200'),
    (8,  1, 1, 'CS', '38100', 'CS 38100', 'Introduction To The Analysis Of Algorithms',
     'Algorithm design techniques and complexity analysis.', 3, 3,
     'CS 25100 and (MA 26100 or MA 27101)'),
    (9,  1, 2, 'MA', '16100', 'MA 16100', 'Plane Analytic Geometry And Calculus I',
     'Differential calculus of one variable.', 5, 5, NULL),
    (10, 1, 2, 'MA', '16200', 'MA 16200', 'Plane Analytic Geometry And Calculus II',
     'Integral calculus, sequences and series.', 5, 5, 'MA 16100'),
    (11, 1, 2, 'MA', '26100', 'MA 26100', 'Multivariate Calculus',
     'Partial derivatives, multiple integrals, vector calculus.', 4, 4, 'MA 16200'),
    (12, 1, 3, 'STAT', '35000', 'STAT 35000', 'Introduction To Statistics',
     'Probability, estimation, hypothesis testing.', 3, 3, 'MA 16200'),
    (13, 1, 2, 'MA', '26500', 'MA 26500', 'Linear Algebra',
     'Vector spaces, matrices, eigenvalues, linear transformations.', 3, 3, 'MA 16200'),
    (14, 1, 4, 'PHYS', '17200', 'PHYS 17200', 'Modern Mechanics',
     'Mechanics for engineers: kinematics, dynamics, energy, momentum.', 4, 4, 'MA 16200');

INSERT INTO course_aliases (id, course_id, alias_code, reason) VALUES
    (1, 5, 'CS 251', 'legacy three-digit code'),
    (2, 7, 'CS 354', 'legacy three-digit code');

INSERT INTO prerequisite_rules (id, course_id, raw_text, parsed_json, parse_confidence) VALUES
    (1, 2, 'CS 18000 and (MA 16100 or MA 16500)',
     '{"and": ["CS 18000", {"or": ["MA 16100", "MA 16500"]}]}', 'high'),
    (2, 5, 'CS 18200 and CS 24000', '{"and": ["CS 18200", "CS 24000"]}', 'high'),
    (3, 7, 'CS 25200', '{"course": "CS 25200"}', 'high');

INSERT INTO programs (id, catalog_year_id, name, degree_type, program_type, campus,
                      total_credits_min, description) VALUES
    (1, 1, 'Computer Science', 'BS', 'undergraduate_major', 'West Lafayette', 120,
     'Bachelor of Science in Computer Science.'),
    (2, 1, 'Data Science', 'BS', 'undergraduate_major', 'West Lafayette', 120,
     'Bachelor of Science in Data Science.');

INSERT INTO requirement_groups (id, program_id, parent_group_id, name, requirement_type,
                                credits_min, raw_text) VALUES
    (1, 1, NULL, 'Computer Science Core', 'core', 24,
     'All of the following core courses are required.'),
    (2, 1, NULL, 'Mathematics Requirements', 'major', 14,
     'Required mathematics sequence.'),
    (3, 2, NULL, 'Data Science Core', 'core', 15,
     'All of the following core courses are required.');

INSERT INTO requirement_options (id, requirement_group_id, course_id, course_code_raw,
                                 option_text, credits, is_required,
                                 is_selective_option) VALUES
    (1,  1, 1,  'CS 18000', 'CS 18000 - Problem Solving And OOP (4 cr)',        4, 1, 0),
    (2,  1, 2,  'CS 18200', 'CS 18200 - Foundations Of Computer Science (3 cr)', 3, 1, 0),
    (3,  1, 3,  'CS 24000', 'CS 24000 - Programming In C (3 cr)',                3, 1, 0),
    (4,  1, 4,  'CS 25000', 'CS 25000 - Computer Architecture (4 cr)',           4, 1, 0),
    (5,  1, 5,  'CS 25100', 'CS 25100 - Data Structures And Algorithms (3 cr)',  3, 1, 0),
    (6,  1, 6,  'CS 25200', 'CS 25200 - Systems Programming (4 cr)',             4, 1, 0),
    (7,  1, 8,  'CS 38100', 'CS 38100 - Analysis Of Algorithms (3 cr)',          3, 1, 0),
    (8,  2, 9,  'MA 16100', 'MA 16100 - Calculus I (5 cr)',                      5, 1, 0),
    (9,  2, 10, 'MA 16200', 'MA 16200 - Calculus II (5 cr)',                     5, 1, 0),
    (10, 2, 11, 'MA 26100', 'MA 26100 - Multivariate Calculus (4 cr)',           4, 1, 0),
    (11, 3, 5,  'CS 25100', 'CS 25100 - Data Structures And Algorithms (3 cr)',  3, 1, 0),
    (12, 3, 12, 'STAT 35000', 'STAT 35000 - Introduction To Statistics (3 cr)',  3, 1, 0);

INSERT INTO program_notes (id, program_id, note_text, note_type) VALUES
    (1, 1, 'A minimum GPA of 2.0 in CS core courses is required for graduation.', 'gpa'),
    (2, 2, 'A minimum GPA of 2.0 in Data Science core courses is required for graduation.', 'gpa');
