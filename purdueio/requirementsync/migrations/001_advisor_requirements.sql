CREATE SCHEMA IF NOT EXISTS advisor;

CREATE TABLE IF NOT EXISTS advisor.schema_migrations (
  version text PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS advisor.requirement_sync_runs (
  id uuid PRIMARY KEY,
  catalog_year integer,
  catalog_label text,
  catalog_base_url text NOT NULL,
  status text NOT NULL,
  started_at timestamptz NOT NULL DEFAULT now(),
  completed_at timestamptz,
  programs_found integer NOT NULL DEFAULT 0,
  programs_loaded integer NOT NULL DEFAULT 0,
  errors jsonb NOT NULL DEFAULT '[]'::jsonb,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS advisor.degree_programs (
  id uuid PRIMARY KEY,
  catalog_year integer NOT NULL,
  school text,
  program_title text NOT NULL,
  degree_code text,
  variant text,
  source_url text NOT NULL,
  source_hash text NOT NULL,
  parser_status text NOT NULL,
  raw_json jsonb NOT NULL,
  sync_run_id uuid NOT NULL REFERENCES advisor.requirement_sync_runs(id) ON DELETE RESTRICT,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_degree_programs_catalog_source
  ON advisor.degree_programs(catalog_year, source_url);

CREATE INDEX IF NOT EXISTS ix_degree_programs_catalog_year
  ON advisor.degree_programs(catalog_year);

CREATE INDEX IF NOT EXISTS ix_degree_programs_school
  ON advisor.degree_programs(school);

CREATE INDEX IF NOT EXISTS ix_degree_programs_title
  ON advisor.degree_programs(program_title);

CREATE TABLE IF NOT EXISTS advisor.degree_requirement_blocks (
  id uuid PRIMARY KEY,
  program_id uuid NOT NULL REFERENCES advisor.degree_programs(id) ON DELETE CASCADE,
  sort_order integer NOT NULL,
  title text,
  credits_text text,
  raw_json jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(program_id, sort_order)
);

CREATE INDEX IF NOT EXISTS ix_degree_requirement_blocks_program_id
  ON advisor.degree_requirement_blocks(program_id);

CREATE TABLE IF NOT EXISTS advisor.degree_requirement_rules (
  id uuid PRIMARY KEY,
  block_id uuid NOT NULL REFERENCES advisor.degree_requirement_blocks(id) ON DELETE CASCADE,
  sort_order integer NOT NULL,
  rule_type text NOT NULL,
  choose_count integer,
  raw_text text,
  raw_json jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(block_id, sort_order)
);

CREATE INDEX IF NOT EXISTS ix_degree_requirement_rules_block_id
  ON advisor.degree_requirement_rules(block_id);

CREATE TABLE IF NOT EXISTS advisor.degree_requirement_options (
  id uuid PRIMARY KEY,
  rule_id uuid NOT NULL REFERENCES advisor.degree_requirement_rules(id) ON DELETE CASCADE,
  option_index integer NOT NULL,
  sort_order integer NOT NULL,
  label text,
  raw_json jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(rule_id, option_index)
);

CREATE INDEX IF NOT EXISTS ix_degree_requirement_options_rule_id
  ON advisor.degree_requirement_options(rule_id);

CREATE TABLE IF NOT EXISTS advisor.degree_requirement_courses (
  id uuid PRIMARY KEY,
  option_id uuid NOT NULL REFERENCES advisor.degree_requirement_options(id) ON DELETE CASCADE,
  sort_order integer NOT NULL,
  course_code_text text NOT NULL,
  course_id uuid REFERENCES public."Courses"("Id") ON DELETE SET NULL,
  course_title text,
  credits_text text,
  raw_text text,
  raw_json jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(option_id, sort_order)
);

CREATE INDEX IF NOT EXISTS ix_degree_requirement_courses_option_id
  ON advisor.degree_requirement_courses(option_id);

CREATE INDEX IF NOT EXISTS ix_degree_requirement_courses_course_id
  ON advisor.degree_requirement_courses(course_id);

CREATE INDEX IF NOT EXISTS ix_degree_requirement_courses_code
  ON advisor.degree_requirement_courses(course_code_text);
