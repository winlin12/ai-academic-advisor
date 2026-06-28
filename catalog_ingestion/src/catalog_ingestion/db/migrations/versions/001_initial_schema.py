"""Initial catalog ingestion schema.

Revision ID: 001
Revises:
Create Date: 2026-06-26
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "catalog_years",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("label", sa.String(20), nullable=False),
        sa.Column("catoid", sa.Integer, nullable=False, unique=True),
        sa.Column("start_year", sa.Integer, nullable=False),
        sa.Column("end_year", sa.Integer, nullable=False),
        sa.Column("is_archived", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("catalog_url", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_catalog_years_start_year", "catalog_years", ["start_year"])

    op.create_table(
        "source_pages",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("catalog_year_id", UUID(as_uuid=True), sa.ForeignKey("catalog_years.id", ondelete="SET NULL"), nullable=True),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("page_type", sa.String(64), nullable=True),
        sa.Column("http_status", sa.Integer, nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=True),
        sa.Column("fetched_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("raw_html", sa.Text, nullable=True),
        sa.Column("raw_text", sa.Text, nullable=True),
        sa.Column("parser_version", sa.String(32), nullable=True),
        sa.UniqueConstraint("url", "content_hash", name="uq_source_pages_url_hash"),
    )
    op.create_index("ix_source_pages_url", "source_pages", ["url"])
    op.create_index("ix_source_pages_catalog_year_id", "source_pages", ["catalog_year_id"])
    op.create_index("ix_source_pages_page_type", "source_pages", ["page_type"])

    op.create_table(
        "scrape_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("catalog_year_id", UUID(as_uuid=True), sa.ForeignKey("catalog_years.id", ondelete="SET NULL"), nullable=True),
        sa.Column("started_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime, nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="running"),
        sa.Column("target_catalog_years", JSONB, nullable=True),
        sa.Column("pages_attempted", sa.Integer, nullable=False, server_default="0"),
        sa.Column("pages_succeeded", sa.Integer, nullable=False, server_default="0"),
        sa.Column("pages_failed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("parser_version", sa.String(32), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
    )
    op.create_index("ix_scrape_runs_catalog_year_id", "scrape_runs", ["catalog_year_id"])
    op.create_index("ix_scrape_runs_status", "scrape_runs", ["status"])

    op.create_table(
        "scrape_errors",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("scrape_run_id", UUID(as_uuid=True), sa.ForeignKey("scrape_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("url", sa.Text, nullable=True),
        sa.Column("error_type", sa.String(128), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_scrape_errors_scrape_run_id", "scrape_errors", ["scrape_run_id"])

    op.create_table(
        "colleges",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("catalog_year_id", UUID(as_uuid=True), sa.ForeignKey("catalog_years.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("source_page_id", UUID(as_uuid=True), sa.ForeignKey("source_pages.id", ondelete="SET NULL"), nullable=True),
        sa.UniqueConstraint("catalog_year_id", "name", name="uq_colleges_year_name"),
    )
    op.create_index("ix_colleges_catalog_year_id", "colleges", ["catalog_year_id"])

    op.create_table(
        "departments",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("catalog_year_id", UUID(as_uuid=True), sa.ForeignKey("catalog_years.id", ondelete="CASCADE"), nullable=False),
        sa.Column("college_id", UUID(as_uuid=True), sa.ForeignKey("colleges.id", ondelete="SET NULL"), nullable=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("source_page_id", UUID(as_uuid=True), sa.ForeignKey("source_pages.id", ondelete="SET NULL"), nullable=True),
        sa.UniqueConstraint("catalog_year_id", "name", name="uq_departments_year_name"),
    )
    op.create_index("ix_departments_catalog_year_id", "departments", ["catalog_year_id"])
    op.create_index("ix_departments_college_id", "departments", ["college_id"])

    op.create_table(
        "subjects",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("catalog_year_id", UUID(as_uuid=True), sa.ForeignKey("catalog_years.id", ondelete="CASCADE"), nullable=False),
        sa.Column("code", sa.String(16), nullable=False),
        sa.Column("name", sa.Text, nullable=True),
        sa.Column("source_page_id", UUID(as_uuid=True), sa.ForeignKey("source_pages.id", ondelete="SET NULL"), nullable=True),
        sa.UniqueConstraint("catalog_year_id", "code", name="uq_subjects_year_code"),
    )
    op.create_index("ix_subjects_catalog_year_id", "subjects", ["catalog_year_id"])
    op.create_index("ix_subjects_code", "subjects", ["code"])

    op.create_table(
        "courses",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("catalog_year_id", UUID(as_uuid=True), sa.ForeignKey("catalog_years.id", ondelete="CASCADE"), nullable=False),
        sa.Column("subject_id", UUID(as_uuid=True), sa.ForeignKey("subjects.id", ondelete="SET NULL"), nullable=True),
        sa.Column("subject_code", sa.String(16), nullable=False),
        sa.Column("course_number", sa.String(16), nullable=False),
        sa.Column("course_code", sa.String(32), nullable=False),
        sa.Column("coid", sa.Integer, nullable=True),
        sa.Column("title", sa.Text, nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("credit_hours_min", sa.Float, nullable=True),
        sa.Column("credit_hours_max", sa.Float, nullable=True),
        sa.Column("credit_hours_raw", sa.Text, nullable=True),
        sa.Column("prerequisites_raw", sa.Text, nullable=True),
        sa.Column("corequisites_raw", sa.Text, nullable=True),
        sa.Column("restrictions_raw", sa.Text, nullable=True),
        sa.Column("attributes_raw", sa.Text, nullable=True),
        sa.Column("source_page_id", UUID(as_uuid=True), sa.ForeignKey("source_pages.id", ondelete="SET NULL"), nullable=True),
        sa.UniqueConstraint("catalog_year_id", "course_code", name="uq_courses_year_code"),
    )
    op.create_index("ix_courses_catalog_year_id", "courses", ["catalog_year_id"])
    op.create_index("ix_courses_subject_code", "courses", ["subject_code"])
    op.create_index("ix_courses_course_code", "courses", ["course_code"])
    op.create_index("ix_courses_coid", "courses", ["coid"])

    op.create_table(
        "course_aliases",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("course_id", UUID(as_uuid=True), sa.ForeignKey("courses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("alias_code", sa.String(32), nullable=False),
        sa.Column("reason", sa.Text, nullable=True),
        sa.UniqueConstraint("course_id", "alias_code", name="uq_course_aliases"),
    )
    op.create_index("ix_course_aliases_course_id", "course_aliases", ["course_id"])
    op.create_index("ix_course_aliases_alias_code", "course_aliases", ["alias_code"])

    op.create_table(
        "prerequisite_rules",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("course_id", UUID(as_uuid=True), sa.ForeignKey("courses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("raw_text", sa.Text, nullable=False),
        sa.Column("parsed_json", JSONB, nullable=True),
        sa.Column("parse_confidence", sa.String(16), nullable=True),
        sa.Column("parser_notes", sa.Text, nullable=True),
    )
    op.create_index("ix_prerequisite_rules_course_id", "prerequisite_rules", ["course_id"])

    op.create_table(
        "programs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("catalog_year_id", UUID(as_uuid=True), sa.ForeignKey("catalog_years.id", ondelete="CASCADE"), nullable=False),
        sa.Column("college_id", UUID(as_uuid=True), sa.ForeignKey("colleges.id", ondelete="SET NULL"), nullable=True),
        sa.Column("department_id", UUID(as_uuid=True), sa.ForeignKey("departments.id", ondelete="SET NULL"), nullable=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("degree_type", sa.String(32), nullable=True),
        sa.Column("program_type", sa.String(64), nullable=True),
        sa.Column("campus", sa.String(128), nullable=True),
        sa.Column("total_credits_raw", sa.Text, nullable=True),
        sa.Column("total_credits_min", sa.Float, nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("poid", sa.Integer, nullable=True),
        sa.Column("source_page_id", UUID(as_uuid=True), sa.ForeignKey("source_pages.id", ondelete="SET NULL"), nullable=True),
        sa.UniqueConstraint("catalog_year_id", "name", "degree_type", name="uq_programs_year_name_degree"),
    )
    op.create_index("ix_programs_catalog_year_id", "programs", ["catalog_year_id"])
    op.create_index("ix_programs_degree_type", "programs", ["degree_type"])
    op.create_index("ix_programs_poid", "programs", ["poid"])

    op.create_table(
        "requirement_groups",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("program_id", UUID(as_uuid=True), sa.ForeignKey("programs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("parent_group_id", UUID(as_uuid=True), sa.ForeignKey("requirement_groups.id", ondelete="CASCADE"), nullable=True),
        sa.Column("name", sa.Text, nullable=True),
        sa.Column("requirement_type", sa.String(64), nullable=True),
        sa.Column("credits_min", sa.Float, nullable=True),
        sa.Column("credits_max", sa.Float, nullable=True),
        sa.Column("raw_text", sa.Text, nullable=True),
        sa.Column("display_order", sa.Integer, nullable=False, server_default="0"),
        sa.Column("source_page_id", UUID(as_uuid=True), sa.ForeignKey("source_pages.id", ondelete="SET NULL"), nullable=True),
    )
    op.create_index("ix_requirement_groups_program_id", "requirement_groups", ["program_id"])
    op.create_index("ix_requirement_groups_parent_group_id", "requirement_groups", ["parent_group_id"])

    op.create_table(
        "requirement_options",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("requirement_group_id", UUID(as_uuid=True), sa.ForeignKey("requirement_groups.id", ondelete="CASCADE"), nullable=False),
        sa.Column("course_id", UUID(as_uuid=True), sa.ForeignKey("courses.id", ondelete="SET NULL"), nullable=True),
        sa.Column("course_code_raw", sa.String(32), nullable=True),
        sa.Column("option_text", sa.Text, nullable=True),
        sa.Column("credits", sa.Float, nullable=True),
        sa.Column("minimum_grade", sa.String(8), nullable=True),
        sa.Column("is_required", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("is_selective_option", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("display_order", sa.Integer, nullable=False, server_default="0"),
    )
    op.create_index("ix_requirement_options_group_id", "requirement_options", ["requirement_group_id"])
    op.create_index("ix_requirement_options_course_id", "requirement_options", ["course_id"])
    op.create_index("ix_requirement_options_course_code_raw", "requirement_options", ["course_code_raw"])

    op.create_table(
        "program_notes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("program_id", UUID(as_uuid=True), sa.ForeignKey("programs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("note_text", sa.Text, nullable=False),
        sa.Column("note_type", sa.String(64), nullable=True),
        sa.Column("source_page_id", UUID(as_uuid=True), sa.ForeignKey("source_pages.id", ondelete="SET NULL"), nullable=True),
    )
    op.create_index("ix_program_notes_program_id", "program_notes", ["program_id"])

    op.create_table(
        "purdueapi_courses_staging",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("purdueapi_id", sa.String(64), nullable=True),
        sa.Column("subject_abbreviation", sa.String(16), nullable=True),
        sa.Column("number", sa.String(16), nullable=True),
        sa.Column("title", sa.Text, nullable=True),
        sa.Column("credit_hours", sa.Float, nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("raw_json", JSONB, nullable=True),
        sa.Column("imported_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_purdueapi_courses_staging_subject", "purdueapi_courses_staging", ["subject_abbreviation"])
    op.create_index("ix_purdueapi_courses_staging_number", "purdueapi_courses_staging", ["number"])

    op.create_table(
        "purdueapi_subjects_staging",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("purdueapi_id", sa.String(64), nullable=True),
        sa.Column("abbreviation", sa.String(16), nullable=True),
        sa.Column("name", sa.Text, nullable=True),
        sa.Column("raw_json", JSONB, nullable=True),
        sa.Column("imported_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_purdueapi_subjects_staging_abbreviation", "purdueapi_subjects_staging", ["abbreviation"])

    op.create_table(
        "purdueapi_terms_staging",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("purdueapi_id", sa.String(64), nullable=True),
        sa.Column("code", sa.String(16), nullable=True),
        sa.Column("name", sa.Text, nullable=True),
        sa.Column("start_date", sa.String(32), nullable=True),
        sa.Column("end_date", sa.String(32), nullable=True),
        sa.Column("raw_json", JSONB, nullable=True),
        sa.Column("imported_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    for table in [
        "purdueapi_terms_staging",
        "purdueapi_subjects_staging",
        "purdueapi_courses_staging",
        "program_notes",
        "requirement_options",
        "requirement_groups",
        "programs",
        "prerequisite_rules",
        "course_aliases",
        "courses",
        "subjects",
        "departments",
        "colleges",
        "scrape_errors",
        "scrape_runs",
        "source_pages",
        "catalog_years",
    ]:
        op.drop_table(table)
