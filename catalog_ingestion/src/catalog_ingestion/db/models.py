"""SQLAlchemy 2.x ORM models for the catalog ingestion schema."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# catalog_years
# ---------------------------------------------------------------------------

class CatalogYear(Base):
    __tablename__ = "catalog_years"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    label: Mapped[str] = mapped_column(String(20), nullable=False)
    catoid: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    start_year: Mapped[int] = mapped_column(Integer, nullable=False)
    end_year: Mapped[int] = mapped_column(Integer, nullable=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    catalog_url: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_now, onupdate=_now
    )

    source_pages: Mapped[list[SourcePage]] = relationship(back_populates="catalog_year")
    colleges: Mapped[list[College]] = relationship(back_populates="catalog_year")
    departments: Mapped[list[Department]] = relationship(back_populates="catalog_year")
    subjects: Mapped[list[Subject]] = relationship(back_populates="catalog_year")
    courses: Mapped[list[Course]] = relationship(back_populates="catalog_year")
    programs: Mapped[list[Program]] = relationship(back_populates="catalog_year")
    scrape_runs: Mapped[list[ScrapeRun]] = relationship(back_populates="catalog_year")

    __table_args__ = (
        Index("ix_catalog_years_start_year", "start_year"),
    )


# ---------------------------------------------------------------------------
# source_pages
# ---------------------------------------------------------------------------

class SourcePage(Base):
    __tablename__ = "source_pages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    catalog_year_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("catalog_years.id", ondelete="SET NULL"), nullable=True
    )
    url: Mapped[str] = mapped_column(Text, nullable=False)
    page_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    raw_html: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    parser_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    catalog_year: Mapped[CatalogYear | None] = relationship(back_populates="source_pages")

    __table_args__ = (
        UniqueConstraint("url", "content_hash", name="uq_source_pages_url_hash"),
        Index("ix_source_pages_url", "url"),
        Index("ix_source_pages_catalog_year_id", "catalog_year_id"),
        Index("ix_source_pages_page_type", "page_type"),
    )


# ---------------------------------------------------------------------------
# scrape_runs
# ---------------------------------------------------------------------------

class ScrapeRun(Base):
    __tablename__ = "scrape_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    catalog_year_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("catalog_years.id", ondelete="SET NULL"), nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    target_catalog_years: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    pages_attempted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pages_succeeded: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pages_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    parser_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    catalog_year: Mapped[CatalogYear | None] = relationship(back_populates="scrape_runs")
    errors: Mapped[list[ScrapeError]] = relationship(back_populates="scrape_run")

    __table_args__ = (
        Index("ix_scrape_runs_catalog_year_id", "catalog_year_id"),
        Index("ix_scrape_runs_status", "status"),
    )


# ---------------------------------------------------------------------------
# scrape_errors
# ---------------------------------------------------------------------------

class ScrapeError(Base):
    __tablename__ = "scrape_errors"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    scrape_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("scrape_runs.id", ondelete="CASCADE"), nullable=False
    )
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)

    scrape_run: Mapped[ScrapeRun] = relationship(back_populates="errors")

    __table_args__ = (
        Index("ix_scrape_errors_scrape_run_id", "scrape_run_id"),
    )


# ---------------------------------------------------------------------------
# colleges
# ---------------------------------------------------------------------------

class College(Base):
    __tablename__ = "colleges"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    catalog_year_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("catalog_years.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    source_page_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_pages.id", ondelete="SET NULL"), nullable=True
    )

    catalog_year: Mapped[CatalogYear] = relationship(back_populates="colleges")
    departments: Mapped[list[Department]] = relationship(back_populates="college")
    programs: Mapped[list[Program]] = relationship(back_populates="college")

    __table_args__ = (
        UniqueConstraint("catalog_year_id", "name", name="uq_colleges_year_name"),
        Index("ix_colleges_catalog_year_id", "catalog_year_id"),
    )


# ---------------------------------------------------------------------------
# departments
# ---------------------------------------------------------------------------

class Department(Base):
    __tablename__ = "departments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    catalog_year_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("catalog_years.id", ondelete="CASCADE"), nullable=False
    )
    college_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("colleges.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    source_page_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_pages.id", ondelete="SET NULL"), nullable=True
    )

    catalog_year: Mapped[CatalogYear] = relationship(back_populates="departments")
    college: Mapped[College | None] = relationship(back_populates="departments")
    programs: Mapped[list[Program]] = relationship(back_populates="department")

    __table_args__ = (
        UniqueConstraint("catalog_year_id", "name", name="uq_departments_year_name"),
        Index("ix_departments_catalog_year_id", "catalog_year_id"),
        Index("ix_departments_college_id", "college_id"),
    )


# ---------------------------------------------------------------------------
# subjects
# ---------------------------------------------------------------------------

class Subject(Base):
    __tablename__ = "subjects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    catalog_year_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("catalog_years.id", ondelete="CASCADE"), nullable=False
    )
    code: Mapped[str] = mapped_column(String(16), nullable=False)
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_page_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_pages.id", ondelete="SET NULL"), nullable=True
    )

    catalog_year: Mapped[CatalogYear] = relationship(back_populates="subjects")
    courses: Mapped[list[Course]] = relationship(back_populates="subject")

    __table_args__ = (
        UniqueConstraint("catalog_year_id", "code", name="uq_subjects_year_code"),
        Index("ix_subjects_catalog_year_id", "catalog_year_id"),
        Index("ix_subjects_code", "code"),
    )


# ---------------------------------------------------------------------------
# courses
# ---------------------------------------------------------------------------

class Course(Base):
    __tablename__ = "courses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    catalog_year_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("catalog_years.id", ondelete="CASCADE"), nullable=False
    )
    subject_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("subjects.id", ondelete="SET NULL"), nullable=True
    )
    subject_code: Mapped[str] = mapped_column(String(16), nullable=False)
    course_number: Mapped[str] = mapped_column(String(16), nullable=False)
    course_code: Mapped[str] = mapped_column(String(32), nullable=False)
    coid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    credit_hours_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    credit_hours_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    credit_hours_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    prerequisites_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    corequisites_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    restrictions_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    attributes_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_page_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_pages.id", ondelete="SET NULL"), nullable=True
    )

    catalog_year: Mapped[CatalogYear] = relationship(back_populates="courses")
    subject: Mapped[Subject | None] = relationship(back_populates="courses")
    aliases: Mapped[list[CourseAlias]] = relationship(back_populates="course")
    prerequisite_rules: Mapped[list[PrerequisiteRule]] = relationship(back_populates="course")

    __table_args__ = (
        UniqueConstraint(
            "catalog_year_id", "course_code", name="uq_courses_year_code"
        ),
        Index("ix_courses_catalog_year_id", "catalog_year_id"),
        Index("ix_courses_subject_code", "subject_code"),
        Index("ix_courses_course_code", "course_code"),
        Index("ix_courses_coid", "coid"),
    )


# ---------------------------------------------------------------------------
# course_aliases
# ---------------------------------------------------------------------------

class CourseAlias(Base):
    __tablename__ = "course_aliases"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    course_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("courses.id", ondelete="CASCADE"), nullable=False
    )
    alias_code: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    course: Mapped[Course] = relationship(back_populates="aliases")

    __table_args__ = (
        UniqueConstraint("course_id", "alias_code", name="uq_course_aliases"),
        Index("ix_course_aliases_course_id", "course_id"),
        Index("ix_course_aliases_alias_code", "alias_code"),
    )


# ---------------------------------------------------------------------------
# prerequisite_rules
# ---------------------------------------------------------------------------

class PrerequisiteRule(Base):
    __tablename__ = "prerequisite_rules"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    course_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("courses.id", ondelete="CASCADE"), nullable=False
    )
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    parsed_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    parse_confidence: Mapped[str | None] = mapped_column(
        String(16), nullable=True
    )
    parser_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    course: Mapped[Course] = relationship(back_populates="prerequisite_rules")

    __table_args__ = (
        Index("ix_prerequisite_rules_course_id", "course_id"),
    )


# ---------------------------------------------------------------------------
# programs
# ---------------------------------------------------------------------------

class Program(Base):
    __tablename__ = "programs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    catalog_year_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("catalog_years.id", ondelete="CASCADE"), nullable=False
    )
    college_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("colleges.id", ondelete="SET NULL"), nullable=True
    )
    department_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("departments.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    degree_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    program_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    campus: Mapped[str | None] = mapped_column(String(128), nullable=True)
    total_credits_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_credits_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    poid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_page_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_pages.id", ondelete="SET NULL"), nullable=True
    )

    catalog_year: Mapped[CatalogYear] = relationship(back_populates="programs")
    college: Mapped[College | None] = relationship(back_populates="programs")
    department: Mapped[Department | None] = relationship(back_populates="programs")
    requirement_groups: Mapped[list[RequirementGroup]] = relationship(back_populates="program")
    notes: Mapped[list[ProgramNote]] = relationship(back_populates="program")

    __table_args__ = (
        UniqueConstraint(
            "catalog_year_id", "name", "degree_type", name="uq_programs_year_name_degree"
        ),
        Index("ix_programs_catalog_year_id", "catalog_year_id"),
        Index("ix_programs_degree_type", "degree_type"),
        Index("ix_programs_poid", "poid"),
    )


# ---------------------------------------------------------------------------
# requirement_groups
# ---------------------------------------------------------------------------

class RequirementGroup(Base):
    __tablename__ = "requirement_groups"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    program_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("programs.id", ondelete="CASCADE"), nullable=False
    )
    parent_group_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("requirement_groups.id", ondelete="CASCADE"),
        nullable=True,
    )
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    requirement_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    credits_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    credits_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    display_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_page_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_pages.id", ondelete="SET NULL"), nullable=True
    )

    program: Mapped[Program] = relationship(back_populates="requirement_groups")
    parent: Mapped[RequirementGroup | None] = relationship(
        back_populates="children", remote_side="RequirementGroup.id"
    )
    children: Mapped[list[RequirementGroup]] = relationship(back_populates="parent")
    options: Mapped[list[RequirementOption]] = relationship(back_populates="group")

    __table_args__ = (
        Index("ix_requirement_groups_program_id", "program_id"),
        Index("ix_requirement_groups_parent_group_id", "parent_group_id"),
    )


# ---------------------------------------------------------------------------
# requirement_options
# ---------------------------------------------------------------------------

class RequirementOption(Base):
    __tablename__ = "requirement_options"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    requirement_group_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("requirement_groups.id", ondelete="CASCADE"),
        nullable=False,
    )
    course_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("courses.id", ondelete="SET NULL"), nullable=True
    )
    coid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    course_code_raw: Mapped[str | None] = mapped_column(String(32), nullable=True)
    course_title_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    option_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    credits: Mapped[float | None] = mapped_column(Float, nullable=True)
    credits_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    minimum_grade: Mapped[str | None] = mapped_column(String(8), nullable=True)
    is_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_selective_option: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    display_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    group: Mapped[RequirementGroup] = relationship(back_populates="options")
    course: Mapped[Course | None] = relationship()

    __table_args__ = (
        Index("ix_requirement_options_group_id", "requirement_group_id"),
        Index("ix_requirement_options_course_id", "course_id"),
        Index("ix_requirement_options_coid", "coid"),
        Index("ix_requirement_options_course_code_raw", "course_code_raw"),
    )


# ---------------------------------------------------------------------------
# program_notes
# ---------------------------------------------------------------------------

class ProgramNote(Base):
    __tablename__ = "program_notes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    program_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("programs.id", ondelete="CASCADE"), nullable=False
    )
    note_text: Mapped[str] = mapped_column(Text, nullable=False)
    note_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_page_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_pages.id", ondelete="SET NULL"), nullable=True
    )

    program: Mapped[Program] = relationship(back_populates="notes")

    __table_args__ = (
        Index("ix_program_notes_program_id", "program_id"),
    )


# ---------------------------------------------------------------------------
# purdueapi staging tables  (Phase 5)
# ---------------------------------------------------------------------------

class PurdueapiCoursesStaging(Base):
    __tablename__ = "purdueapi_courses_staging"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    purdueapi_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subject_abbreviation: Mapped[str | None] = mapped_column(String(16), nullable=True)
    number: Mapped[str | None] = mapped_column(String(16), nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    credit_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    imported_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)

    __table_args__ = (
        Index("ix_purdueapi_courses_staging_subject", "subject_abbreviation"),
        Index("ix_purdueapi_courses_staging_number", "number"),
    )


class PurdueapiSubjectsStaging(Base):
    __tablename__ = "purdueapi_subjects_staging"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    purdueapi_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    abbreviation: Mapped[str | None] = mapped_column(String(16), nullable=True)
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    imported_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)

    __table_args__ = (
        Index("ix_purdueapi_subjects_staging_abbreviation", "abbreviation"),
    )


class PurdueapiTermsStaging(Base):
    __tablename__ = "purdueapi_terms_staging"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    purdueapi_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    code: Mapped[str | None] = mapped_column(String(16), nullable=True)
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    start_date: Mapped[str | None] = mapped_column(String(32), nullable=True)
    end_date: Mapped[str | None] = mapped_column(String(32), nullable=True)
    raw_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    imported_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)


# ---------------------------------------------------------------------------
# students / plans (app-side persistence — written by the backend, not ingestion)
# ---------------------------------------------------------------------------

class Student(Base):
    __tablename__ = "students"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    # The full StudentProfile as the backend's schema serialises it (program_id, completed
    # courses, preferences, ...). JSONB rather than columns: the profile shape belongs to the
    # backend and evolves with it; the ingestion schema only provides durable storage.
    profile: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_now, onupdate=_now
    )

    plans: Mapped[list[StudentPlan]] = relationship(
        back_populates="student", cascade="all, delete-orphan"
    )


class StudentPlan(Base):
    __tablename__ = "plans"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("students.id", ondelete="CASCADE"), nullable=False
    )
    # Free-text feedback that produced this plan (revise-plan flow); NULL for a plain generate.
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    plan: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)

    student: Mapped[Student] = relationship(back_populates="plans")

    __table_args__ = (
        Index("ix_plans_student_id", "student_id"),
    )
