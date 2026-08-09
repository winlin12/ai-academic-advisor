"""One program's whole catalog, read from Postgres into the shapes the planner works in.

This is the layer `planner_catalog` was missing. That module built planner ``Course`` objects
straight from the ``courses`` table and had to hard-code ``prereqs=[]`` and ``offered_terms=
list(TERM_ORDER)``, because Acalog publishes neither — so every plan the app produced believed
that nothing had prerequisites and everything ran every term. Both facts now live in the
``advisor`` schema (see advisor_schema.sql), and this module is where they get joined back
onto the course rows before anything downstream sees them.

It serves three callers with one query set, which is the point — they must not disagree about
what the catalog says:

    planner_catalog.resolve_profile_and_catalog  the deterministic planner's course list
    plan_context.render_program_context          Mode B's database export (the model's context)
    plan_validation.validate_semesters           the checker that scores what Mode B returned

``ProgramCatalog`` is deliberately close to ``model_eval/harness``'s ``Fixture``: same notion
of requirement groups (all-of vs choose-credits), same alias collapsing, same chain-ordered
course list. The harness measured Gemma against that shape; serving it a different one would
make the eval's numbers describe a system we do not run.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import psycopg
from psycopg.rows import dict_row

from app.core.config import settings
from app.models.schemas import Course, StudentProfile

logger = logging.getLogger(__name__)

# requirement_group.requirement_type values that mean "choose from this list" outright. The
# crawled catalog says `elective`; the eval export says `choose_credits`; `academic_db`'s older
# block builder said `selective`. All three resolve the same way so a program ingested under
# any convention plans identically.
SELECTIVE_TYPES = {"choose_credits", "choose", "selective", "elective", "free_elective"}

# `plan_of_study` groups are the department's PUBLISHED SAMPLE 4-YEAR PLAN — "Fall 1st Year",
# "Spring 2nd Year", and so on. They are a schedule, not a requirement: every course in them
# already appears in a real requirement group, so counting them would double every course in
# the degree and hand the planner a remaining-list twice the size it should be. Excluded from
# requirements on purpose; they remain in the database for anything that wants the reference
# layout.
NON_REQUIREMENT_TYPES = {"plan_of_study"}

# A selective group with no credit target and options carrying no credits degrades to
# "choose one", using this as the assumed course size.
DEFAULT_OPTION_CREDITS = 3.0

ALL_SEASONS = ("fall", "spring", "summer")


class ProgramNotFoundError(LookupError):
    """The requested ``program_id`` is not in the academic database."""


def normalize_course_code(code: str) -> str:
    """Best-effort canonicalisation to the DB's ``'CS 18000'`` form.

    Accepts ``'cs18000'``, ``'CS  18000'``, ``'CS-18000'``; leaves codes with no digits
    untouched. Purely lexical — it never checks the code exists.
    """
    compact = "".join(str(code).split()).upper().replace("-", "")
    for index, char in enumerate(compact):
        if char.isdigit():
            return compact if index == 0 else f"{compact[:index]} {compact[index:]}"
    return compact


@dataclass
class RequirementGroup:
    """One degree requirement, flattened to the two kinds a planner can act on.

    ``options`` is a list of INTERCHANGEABLE SETS, not of courses. Purdue writes "MA 26100 -
    Multivariate Calculus ... or / MA 27101 - Honors Multivariate Calculus" as two rows joined
    by a trailing "or", which the crawler records in ``is_selective_option``; treating those as
    two separate courses puts a student in both the regular and the honors version of the same
    class. Almost every run has length 1, and the ones that don't are exactly the alternatives.
    """

    id: str
    name: str
    selective: bool
    credits_min: float | None
    options: list[list[str]] = field(default_factory=list)
    raw_text: str | None = None
    # Set only by `narrow_language_menu` for a language-sequence group once the student has
    # picked a language: the course at (or nearest below) their stated proficiency, so the
    # UI/model can point at a starting course instead of the student re-deriving one from a
    # bare list of three levels.
    recommended_code: str | None = None

    @property
    def courses(self) -> list[str]:
        """Every course the group names, alternatives included — for display and coverage."""
        return [code for run in self.options for code in run]

    def describe(self) -> str:
        kind = (f"choose {self.credits_min:g} credits" if self.selective and self.credits_min
                else "choose one" if self.selective else "take all")
        return f"{self.name} ({kind})"


# Group types that are PROSE, not requirements: page furniture, policies and advice. They carry
# no credits and nothing to satisfy, so they are neither planned nor reported.
PROSE_TYPES = {
    "overview", "disclaimer", "gpa", "policy", "learning_outcomes", "prerequisite_info",
    "critical_course", "licensure", "plan_of_study",
}

# Types that ARE requirements even when the catalog states no credit minimum for them —
# "University Core Requirements", "Civics Literacy Proficiency Requirement", "Upper Level
# Requirement". Leaving these out would drop the university-wide half of every degree.
#
# These are also where a requirement's SCOPE lives. The crawler already tells us "University
# Core Requirements" is identical across all 295 programs that carry it, and "College of
# Science Core Requirements" is identical across all 65 College of Science programs — the
# catalog files them under a type, not a program. Splitting the two lets the checklist say so
# instead of repeating the same paragraph under every major as if it were that major's own.
UNIVERSITY_REQUIREMENT_TYPES = {"university", "core", "world_language"}
# "college" was missing from the old CREDITLESS_REQUIREMENT_TYPES entirely, which silently
# dropped 183 of 303 college-level groups — "Curriculum and Degree Requirements for College of
# Science", "College of Agriculture Core Requirements" — anywhere the catalog stated no credit
# minimum for them. Not shown as unresolved; not shown at all.
COLLEGE_REQUIREMENT_TYPES = {"college"}
CREDITLESS_REQUIREMENT_TYPES = UNIVERSITY_REQUIREMENT_TYPES | COLLEGE_REQUIREMENT_TYPES


def _scope_for(requirement_type: str) -> Literal["university", "college", "program"]:
    if requirement_type in UNIVERSITY_REQUIREMENT_TYPES:
        return "university"
    if requirement_type in COLLEGE_REQUIREMENT_TYPES:
        return "college"
    return "program"


@dataclass
class UnresolvedRequirement:
    """A requirement the catalog states but does not express as a list of courses.

    THESE ARE THE MAJORITY, and hiding them was this app's worst bug. 749 of 928 crawled
    programs have at least one, averaging 2.6 per program against 3.8 that do carry courses —
    so a Women's, Gender & Sexuality Studies student saw two requirement groups, both from the
    major, and a coverage figure of 100% while 24 of their ~120 credits were accounted for.

    They are not a crawler failure. The catalog genuinely does not enumerate them:

        "Choose 1 course in 6 different disciplines within the College of Liberal Arts"
            — a rule over disciplines; there is no list to parse.
        "For a complete listing of University Core Course Selectives, visit the University
         Senate Website"
            — the list is on a different site.
        "Completion of 10100, 10200, 20100 and 20200 -or- 10500 and 20500 in one world language"
            — course numbers with no subject, because the subject is "any world language".

    So they cannot be planned and cannot be checked, and the honest thing is to say exactly
    that: show the requirement, show the catalog's own words, and never count it as satisfied.
    NOT DECIDABLE is a different fact from NOT MET, and rendering one as the other would tell a
    student they are behind when the truth is that we do not know.

    ``scope`` is the other half of being honest: "University Core Requirements" is not this
    program's problem, it is every Purdue undergraduate's, and a checklist that files it under
    Women's, Gender & Sexuality Studies as if the department wrote it is misleading in the
    other direction. See ``_scope_for``.
    """

    id: str
    name: str
    requirement_type: str
    credits_min: float | None
    scope: Literal["university", "college", "program"]
    # The catalog's own prose. It is the only thing we can honestly show, and it is usually
    # enough for a student to act on.
    raw_text: str | None = None


@dataclass
class ProgramCatalog:
    """Everything about one program the planner, the model, and the checker need."""

    program_id: str
    name: str
    catalog_year: str
    degree_type: str | None = None
    program_type: str | None = None
    campus: str | None = None
    total_credits_min: float | None = None
    notes: list[str] = field(default_factory=list)
    groups: list[RequirementGroup] = field(default_factory=list)
    courses: list[Course] = field(default_factory=list)
    # substitute code -> the PRIMARY course it counts as. Identity for everything unpaired.
    aliases: dict[str, str] = field(default_factory=dict)
    # Raw rows kept for the model's context document, which prints the database as stored
    # rather than as interpreted — see plan_context.
    prereq_rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    offering_rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Requirements the catalog states but does not enumerate. Kept OUT of `groups` on purpose:
    # they have no courses, so nothing can plan them, schedule them or check them — and letting
    # them into the planning path would only produce requirements the planner silently fails.
    # They are reported instead. See UnresolvedRequirement.
    unresolved: list[UnresolvedRequirement] = field(default_factory=list)
    # Subject code -> display name (e.g. "SPAN" -> "Spanish") for every language-sequence
    # requirement group this program has, computed BEFORE narrowing so it always lists every
    # choice regardless of what (if anything) the student already picked. See
    # `detect_language_sequences`/`narrow_language_menu`.
    available_languages: dict[str, str] = field(default_factory=dict)

    @property
    def by_code(self) -> dict[str, Course]:
        return {course.code: course for course in self.courses}

    @property
    def canonical(self) -> dict[str, str]:
        """code -> the primary course it counts as, following alias chains.

        An approved substitute satisfies the requirement AND the prerequisite its primary
        appears under, and taking both is taking one course twice. Collapsing before any of
        those three checks is what makes all three agree; doing it in only one is how you get
        a plan that covers the degree but still reports a prereq violation.
        """
        out: dict[str, str] = {}
        for course in self.courses:
            code, seen = course.code, {course.code}
            while self.aliases.get(code) and self.aliases[code] not in seen:
                code = self.aliases[code]
                seen.add(code)
            out[course.code] = code
        return out

    def credits(self, code: str) -> int:
        course = self.by_code.get(code)
        return course.credits if course else 0

    def major_subject(self) -> str:
        """The subject that contributes the most REQUIRED courses — the degree's own department.

        Derived rather than asked of the student: 'CS' is not a fact about a person, it is a
        fact about the program they picked, and a profile that names the wrong one silently
        disables the per-semester major-course cap.
        """
        counts: dict[str, int] = {}
        for group in self.groups:
            if group.selective:
                continue
            for code in group.courses:
                subject = normalize_course_code(code).split(" ")[0]
                counts[subject] = counts.get(subject, 0) + 1
        return max(counts, key=lambda s: (counts[s], s)) if counts else "CS"


def _connect() -> psycopg.Connection:
    return psycopg.connect(settings.academic_database_url, row_factory=dict_row)


def _parse_attributes(raw: str | None) -> list[str]:
    """`courses.attributes_raw` holds a JSON tag list when the loader wrote it, and Acalog's
    own free-text attribute string when the crawler did. Only the first is a tag list; prose
    is dropped rather than split into nonsense tags."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(tag) for tag in parsed] if isinstance(parsed, list) else []


def _chain_depths(prereq_groups: dict[str, list[list[str]]], known: set[str]) -> dict[str, int]:
    """Longest prerequisite chain behind each course. 0 = nothing has to come first.

    Used only for ORDERING the course list, so a course never appears above one it depends on.
    Cycle-safe: a course being resolved counts as 0 while in progress, so a bad edge cannot
    hang plan building — it costs ordering quality, never availability.
    """
    depth: dict[str, int] = {}
    resolving: set[str] = set()

    def resolve(code: str) -> int:
        if code in depth:
            return depth[code]
        if code in resolving or code not in known:
            return 0
        resolving.add(code)
        edges = [c for group in prereq_groups.get(code, []) for c in group if c in known]
        value = max((resolve(edge) + 1 for edge in edges), default=0)
        resolving.discard(code)
        depth[code] = value
        return value

    return {code: resolve(code) for code in known}


def _fetch_program_row(cur: psycopg.Cursor, program_id: str) -> dict[str, Any] | None:
    cur.execute(
        """
        SELECT p.id::text AS id, p.name, p.degree_type, p.program_type, p.campus,
               p.total_credits_min, cy.label AS catalog_year
        FROM programs p
        JOIN catalog_years cy ON cy.id = p.catalog_year_id
        WHERE p.id = %s
        """,
        (program_id,),
    )
    return cur.fetchone()


def _is_selective(requirement_type: str, credits_min: float | None,
                  option_credits: list[float]) -> bool:
    """Does this group mean "choose from this menu" or "take every one of these"?

    Two signals, and the crawled catalog needs both. The `requirement_type` alone is not
    enough: Purdue files "Required CS Major Math Courses (7-8 credits)" under `major` with
    four options worth 15 credits, which is a menu (MA 26100 *or* MA 27101, MA 26500 *or*
    MA 35100) wearing the word "Required" in its title. So the second signal is arithmetic —
    a group whose options total MORE credits than the group requires cannot be a take-all,
    whatever it calls itself.

    The converse is the check that keeps it honest: "Required CS Major Core Courses (21
    credits)" lists six courses totalling exactly 21, so it stays required. Equality is the
    discriminator, which is why the tolerance below is small rather than generous.

    A THIRD SIGNAL, added once resolving linked requirement lists (see
    `parse.requirements.resolve_requirement_links` in catalog_ingestion) started producing
    groups neither of the first two signals were built for: a resolved department page
    ("Spanish (SPAN)", 29 courses; "History (HIST)", 103; "Culture Course List", 619) states
    no credit target of its own — the real target lives on an ancestor group further up the
    tree — and its generic name never matches an "elective"/"choose" keyword either. Both
    signals defaulted these to "required", and `select_remaining_courses` dutifully added
    every single course from every one of them: one real profile's `remaining_courses` came
    back at 984 courses (should have been on the order of 100). No real required list in this
    catalog exceeds ~7 options (see this function's own examples above) — Purdue does not
    require a student to take every course a whole department teaches — so an OPTION COUNT
    this large is itself conclusive, with no credit annotation needed to read it.
    """
    if (requirement_type or "") in SELECTIVE_TYPES:
        return True
    if credits_min:
        return sum(option_credits) > float(credits_min) + 0.01
    return len(option_credits) > 10


# "Spanish Level I", "Modern Hebrew Level II", "Biblical Hebrew Level III" — Purdue's own
# course titles for a language-sequence course state the language name and level in plain
# text, so detection reads it off the title directly rather than a hand-maintained
# subject-code -> language-name table, which would silently go stale the next language a
# program's requirement page adds.
_LANGUAGE_LEVEL_RE = re.compile(r"^(.+?)\s+Level\s+([IVX]+)\b")
_ROMAN_NUMERALS = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}


@dataclass
class LanguageOption:
    code: str
    title: str
    level: int


def detect_language_sequences(
    group: RequirementGroup, by_code: dict[str, Course]
) -> dict[str, list["LanguageOption"]]:
    """Subject code -> that language's options, sorted by level. Empty if this isn't a
    language-sequence group at all (fewer than half its courses title-match), so a mixed
    group — mostly ordinary electives with one similarly-titled course — is left alone
    rather than partially, confusingly narrowed.

    GROUPED BY SUBJECT CODE, NOT BY THE PARSED LANGUAGE NAME. Hebrew's two tracks ("Modern
    Hebrew Level I", "Biblical Hebrew Level I") share one subject code, HEBR, and are not
    interchangeable — but `requirement_options.course_code` and `StudentProfile
    .language_of_interest` both speak in subject codes, not display names, so the unit a
    student picks has to be the same unit this groups by. A program where Modern and Biblical
    Hebrew used different codes would split naturally; this one does not, so they merge under
    HEBR, mixing both tracks' levels — a known, narrow limitation of using the subject code as
    the key, accepted because splitting on parsed name text would make the student's
    `language_of_interest` selection (a subject code) unable to name a track at all.
    """
    by_subject: dict[str, list[LanguageOption]] = {}
    matched = 0
    codes = group.courses
    for code in codes:
        course = by_code.get(code)
        if course is None:
            continue
        m = _LANGUAGE_LEVEL_RE.match(course.title)
        if not m:
            continue
        level = _ROMAN_NUMERALS.get(m.group(2).upper())
        if level is None:
            continue
        matched += 1
        subject = code.split(" ", 1)[0]
        by_subject.setdefault(subject, []).append(
            LanguageOption(code=code, title=course.title, level=level)
        )

    if not codes or matched < len(codes) / 2:
        return {}
    for options in by_subject.values():
        options.sort(key=lambda o: o.level)
    return by_subject


def narrow_language_menu(
    group: RequirementGroup, by_code: dict[str, Course], profile: StudentProfile
) -> RequirementGroup:
    """A detected language-sequence group, narrowed to what THIS student needs to see —
    their chosen language's full sequence (with a recommended starting course) if they
    picked one, or one representative entry-level course per language if they have not.
    Every other group (the overwhelming majority) is returned unchanged, by identity.

    Full sequence, not just the recommended course, once a language is picked: three rows
    is a perfectly reasonable menu size, and showing the whole sequence lets the student (or
    the model, deciding whether a prerequisite is satisfied) see the progression rather than
    a single opaque course code.
    """
    sequences = detect_language_sequences(group, by_code)
    if not sequences:
        return group

    language = (profile.language_of_interest or "").strip().upper() or None
    if language and language in sequences:
        chosen = sequences[language]
        target_level = profile.language_proficiency or 1
        # The option AT the student's level, or the closest one below it if they claim a
        # level this sequence doesn't reach — never silently drop their whole language.
        starting = max((o for o in chosen if o.level <= target_level),
                       key=lambda o: o.level, default=chosen[0])
        return RequirementGroup(
            id=group.id, name=group.name, selective=group.selective,
            credits_min=group.credits_min, raw_text=group.raw_text,
            options=[[o.code] for o in chosen], recommended_code=starting.code,
        )

    # No language picked yet (or one not actually offered by this program) — one entry-level
    # course per language, so the menu stays a manageable size until the student decides.
    return RequirementGroup(
        id=group.id, name=group.name, selective=group.selective,
        credits_min=group.credits_min, raw_text=group.raw_text,
        options=[[min(opts, key=lambda o: o.level).code] for opts in sequences.values()],
    )


def _fetch_groups(cur: psycopg.Cursor, program_id: str) -> list[RequirementGroup]:
    """The program's plannable requirement groups, in catalog display order.

    Groups with NO course options are dropped. Roughly two thirds of the crawled rows are
    prose — "About the Program", "GPA Requirements", "Pass/No Pass Option Policy" — and they
    are requirement_groups only in the sense that the page nested them there. Keeping them
    would silently inflate `requirement_coverage`, because a group with no courses has nothing
    missing and therefore scores as satisfied by an empty plan.
    """
    cur.execute(
        """
        SELECT rg.id::text AS group_id, rg.name, rg.requirement_type, rg.credits_min,
               rg.raw_text, rg.display_order,
               COALESCE(c.course_code, ro.course_code_raw) AS course_code,
               ro.credits AS option_credits, ro.is_selective_option
        FROM requirement_groups rg
        JOIN requirement_options ro ON ro.requirement_group_id = rg.id
        LEFT JOIN courses c ON c.id = ro.course_id
        WHERE rg.program_id = %s
          AND COALESCE(rg.requirement_type, '') <> ALL(%s)
        ORDER BY rg.display_order ASC, ro.display_order ASC
        """,
        (program_id, sorted(NON_REQUIREMENT_TYPES)),
    )

    groups: dict[str, RequirementGroup] = {}
    option_credits: dict[str, list[float]] = {}
    meta: dict[str, str] = {}
    # True while the previous row ended with "or", i.e. the next row continues that run.
    chaining: dict[str, bool] = {}
    for row in cur.fetchall():
        code = normalize_course_code((row["course_code"] or "").strip())
        if not code:
            continue
        group_id = row["group_id"]
        group = groups.get(group_id)
        if group is None:
            group = RequirementGroup(
                id=group_id,
                name=row["name"] or "Requirement",
                selective=False,          # decided below, once every option is counted
                credits_min=row["credits_min"],
                raw_text=row["raw_text"],
            )
            groups[group_id] = group
            option_credits[group_id] = []
            meta[group_id] = row["requirement_type"] or ""
            chaining[group_id] = False
        if code in group.courses:
            continue                      # the same course listed twice in one group
        credits = (float(row["option_credits"]) if row["option_credits"] is not None
                   else DEFAULT_OPTION_CREDITS)
        if chaining[group_id] and group.options:
            # Continues the previous "... or" — an alternative, not another requirement, so it
            # does not add to the group's credit total.
            group.options[-1].append(code)
        else:
            group.options.append([code])
            option_credits[group_id].append(credits)
        chaining[group_id] = bool(row["is_selective_option"])

    for group_id, group in groups.items():
        group.selective = _is_selective(meta[group_id], group.credits_min,
                                        option_credits[group_id])
    return list(groups.values())


def _fetch_unresolved(cur: psycopg.Cursor, program_id: str) -> list[UnresolvedRequirement]:
    """Requirements the catalog states but does not enumerate. See UnresolvedRequirement.

    Three filters, and each one is load-bearing:

    * NO COURSE OPTIONS — a group with courses is plannable and belongs in ``groups``.
    * NO CHILDREN — "Departmental/Program Major Courses (24 credits)" is a CONTAINER whose
      three children carry the actual requirements. Reporting the parent as well would tell a
      student they have four requirements where they have three, and double its credits.
    * NOT PROSE — "About the Program", "Disclaimer", "GPA Requirements" are page furniture.
      The discriminator is a stated credit minimum, or a type that is a requirement even when
      the catalog states no credits for it (University Core, Civics Literacy, Upper Level).
    """
    cur.execute(
        """
        SELECT rg.id::text AS id, rg.name, rg.requirement_type, rg.credits_min, rg.raw_text
        FROM requirement_groups rg
        WHERE rg.program_id = %s
          AND COALESCE(rg.requirement_type, '') <> ALL(%s)
          AND NOT EXISTS (SELECT 1 FROM requirement_options ro
                          WHERE ro.requirement_group_id = rg.id)
          AND NOT EXISTS (SELECT 1 FROM requirement_groups child
                          WHERE child.parent_group_id = rg.id)
          AND (rg.credits_min IS NOT NULL
               OR COALESCE(rg.requirement_type, '') = ANY(%s))
        ORDER BY rg.display_order ASC
        """,
        (program_id, sorted(PROSE_TYPES), sorted(CREDITLESS_REQUIREMENT_TYPES)),
    )
    return [
        UnresolvedRequirement(
            id=row["id"],
            name=(row["name"] or "Requirement").strip(),
            requirement_type=row["requirement_type"] or "",
            credits_min=row["credits_min"],
            scope=_scope_for(row["requirement_type"] or ""),
            raw_text=(row["raw_text"] or "").strip() or None,
        )
        for row in cur.fetchall()
    ]


def _fetch_courses(cur: psycopg.Cursor, codes: set[str]) -> dict[str, dict[str, Any]]:
    """Course rows by canonical code, newest catalog year wins when a code exists in several."""
    if not codes:
        return {}
    cur.execute(
        """
        SELECT DISTINCT ON (c.course_code)
               c.course_code, c.title, c.credit_hours_min, c.description, c.attributes_raw
        FROM courses c
        JOIN catalog_years cy ON cy.id = c.catalog_year_id
        WHERE c.course_code = ANY(%s)
        ORDER BY c.course_code, cy.start_year DESC
        """,
        (sorted(codes),),
    )
    return {row["course_code"]: row for row in cur.fetchall()}


def _fetch_prereqs(cur: psycopg.Cursor, codes: set[str]) -> dict[str, dict[str, Any]]:
    """Prerequisite rows, or {} when the advisor schema is absent.

    A missing schema must not take the app down: the planner degrades to the pre-advisor
    behaviour (no edges) and the warning says so, which is a knowable state rather than a 500.
    """
    if not codes:
        return {}
    try:
        cur.execute(
            """
            SELECT course_code, raw_text, prereq_groups, coreq_codes, confidence
            FROM advisor.course_prerequisites
            WHERE course_code = ANY(%s)
            """,
            (sorted(codes),),
        )
    except psycopg.errors.UndefinedTable:
        logger.warning("advisor.course_prerequisites is missing — planning without prereq edges")
        return {}
    return {row["course_code"]: row for row in cur.fetchall()}


def _fetch_offerings(cur: psycopg.Cursor, codes: set[str]) -> dict[str, dict[str, Any]]:
    if not codes:
        return {}
    try:
        cur.execute(
            """
            SELECT course_code, offered_terms, terms_observed
            FROM advisor.course_planner_terms
            WHERE course_code = ANY(%s)
            """,
            (sorted(codes),),
        )
    except psycopg.errors.UndefinedTable:
        logger.warning("advisor.course_planner_terms is missing — treating every course as "
                       "offered every term")
        return {}
    return {row["course_code"]: row for row in cur.fetchall()}


def _fetch_aliases(cur: psycopg.Cursor, codes: set[str]) -> dict[str, str]:
    """substitute code -> primary code. The table hangs the alias off the PRIMARY's row."""
    if not codes:
        return {}
    cur.execute(
        """
        SELECT ca.alias_code, c.course_code AS primary_code
        FROM course_aliases ca
        JOIN courses c ON c.id = ca.course_id
        WHERE c.course_code = ANY(%s)
        """,
        (sorted(codes),),
    )
    return {normalize_course_code(row["alias_code"]): row["primary_code"]
            for row in cur.fetchall()}


def load_program_catalog(program_id: str, *, extra_codes: set[str] | None = None,
                         profile: StudentProfile | None = None) -> ProgramCatalog:
    """Everything about one program, in one round trip's worth of queries.

    ``extra_codes`` pulls in courses outside the program's requirement rows — the student's
    completed transcript, typically, which has to be in the catalog for prerequisite checks
    and coverage to see it at all.

    ``profile``, if given, narrows any detected language-sequence requirement group down to
    what that student needs to see — see ``narrow_language_menu``. Optional and keyword-only
    because most callers (validation re-checking an EXISTING plan, for one) have no student
    preference to apply and must see every group exactly as the catalog states it.
    """
    try:
        uuid.UUID(program_id)
    except ValueError:
        raise ProgramNotFoundError(program_id) from None

    with _connect() as conn, conn.cursor() as cur:
        program = _fetch_program_row(cur, program_id)
        if program is None:
            raise ProgramNotFoundError(program_id)

        groups = _fetch_groups(cur, program_id)
        unresolved = _fetch_unresolved(cur, program_id)
        cur.execute(
            "SELECT note_text FROM program_notes WHERE program_id = %s ORDER BY note_type",
            (program_id,),
        )
        notes = [row["note_text"] for row in cur.fetchall()]

        codes = {code for group in groups for code in group.courses}
        codes |= {normalize_course_code(c) for c in (extra_codes or set())}

        course_rows = _fetch_courses(cur, codes)
        # Prerequisites reach outside the requirement list: CS 25100 requires CS 18200, which a
        # Machine Intelligence student takes but which some other program's option list might
        # not name. Pull the closure so the planner can see the whole chain it must sequence.
        prereq_rows = _fetch_prereqs(cur, set(course_rows))
        for _ in range(4):  # depth-bounded: prerequisite chains here run 4-5 courses deep
            referenced = {
                normalize_course_code(code)
                for row in prereq_rows.values()
                for code in (
                    [c for group in (row["prereq_groups"] or []) for c in group]
                    + list(row["coreq_codes"] or [])
                )
            }
            missing = referenced - set(course_rows)
            if not missing:
                break
            course_rows |= _fetch_courses(cur, missing)
            prereq_rows |= _fetch_prereqs(cur, missing)

        all_codes = set(course_rows)
        offering_rows = _fetch_offerings(cur, all_codes)
        aliases = _fetch_aliases(cur, all_codes)

    prereq_groups = {
        code: [[normalize_course_code(c) for c in group]
               for group in (row["prereq_groups"] or [])]
        for code, row in prereq_rows.items()
    }
    depths = _chain_depths(prereq_groups, all_codes)

    courses: list[Course] = []
    for code in sorted(all_codes, key=lambda c: (depths.get(c, 0), c)):
        row = course_rows[code]
        prereq_row = prereq_rows.get(code, {})
        groups_for_code = prereq_groups.get(code, [])
        offering = offering_rows.get(code)
        courses.append(
            Course(
                code=code,
                title=(row["title"] or "Untitled course").strip(),
                credits=(max(round(float(row["credit_hours_min"])), 0)
                         if row["credit_hours_min"] is not None else 0),
                # FLATTENED for display and for the planner's "why is this blocked" warnings;
                # `prereq_groups` below is the truth used to decide satisfaction, because a
                # flat AND-list cannot express "CS 25100 OR CS 25300" without demanding both.
                prereqs=sorted({c for group in groups_for_code for c in group}),
                prereq_groups=groups_for_code,
                coreqs=[normalize_course_code(c) for c in (prereq_row.get("coreq_codes") or [])],
                # No offerings row means no observation either way. Treating that as "every
                # term" is the same assumption the app made before offerings existed, and it is
                # the permissive one — the planner may schedule it and validation will not
                # invent a violation from missing data.
                offered_terms=list(offering["offered_terms"]) if offering else list(ALL_SEASONS),
                requirement_tags=_parse_attributes(row["attributes_raw"]),
                notes=row.get("description"),
            )
        )

    by_code = {course.code: course for course in courses}
    # Computed from the UNNARROWED groups, regardless of `profile` — this is "what languages
    # could this student pick", which has to survive narrowing already having happened, so a
    # student can change their mind without the choice list shrinking to just what they
    # already chose.
    available_languages: dict[str, str] = {}
    for group in groups:
        for subject, opts in detect_language_sequences(group, by_code).items():
            available_languages.setdefault(subject, opts[0].title.rsplit(" Level", 1)[0])

    if profile is not None:
        groups = [narrow_language_menu(group, by_code, profile) for group in groups]

    return ProgramCatalog(
        program_id=program["id"],
        name=program["name"],
        catalog_year=program["catalog_year"],
        degree_type=program["degree_type"],
        program_type=program["program_type"],
        campus=program["campus"],
        total_credits_min=program["total_credits_min"],
        notes=notes,
        groups=groups,
        courses=courses,
        aliases=aliases,
        prereq_rows=prereq_rows,
        offering_rows=offering_rows,
        unresolved=unresolved,
        available_languages=available_languages,
    )


def fetch_course_facts(codes: list[str]) -> list[dict[str, Any]]:
    """Prerequisites, corequisites and observed term offerings for named courses.

    THE ADVISOR SCHEMA IS INVISIBLE TO RAG, and that is the gap this closes. The retrieval
    corpus (``academic_rules``) is built from the crawled Acalog tables, which publish no
    prerequisites at all — so a student asking "what do I need before CS 25100?" was told
    "no prerequisites listed in the catalog", which is true of the chunk that was retrieved and
    false about the world. The app knows the answer: it is the same table the planner sequences
    every plan from. This makes that table answerable.

    Deterministic and cheap: an exact lookup on codes the question literally named, no
    embedding involved. Same tier as ``store.fetch_by_course_codes`` and merged alongside it.
    Returns [] when the advisor schema is absent, so retrieval degrades to Acalog alone.
    """
    if not codes:
        return []
    normalized = sorted({normalize_course_code(code) for code in codes})
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.course_code, c.title, c.credit_hours_min,
                       p.raw_text, p.prereq_groups, p.coreq_codes, p.confidence,
                       t.offered_terms, t.terms_observed
                FROM (SELECT DISTINCT ON (course_code) course_code, title, credit_hours_min
                      FROM courses WHERE course_code = ANY(%s)) c
                LEFT JOIN advisor.course_prerequisites p ON p.course_code = c.course_code
                LEFT JOIN advisor.course_planner_terms t ON t.course_code = c.course_code
                WHERE p.course_code IS NOT NULL OR t.course_code IS NOT NULL
                """,
                (normalized,),
            )
            rows = cur.fetchall()
    except psycopg.errors.UndefinedTable:
        return []
    except psycopg.Error as exc:
        logger.warning("course-facts lookup failed, continuing without it: %s", exc)
        return []

    facts: list[dict[str, Any]] = []
    for row in rows:
        lines = [f"{row['course_code']} — {row['title']}"]
        groups = row["prereq_groups"] or []
        if groups:
            expression = " AND ".join(
                "(" + " or ".join(group) + ")" if len(group) > 1 else group[0]
                for group in groups
            )
            lines.append(f"Prerequisites: {expression}.")
            if row["raw_text"]:
                lines.append(f"As published: {row['raw_text']}")
            # The confidence travels with the claim, because these edges are read from
            # course-detail pages rather than from the catalog and are the highest-risk rows
            # in the corpus. A reader is entitled to know which they are holding.
            lines.append(f"Source confidence: {row['confidence']}.")
        elif row["raw_text"] is not None:
            lines.append("Prerequisites: none on file for this course.")
        if row["coreq_codes"]:
            lines.append(
                f"Corequisites (may be taken the same semester): "
                f"{', '.join(row['coreq_codes'])}."
            )
        if row["offered_terms"]:
            observed = row["terms_observed"] or 0
            evidence = (f"observed across {observed} terms of class history" if observed
                        else "not observed in class history — this is an editorial guess")
            lines.append(f"Offered in: {', '.join(row['offered_terms'])} ({evidence}).")
        facts.append({
            # Negative ids cannot collide with academic_rules' serial primary keys, which is
            # what merge_and_budget dedupes on.
            "id": -abs(hash(row["course_code"])) % 10**9 - 1,
            "similarity": 1.0,
            "metadata": {"type": "course_requirements", "code": row["course_code"],
                         "source": "advisor schema"},
            "content": "\n".join(lines),
        })
    return facts


def select_remaining_courses(
    groups: list[RequirementGroup], completed: set[str], credits_of
) -> list[str]:
    """Turn requirement groups into the ordered course list a profile carries.

    Required groups contribute every unmet course. Selective groups contribute just enough
    unmet options — in catalog display order — to cover the group's credit target, counting
    completed courses toward that target first. Required blocks come before all selectives and
    duplicates keep their first slot.

    Order is not cosmetic: the deterministic planner reads it as the preference order, and it
    is the knob the Mode A revise agent turns via reorder/defer.
    """
    required: list[str] = []
    selective: list[str] = []
    seen: set[str] = set(completed)

    for group in groups:
        if group.selective:
            continue
        for run in group.options:
            # One course per alternative-run: a student who already took the honours version
            # has met the requirement, and one who has taken neither takes the first listed.
            if any(code in seen for code in run):
                continue
            required.append(run[0])
            seen.add(run[0])

    for group in groups:
        if not group.selective or not group.options:
            continue
        target = group.credits_min
        if target is None:
            target = min((credits_of(run[0]) or DEFAULT_OPTION_CREDITS)
                         for run in group.options)
        needed = float(target) - sum(
            credits_of(code) or DEFAULT_OPTION_CREDITS
            for code in group.courses if code in completed
        )
        for run in group.options:
            if needed <= 0:
                break
            if any(code in seen for code in run):
                continue
            selective.append(run[0])
            seen.add(run[0])
            needed -= credits_of(run[0]) or DEFAULT_OPTION_CREDITS

    return required + selective


def requirement_coverage(
    catalog: ProgramCatalog, satisfied: set[str]
) -> tuple[float, list[str]]:
    """Fraction of requirement groups met, plus what is short and by how much.

    ``satisfied`` must already be canonicalised: a student who took MA 16500 has met the group
    that lists MA 16100. Credits are counted from the code the GROUP lists, since the group is
    what defines the target.
    """
    if not catalog.groups:
        return 1.0, []
    canonical = catalog.canonical
    met = 0
    missing: list[str] = []
    for group in catalog.groups:
        if not group.selective:
            # A take-all group is met when every alternative-RUN has one member satisfied —
            # "MA 26100 or MA 27101" is one requirement, not two.
            short = [run[0] for run in group.options
                     if not any(canonical.get(c, c) in satisfied for c in run)]
            if short:
                missing.append(f"{group.name}: missing {', '.join(short)}")
            else:
                met += 1
            continue
        have = sum(catalog.credits(c) for c in group.courses
                   if canonical.get(c, c) in satisfied)
        target = float(group.credits_min or 0)
        if have >= target:
            met += 1
        else:
            missing.append(
                f"{group.name}: {have:g}/{target:g} credits "
                f"(options: {', '.join(group.courses)})"
            )
    return met / len(catalog.groups), missing
