"""The degree as a checklist: which requirement each course fills, and which are still open.

WHY THIS EXISTS. The semester view answers "when am I taking things"; it cannot answer "am I
going to graduate". Those are different questions and the second is the one a student actually
worries about — myPurduePlan puts the requirement tree front and centre for exactly that reason.
A plan of eight tidy semesters that quietly misses a 3-credit selective looks fine right up
until it isn't.

It matters more now that nothing is auto-filled. The app used to pick a student's electives for
them and report 100% coverage; it no longer does, so the holes are real and this is where they
become actionable rather than just a lower percentage.

FOUR STATES, and the distinctions are the whole point:

    completed   already on the transcript. Nothing more to do — shown in gold.
    planned     scheduled in the plan on screen, not yet taken. Provisional.
    unfilled    nothing satisfies it, and we know what could. Red, with the options.
    unresolved  the catalog states the requirement but never enumerates it, so we cannot
                check it at all. Amber, with the catalog's own words.

"unfilled" is deliberately not "missing": it is a decision the student has not made, not a
mistake they made. And "unresolved" is deliberately not "unfilled": NOT DECIDABLE is a
different fact from NOT MET, and rendering one as the other tells a student they are behind
when the truth is that we do not know.

THE UNRESOLVED ONES ARE THE MAJORITY, and hiding them was this app's worst bug. 749 of 928
crawled programs have at least one; the average program has 3.8 checkable groups and 2.6
uncheckable. Women's, Gender & Sexuality Studies showed TWO requirement groups — both from the
major — and a coverage figure of 100%, while the Liberal Arts core, the university core, the
world-language requirement and 20 credits of electives were simply absent from the screen. See
``planner_db.UnresolvedRequirement`` for why the catalog does not enumerate them.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from app.models.schemas import StudentProfile
from app.services.plan_suggestions import suggest_courses
from app.services.plan_validation import legal_slots, normalize_code, term_sequence
from app.services.planner_db import DEFAULT_OPTION_CREDITS, ProgramCatalog, UnresolvedRequirement

SlotStatus = Literal["completed", "planned", "unfilled"]


class SlotOption(BaseModel):
    """One course a student could put in a slot, with the terms it would legally fit."""

    code: str
    title: str
    credits: int
    # Semester indices where adding this course creates no new violation — computed against the
    # plan on screen, so the UI can offer a term rather than making the student guess and then
    # rejecting them. Empty means "nowhere right now", which is itself worth showing.
    legal_semesters: list[int] = Field(default_factory=list)


class UnresolvedRequirementView(BaseModel):
    """A requirement we can show but cannot check. Carries the catalog's own text.

    ``scope`` says who else has this exact requirement: "university" ones are identical across
    every program that carries them (University Core, Civics Literacy, world language),
    "college" ones identical across every program in that college, and "program" is the rest —
    genuinely specific to this major. See ``planner_db._scope_for``.

    ``suggested_courses`` is a best-effort "start here" list, NOT verification — see
    ``plan_suggestions.suggest_courses``. Rendered with the exact same "choose a term and add
    it" affordance as a real slot's options (``SlotOption``, computed the same way via
    ``_options_for``) — a guess at WHICH requirement a course satisfies is not a guess at
    whether it can legally be scheduled, so there is no reason to make it read-only.
    """

    id: str
    name: str
    scope: Literal["university", "college", "program"] = "program"
    credits_min: float | None = None
    raw_text: str | None = None
    suggested_courses: list[SlotOption] = Field(default_factory=list)


class ComputedCourseView(BaseModel):
    """One course counting toward a computed requirement — informational only, no "add"
    action, because nothing needs picking: it either already qualifies or it doesn't."""

    code: str
    title: str
    credits: int
    status: SlotStatus


class ComputedRequirementView(BaseModel):
    """A requirement the catalog states as a CREDIT THRESHOLD over a course-number range
    ("at least 32 semester hours... at least junior-level (30000+)") rather than as any
    specific course list — so, unlike ``UnresolvedRequirementView``, it genuinely can be
    checked: sum the credits of whatever the student already has that clears the bar.

    DOUBLE-COUNTING IS THE CORRECT BEHAVIOR HERE, NOT A BUG TO GUARD AGAINST. Purdue's own
    catalog text for "Upper Level Requirement" says so directly: "Students should be able to
    fulfill most, if not all, of these credits within their major requirements." A course
    already satisfying a major requirement group still counts here too — this view is
    computed independently over every completed/planned course, never subtracting what
    another group already claimed.
    """

    id: str
    name: str
    credits_required: float
    credits_filled: float
    satisfied: bool
    contributing_courses: list[ComputedCourseView] = Field(default_factory=list)


# "...at least 32 semester hours of coursework..." — the credit target. Matches "32 credit
# hours" too; Purdue's own wording for this university-wide requirement varies slightly by
# catalog year, not by program, so this is stated once and applies everywhere it appears.
_CREDIT_TARGET_RE = re.compile(r"at least (\d+)\s*(?:semester|credit)\s*hours", re.IGNORECASE)
# "...at least junior-level (30000+) courses" — the course-number floor.
_LEVEL_FLOOR_RE = re.compile(r"\((\d{4,6})\s*\+\)")


def _parse_credit_threshold_rule(raw_text: str | None) -> tuple[float, int] | None:
    """(credits required, course-number floor) if ``raw_text`` states a rule in this shape,
    else None. Pattern-matched against the catalog's own words, not against a specific
    requirement's name or id — a different program, or a future catalog year, stating the
    identical kind of rule is picked up the same way, no hardcoded name required."""
    if not raw_text:
        return None
    credit_match = _CREDIT_TARGET_RE.search(raw_text)
    level_match = _LEVEL_FLOOR_RE.search(raw_text)
    if not credit_match or not level_match:
        return None
    return float(credit_match.group(1)), int(level_match.group(1))


def _course_number(code: str) -> int | None:
    """The numeric part of a course code ("STAT 35000" -> 35000), or None if it has none."""
    match = re.search(r"\d+", code)
    return int(match.group()) if match else None


class RequirementSlot(BaseModel):
    """One thing the student has to satisfy, and what (if anything) satisfies it.

    For a take-all group this is one alternative-RUN — "MA 26100 or MA 27101" is one slot, not
    two, because taking both would be taking one requirement twice. For a selective group it is
    one unit of the credit target.
    """

    status: SlotStatus
    # The course that fills it, when one does.
    filled_by: str | None = None
    title: str | None = None
    credits: int = 0
    # Which semester of the plan it sits in (0-based), for "planned" slots only.
    semester_index: int | None = None
    # What could fill it. Only populated for unfilled slots — a filled slot's alternatives are
    # noise, and on a 17-option selective menu they are a lot of noise.
    options: list[SlotOption] = Field(default_factory=list)
    # A label for an unfilled selective slot ("3 more credits from Selectives").
    label: str | None = None


class RequirementGroupProgress(BaseModel):
    id: str
    name: str
    selective: bool
    credits_required: float | None = None
    credits_filled: float = 0.0
    satisfied: bool = False
    slots: list[RequirementSlot] = Field(default_factory=list)


class RequirementProgressResponse(BaseModel):
    program_title: str
    catalog_year: str
    groups: list[RequirementGroupProgress] = Field(default_factory=list)
    # Counted over GROUPS, matching `planner_db.requirement_coverage`, so this number and the
    # one on the plan panel can never disagree.
    groups_satisfied: int = 0
    groups_total: int = 0
    credits_completed: float = 0.0
    credits_planned: float = 0.0
    # "Fall 2026", ... so the UI can name a legal semester rather than printing an index.
    semester_labels: list[str] = Field(default_factory=list)
    # Requirements the catalog states but never enumerates. They are NOT in `groups` and NOT in
    # the `groups_satisfied`/`groups_total` fraction — counting a requirement nobody can check
    # as either met or unmet would be a guess either way. The UI must show them, and must never
    # report a degree complete while any remain.
    unresolved: list[UnresolvedRequirementView] = Field(default_factory=list)
    # Requirements the catalog states as a CREDIT THRESHOLD rather than a course list — see
    # ComputedRequirementView. Genuinely checkable, unlike `unresolved`, so these are counted
    # into neither `unresolved` (they are not "can't check this") nor `groups`/
    # `groups_satisfied` (they are not a course-list group) — a third bucket for a third kind
    # of requirement.
    computed: list[ComputedRequirementView] = Field(default_factory=list)
    # Subject code -> display name ("SPAN" -> "Spanish") for every language-sequence
    # requirement this program has, regardless of what the student already picked — see
    # `planner_db.ProgramCatalog.available_languages`. Empty for a program with no detected
    # language-sequence requirement.
    available_languages: dict[str, str] = Field(default_factory=dict)


def _placement(semesters: list[list[str]]) -> dict[str, int]:
    """course code -> the semester index it is planned in (first wins)."""
    out: dict[str, int] = {}
    for index, codes in enumerate(semesters):
        for code in codes:
            out.setdefault(normalize_code(code), index)
    return out


def _options_for(
    codes: list[str],
    catalog: ProgramCatalog,
    profile: StudentProfile,
    semesters: list[list[str]],
    taken: set[str],
    *,
    limit: int = 24,
) -> list[SlotOption]:
    """The courses that could fill a slot, each with the terms it would legally land in.

    Already-used codes are excluded: offering a student a course their plan already contains is
    offering them a duplicate. The legality check is the same ``validate_semesters`` every other
    path uses, so an option shown here cannot be rejected by the edit route afterwards.
    """
    canonical = catalog.canonical
    by_code = catalog.by_code
    out: list[SlotOption] = []
    for code in codes[:limit]:
        course = by_code.get(code)
        if course is None or canonical.get(code, code) in taken:
            continue
        out.append(SlotOption(
            code=code,
            title=course.title,
            credits=course.credits,
            legal_semesters=legal_slots(semesters, profile, catalog.courses, code,
                                        canonical=canonical),
        ))
    return out


def _without_title(raw: str | None, name: str) -> str | None:
    """Drop the leading copy of the group's own name from its catalog prose.

    The crawler stores the heading as the first line of `raw_text`, so rendering both puts the
    title twice in a row on every unresolved card. Only an exact leading match is removed —
    anything else is the catalog's words and is left exactly as written.
    """
    if not raw:
        return None
    body = raw.strip()
    if body.startswith(name.strip()):
        body = body[len(name.strip()):].lstrip(" \n\t:-—")
    return body or None


def _credits_of(catalog: ProgramCatalog, code: str) -> float:
    return float(catalog.credits(code) or DEFAULT_OPTION_CREDITS)


def requirement_progress(
    profile: StudentProfile,
    catalog: ProgramCatalog,
    semesters: list[list[str]],
) -> RequirementProgressResponse:
    """Cross-reference the program's requirements against the transcript and the plan."""
    canonical = catalog.canonical
    completed = {canonical.get(normalize_code(c), normalize_code(c))
                 for c in profile.completed_courses}
    placed = _placement(semesters)
    planned = {canonical.get(code, code) for code in placed}
    by_code = catalog.by_code

    def status_of(code: str) -> SlotStatus | None:
        key = canonical.get(code, code)
        if key in completed:
            return "completed"
        if key in planned:
            return "planned"
        return None

    groups: list[RequirementGroupProgress] = []
    credits_completed = 0.0
    credits_planned = 0.0

    for group in catalog.groups:
        slots: list[RequirementSlot] = []
        filled_credits = 0.0

        if not group.selective:
            # TAKE-ALL: one slot per alternative-run, so "MA 26100 or MA 27101" reads as the
            # single requirement it is.
            for run in group.options:
                hit = next((code for code in run if status_of(code)), None)
                if hit is not None:
                    state = status_of(hit)
                    course = by_code.get(hit)
                    filled_credits += _credits_of(catalog, hit)
                    slots.append(RequirementSlot(
                        status=state or "planned",
                        filled_by=hit,
                        title=course.title if course else None,
                        credits=course.credits if course else 0,
                        semester_index=placed.get(hit) if state == "planned" else None,
                    ))
                else:
                    slots.append(RequirementSlot(
                        status="unfilled",
                        label=" or ".join(run),
                        credits=int(_credits_of(catalog, run[0])),
                        options=_options_for(run, catalog, profile, semesters,
                                             completed | planned),
                    ))
        else:
            # SELECTIVE: a credit target met by whichever options the student has. Everything
            # already counting toward it is listed, then one unfilled slot for the shortfall —
            # rather than one per missing course, because the student chooses how to make it up
            # and a 3-credit gap might be one course or two.
            chosen = [code for code in group.courses if status_of(code)]
            for code in chosen:
                state = status_of(code)
                course = by_code.get(code)
                filled_credits += _credits_of(catalog, code)
                slots.append(RequirementSlot(
                    status=state or "planned",
                    filled_by=code,
                    title=course.title if course else None,
                    credits=course.credits if course else 0,
                    semester_index=placed.get(code) if state == "planned" else None,
                ))
            target = float(group.credits_min or 0) or min(
                (_credits_of(catalog, run[0]) for run in group.options), default=0.0)
            short = target - filled_credits
            if short > 0.01:
                remaining = [run[0] for run in group.options
                             if canonical.get(run[0], run[0]) not in completed | planned]
                slots.append(RequirementSlot(
                    status="unfilled",
                    label=f"{short:g} more credit{'s' if short != 1 else ''} from this list",
                    credits=int(short),
                    options=_options_for(remaining, catalog, profile, semesters,
                                         completed | planned),
                ))

        satisfied = all(slot.status != "unfilled" for slot in slots) and bool(slots)
        for slot in slots:
            if slot.status == "completed":
                credits_completed += slot.credits
            elif slot.status == "planned":
                credits_planned += slot.credits

        groups.append(RequirementGroupProgress(
            id=group.id,
            name=group.name,
            selective=group.selective,
            credits_required=group.credits_min,
            credits_filled=filled_credits,
            satisfied=satisfied,
            slots=slots,
        ))

    taken = frozenset(completed | planned)

    # A THIRD BUCKET, separate from `groups` and `unresolved`. Some catalog prose states a
    # credit threshold over a course-NUMBER range rather than a course list ("at least 32
    # semester hours... junior-level (30000+)") — genuinely checkable, just not against a
    # fixed option list, so it does not fit `RequirementGroupProgress`'s slot model either.
    unresolved_items: list[UnresolvedRequirement] = []
    computed: list[ComputedRequirementView] = []
    for item in catalog.unresolved:
        threshold = _parse_credit_threshold_rule(item.raw_text)
        if threshold is None:
            unresolved_items.append(item)
            continue
        credits_required, level_floor = threshold
        contributing = []
        credits_filled = 0.0
        for code in sorted(completed | planned):
            number = _course_number(code)
            if number is None or number < level_floor:
                continue
            course = by_code.get(code)
            if course is None:
                continue
            credits_filled += course.credits
            contributing.append(ComputedCourseView(
                code=code, title=course.title, credits=course.credits,
                status="completed" if code in completed else "planned",
            ))
        computed.append(ComputedRequirementView(
            id=item.id, name=item.name, credits_required=credits_required,
            credits_filled=credits_filled, satisfied=credits_filled >= credits_required,
            contributing_courses=contributing,
        ))

    return RequirementProgressResponse(
        unresolved=[
            UnresolvedRequirementView(
                id=item.id, name=item.name, scope=item.scope, credits_min=item.credits_min,
                raw_text=_without_title(item.raw_text, item.name),
                suggested_courses=_options_for(
                    [c.code for c in suggest_courses(item.name, item.raw_text, exclude=taken)],
                    catalog, profile, semesters, taken,
                ),
            )
            for item in unresolved_items
        ],
        computed=computed,
        program_title=catalog.name,
        catalog_year=catalog.catalog_year,
        groups=groups,
        groups_satisfied=sum(1 for g in groups if g.satisfied),
        groups_total=len(groups),
        credits_completed=credits_completed,
        credits_planned=credits_planned,
        available_languages=catalog.available_languages,
    )


def semester_labels(profile: StudentProfile) -> list[str]:
    """"Fall 2026", ... — so the UI can name a legal semester instead of printing an index."""
    return [f"{term.capitalize()} {year}"
            for term, year in term_sequence(profile, profile.semesters_to_plan)]
