"use client";

import { useEffect, useState } from "react";
import CourseChipInput from "./CourseChipInput";
import ProgramPicker from "./ProgramPicker";
import {
  fetchAvailableLanguages,
  fetchModelStatus,
  ModelStatus,
  selectModel,
  StudentProfile,
} from "@/lib/api";

const PROFICIENCY_LABELS: Record<number, string> = {
  1: "New to this language",
  2: "Some experience",
  3: "Comfortable",
};

type Props = {
  // Prefill for "edit profile"; omitted on first-run onboarding.
  initial?: StudentProfile;
  // Carries over the previous choice on "edit profile"; defaults to on (today's only
  // behavior) for first-run onboarding.
  initialAiMode?: boolean;
  submitLabel: string;
  busy?: boolean;
  onSubmit: (profile: StudentProfile, aiMode: boolean) => void;
  onCancel?: () => void;
};

const CURRENT_YEAR = new Date().getFullYear();

// The profile form. THE MAJOR IS THE LOAD-BEARING FIELD: picking a real program sets
// `program_id`, and the backend then derives the whole course list from that program's
// requirement rows in Postgres — required groups first, then just enough options from each
// selective menu to cover its credit target, with the student's completed courses counted
// toward it. Nothing here has to list a course by hand.
//
// The manual course list survives for the case the picker cannot serve: a program the crawl
// has not reached, or a student who wants to plan a handful of specific courses. That path
// gets a deterministic plan only — the AI planner refuses without a program, because there is
// no honest way to write a plan of study for a degree nobody named.
export default function ProfileSetup({
  initial,
  initialAiMode,
  submitLabel,
  busy = false,
  onSubmit,
  onCancel,
}: Props) {
  // OFF means the model never runs and nothing gets auto-placed — same planner view, but
  // every semester starts empty and the student fills it themselves from the Requirements
  // tab. Defaults on: that's the app's only behavior before this existed, and an unlabeled
  // toggle silently flipping to "nothing happens" would be a worse surprise than the reverse.
  const [aiMode, setAiMode] = useState(initialAiMode ?? true);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [modelSwitching, setModelSwitching] = useState(false);

  useEffect(() => {
    if (!aiMode) return;
    fetchModelStatus().then(setModelStatus).catch(() => setModelStatus(null));
  }, [aiMode]);

  async function handleModelChange(name: string) {
    if (!modelStatus || name === modelStatus.current) return;
    setModelSwitching(true);
    try {
      setModelStatus(await selectModel(name));
    } catch {
      // The picker in the nav bar shows the authoritative state and any error detail;
      // this dropdown just refetches so it isn't left claiming a switch that didn't happen.
      fetchModelStatus().then(setModelStatus).catch(() => {});
    } finally {
      setModelSwitching(false);
    }
  }

  const [profileLabel, setProfileLabel] = useState(initial?.profile_label ?? "");
  const [programId, setProgramId] = useState<string | null>(initial?.program_id ?? null);
  const [degreeProgram, setDegreeProgram] = useState(initial?.degree_program ?? "");
  const [startTerm, setStartTerm] = useState(initial?.start_term ?? "fall");
  const [startYear, setStartYear] = useState(initial?.start_year ?? CURRENT_YEAR);
  const [semestersToPlan, setSemestersToPlan] = useState(initial?.semesters_to_plan ?? 8);
  const [maxCredits, setMaxCredits] = useState(initial?.max_credits_per_semester ?? 15);
  const [maxMajorCourses, setMaxMajorCourses] = useState(
    initial?.max_major_courses_per_semester ?? 3,
  );
  const [targetGraduation, setTargetGraduation] = useState(
    String(initial?.preferences?.target_graduation ?? ""),
  );
  const [completedCourses, setCompletedCourses] = useState<string[]>(
    initial?.completed_courses ?? [],
  );
  const [remainingCourses, setRemainingCourses] = useState<string[]>(
    initial?.remaining_courses ?? [],
  );
  const [languageOfInterest, setLanguageOfInterest] = useState<string>(
    initial?.language_of_interest ?? "",
  );
  const [languageProficiency, setLanguageProficiency] = useState<number>(
    initial?.language_proficiency ?? 1,
  );
  const [availableLanguages, setAvailableLanguages] = useState<Record<string, string>>({});
  const [validationError, setValidationError] = useState<string | null>(null);

  const programChosen = Boolean(programId);

  // Only meaningful once a program is chosen — a language-sequence requirement is a property
  // of THAT program's own catalog, not something to guess at before one is picked. Guarded
  // against out-of-order responses the same way ProgramPicker guards its own search: a
  // program switch mid-flight must not let an earlier program's stale language list win.
  useEffect(() => {
    if (!programId) {
      setAvailableLanguages({});
      return;
    }
    let cancelled = false;
    fetchAvailableLanguages({
      profile_label: "",
      degree_program: degreeProgram,
      program_id: programId,
      completed_courses: [],
      remaining_courses: [],
      start_term: startTerm,
      start_year: startYear,
      semesters_to_plan: semestersToPlan,
      max_credits_per_semester: maxCredits,
      preferences: {},
    })
      .then((languages) => {
        if (!cancelled) setAvailableLanguages(languages);
      })
      .catch(() => {
        if (!cancelled) setAvailableLanguages({});
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [programId]);

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!programChosen && !degreeProgram.trim()) {
      setValidationError("Choose your major, or enter a degree program by hand.");
      return;
    }
    // Only the manual path needs courses typed in. With a program chosen the backend derives
    // them, and anything listed here is ignored server-side — so demanding it would be asking
    // for work that gets thrown away.
    if (!programChosen && remainingCourses.length === 0) {
      setValidationError(
        "Add at least one course to schedule, or choose a major so your courses can be " +
          "pulled from the catalog.",
      );
      return;
    }
    setValidationError(null);

    const preferences: StudentProfile["preferences"] = {
      ...(initial?.preferences ?? {}),
    };
    if (targetGraduation.trim()) {
      preferences.target_graduation = targetGraduation.trim();
    } else {
      delete preferences.target_graduation;
    }

    onSubmit(
      {
        profile_label: profileLabel.trim(),
        degree_program: degreeProgram.trim(),
        program_id: programId,
        completed_courses: completedCourses,
        remaining_courses: programChosen ? [] : remainingCourses,
        start_term: startTerm,
        start_year: startYear,
        semesters_to_plan: semestersToPlan,
        max_credits_per_semester: maxCredits,
        max_major_courses_per_semester: maxMajorCourses,
        // Kept equal to the hard cap: two limits on the same quantity, one of which is meant
        // to be broken when convenient, is a judgement call that makes plans worse rather
        // than better. Lower it here to bring the soft preference back.
        preferred_major_courses_per_semester: maxMajorCourses,
        language_of_interest: languageOfInterest || null,
        language_proficiency: languageOfInterest ? languageProficiency : null,
        preferences,
      },
      aiMode,
    );
  }

  return (
    <form onSubmit={handleSubmit} className="card card-accent p-6 md:p-8">
      <p className="kicker">Student Profile</p>
      <h2 className="font-display mt-1 text-3xl font-semibold uppercase tracking-wide text-[var(--ink)]">
        {initial ? "Edit your profile" : "Set up your plan"}
      </h2>
      <p className="mt-2 text-sm text-[var(--muted)]">
        Pick your major and the AI planner builds a semester-by-semester plan from that
        program&rsquo;s real requirements — then every placement is re-checked against
        prerequisites, term offerings and your credit caps before you see it.
      </p>

      <div className="mt-6 rounded-xl border border-[var(--stroke)] bg-black/25 p-4">
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="text-sm font-semibold text-[var(--ink)]">AI Mode</p>
            <p className="mt-0.5 text-xs text-[var(--muted)]">
              {aiMode
                ? "The local model writes a first draft of your schedule; you edit from there."
                : "Nothing is auto-placed and no model runs. You'll see the same planner, " +
                  "with every semester empty — fill it yourself from the Requirements tab."}
            </p>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={aiMode}
            disabled={busy}
            onClick={() => setAiMode((v) => !v)}
            className={
              "relative h-6 w-11 shrink-0 rounded-full border transition disabled:cursor-not-allowed disabled:opacity-50 " +
              (aiMode
                ? "border-[var(--gold)] bg-[var(--gold)]/30"
                : "border-[var(--stroke)] bg-black/40")
            }
          >
            <span
              className={
                "absolute top-0.5 h-4 w-4 rounded-full bg-[var(--ink)] transition " +
                (aiMode ? "left-[1.375rem]" : "left-0.5")
              }
            />
          </button>
        </div>

        {aiMode && (
          <label className="mt-3 block border-t border-[var(--stroke)] pt-3">
            <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">
              Model
            </span>
            <select
              value={modelStatus?.switching_to ?? modelStatus?.current ?? ""}
              disabled={busy || modelSwitching || !modelStatus}
              onChange={(event) => handleModelChange(event.target.value)}
              className="field text-sm"
            >
              {!modelStatus && <option value="">Loading…</option>}
              {modelStatus?.available.map((option) => (
                <option key={option.name} value={option.name}>
                  {option.label}
                </option>
              ))}
            </select>
            <p className="mt-1 text-xs text-[var(--muted)]">
              {modelSwitching || modelStatus?.switching_to
                ? "Switching models — this stops and relaunches it, usually 30-90s."
                : (modelStatus?.available.find((m) => m.name === modelStatus.current)?.blurb ??
                  "")}
            </p>
          </label>
        )}
      </div>

      <div className="mt-6 grid grid-cols-1 gap-5 md:grid-cols-2">
        <label className="block md:col-span-2">
          <span className="mb-1 block text-sm font-semibold">
            Profile name{" "}
            <span className="font-normal text-[var(--muted)]">(optional)</span>
          </span>
          <input
            type="text"
            value={profileLabel}
            disabled={busy}
            onChange={(event) => setProfileLabel(event.target.value)}
            placeholder="e.g. backup CS plan"
            className="field"
          />
          <p className="mt-1 text-xs text-[var(--muted)]">
            Just for telling saved profiles apart on this device — never sent to the model,
            and not your real name.
          </p>
        </label>

        <div className="md:col-span-2">
          <ProgramPicker
            label="Major"
            hint="Search the degree catalog. Your courses are pulled from this program's requirements."
            programId={programId}
            degreeText={degreeProgram}
            disabled={busy}
            onChange={(nextId, nextText) => {
              setProgramId(nextId);
              setDegreeProgram(nextText);
            }}
          />
        </div>

        <label className="block">
          <span className="mb-1 block text-sm font-semibold">Start term</span>
          <select
            value={startTerm}
            disabled={busy}
            onChange={(event) => setStartTerm(event.target.value)}
            className="field"
          >
            <option value="fall">Fall</option>
            <option value="spring">Spring</option>
            <option value="summer">Summer</option>
          </select>
        </label>

        <label className="block">
          <span className="mb-1 block text-sm font-semibold">Start year</span>
          <input
            type="number"
            min={CURRENT_YEAR - 6}
            max={CURRENT_YEAR + 6}
            value={startYear}
            disabled={busy}
            onChange={(event) => setStartYear(Number(event.target.value))}
            className="field"
          />
        </label>

        <label className="block">
          <span className="mb-1 block text-sm font-semibold">
            Semesters to plan
          </span>
          <input
            type="number"
            min={1}
            max={12}
            value={semestersToPlan}
            disabled={busy}
            onChange={(event) => setSemestersToPlan(Number(event.target.value))}
            className="field"
          />
        </label>

        <label className="block">
          <span className="mb-1 block text-sm font-semibold">
            Max credits per semester
          </span>
          <input
            type="number"
            min={1}
            max={24}
            value={maxCredits}
            disabled={busy}
            onChange={(event) => setMaxCredits(Number(event.target.value))}
            className="field"
          />
        </label>

        <label className="block md:col-span-2">
          <span className="mb-1 block text-sm font-semibold">
            Max major courses per semester
          </span>
          <input
            type="number"
            min={1}
            max={8}
            value={maxMajorCourses}
            disabled={busy}
            onChange={(event) => setMaxMajorCourses(Number(event.target.value))}
            className="field"
          />
          <p className="mt-1 text-xs text-[var(--muted)]">
            Separate from the credit cap. Four 4-credit CS courses and one CS course plus
            three gen-eds are both 16 credits and nothing like the same semester.
          </p>
        </label>

        {programChosen && Object.keys(availableLanguages).length > 0 && (
          <>
            <label className="block">
              <span className="mb-1 block text-sm font-semibold">
                Language of interest{" "}
                <span className="font-normal text-[var(--muted)]">(optional)</span>
              </span>
              <select
                value={languageOfInterest}
                disabled={busy}
                onChange={(event) => setLanguageOfInterest(event.target.value)}
                className="field"
              >
                <option value="">Not sure yet</option>
                {Object.entries(availableLanguages)
                  .sort((a, b) => a[1].localeCompare(b[1]))
                  .map(([code, name]) => (
                    <option key={code} value={code}>
                      {name}
                    </option>
                  ))}
              </select>
              <p className="mt-1 text-xs text-[var(--muted)]">
                Narrows your language requirement to this language&rsquo;s own sequence
                instead of every language the catalog offers.
              </p>
            </label>

            {languageOfInterest && (
              <label className="block">
                <span className="mb-1 block text-sm font-semibold">
                  Your level in {availableLanguages[languageOfInterest]}
                </span>
                <select
                  value={languageProficiency}
                  disabled={busy}
                  onChange={(event) => setLanguageProficiency(Number(event.target.value))}
                  className="field"
                >
                  {[1, 2, 3].map((level) => (
                    <option key={level} value={level}>
                      {PROFICIENCY_LABELS[level]}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </>
        )}

        <label className="block md:col-span-2">
          <span className="mb-1 block text-sm font-semibold">
            Target graduation{" "}
            <span className="font-normal text-[var(--muted)]">(optional)</span>
          </span>
          <input
            type="text"
            value={targetGraduation}
            disabled={busy}
            onChange={(event) => setTargetGraduation(event.target.value)}
            placeholder="e.g. Spring 2028"
            className="field"
          />
        </label>
      </div>

      <div className="mt-6 space-y-6">
        <CourseChipInput
          label="Completed courses"
          hint="Counted toward prerequisites and requirements — the planner won't schedule these again."
          codes={completedCourses}
          onChange={setCompletedCourses}
          disabled={busy}
        />

        {programChosen ? (
          <div className="rounded-xl border border-[var(--stroke)] bg-black/25 p-4">
            <p className="text-sm font-semibold text-[var(--gold)]">
              Courses to schedule: pulled from your major
            </p>
            <p className="mt-1 text-xs text-[var(--muted)]">
              {degreeProgram}&rsquo;s requirement groups decide what you still need — every
              required course, plus enough options from each selective list to cover its
              credits. Anything in &ldquo;completed&rdquo; above is subtracted first.
            </p>
          </div>
        ) : (
          <CourseChipInput
            label="Courses to schedule"
            hint="What's left for your degree. Choose a major above to have these filled in for you."
            codes={remainingCourses}
            onChange={setRemainingCourses}
            disabled={busy}
          />
        )}
      </div>

      {validationError && (
        <p className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm text-red-300">
          {validationError}
        </p>
      )}

      <div className="mt-8 flex flex-col gap-3 sm:flex-row">
        <button type="submit" disabled={busy} className="btn-gold px-6 py-3">
          {busy ? "Working…" : submitLabel}
        </button>
        {onCancel && (
          <button
            type="button"
            disabled={busy}
            onClick={onCancel}
            className="btn-ghost px-6 py-3"
          >
            Cancel
          </button>
        )}
      </div>
    </form>
  );
}
