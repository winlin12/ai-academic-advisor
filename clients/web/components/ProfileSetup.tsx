"use client";

import { useState } from "react";
import CourseChipInput from "./CourseChipInput";
import { StudentProfile } from "@/lib/api";

type Props = {
  // Prefill for "edit profile"; omitted on first-run onboarding.
  initial?: StudentProfile;
  submitLabel: string;
  busy?: boolean;
  onSubmit: (profile: StudentProfile) => void;
  onCancel?: () => void;
};

const CURRENT_YEAR = new Date().getFullYear();

// The profile form that replaced the hardcoded demo profile: everything the planner
// needs, nothing it doesn't. A program picker (dropdown of real majors) lands once the
// programs table is populated — until then degree name is free text and the courses to
// schedule are chosen by hand (TODO Priority 3).
export default function ProfileSetup({
  initial,
  submitLabel,
  busy = false,
  onSubmit,
  onCancel,
}: Props) {
  const [name, setName] = useState(initial?.name ?? "");
  const [degreeProgram, setDegreeProgram] = useState(
    initial?.degree_program ?? "",
  );
  const [startTerm, setStartTerm] = useState(initial?.start_term ?? "fall");
  const [startYear, setStartYear] = useState(
    initial?.start_year ?? CURRENT_YEAR,
  );
  const [semestersToPlan, setSemestersToPlan] = useState(
    initial?.semesters_to_plan ?? 4,
  );
  const [maxCredits, setMaxCredits] = useState(
    initial?.max_credits_per_semester ?? 15,
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
  const [validationError, setValidationError] = useState<string | null>(null);

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!name.trim()) {
      setValidationError("Enter your name.");
      return;
    }
    if (!degreeProgram.trim()) {
      setValidationError("Enter your degree program.");
      return;
    }
    if (remainingCourses.length === 0) {
      setValidationError("Add at least one course to schedule.");
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

    onSubmit({
      name: name.trim(),
      degree_program: degreeProgram.trim(),
      program_id: initial?.program_id ?? null,
      completed_courses: completedCourses,
      remaining_courses: remainingCourses,
      start_term: startTerm,
      start_year: startYear,
      semesters_to_plan: semestersToPlan,
      max_credits_per_semester: maxCredits,
      preferences,
    });
  }

  return (
    <form onSubmit={handleSubmit} className="card card-accent p-6 md:p-8">
      <p className="kicker">Student Profile</p>
      <h2 className="font-display mt-1 text-3xl font-semibold uppercase tracking-wide text-[var(--ink)]">
        {initial ? "Edit your profile" : "Set up your plan"}
      </h2>
      <p className="mt-2 text-sm text-[var(--muted)]">
        The deterministic planner builds your semesters from this — prerequisites,
        term offerings, and credit caps are all validated server-side.
      </p>

      <div className="mt-6 grid grid-cols-1 gap-5 md:grid-cols-2">
        <label className="block">
          <span className="mb-1 block text-sm font-semibold">Name</span>
          <input
            type="text"
            value={name}
            disabled={busy}
            onChange={(event) => setName(event.target.value)}
            placeholder="e.g. Purdue Pete"
            className="field"
          />
        </label>

        <label className="block">
          <span className="mb-1 block text-sm font-semibold">Degree program</span>
          <input
            type="text"
            value={degreeProgram}
            disabled={busy}
            onChange={(event) => setDegreeProgram(event.target.value)}
            placeholder="e.g. MS Computer Science"
            className="field"
          />
        </label>

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
          hint="Counted toward prerequisites — the planner won't schedule these again."
          codes={completedCourses}
          onChange={setCompletedCourses}
          disabled={busy}
        />

        <CourseChipInput
          label="Courses to schedule"
          hint="What's left for your degree. The planner spreads these across semesters."
          codes={remainingCourses}
          onChange={setRemainingCourses}
          disabled={busy}
        />
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
