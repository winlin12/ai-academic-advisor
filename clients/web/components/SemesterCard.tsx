"use client";

import AddCourseSearch from "./AddCourseSearch";
import CoursePill, { SemesterOption } from "./CoursePill";

type Course = {
  code: string;
  name: string;
};

type Props = {
  title: string;
  credits: number;
  warnings: string[];
  courses: Course[];
  // Which slot this card occupies in plan.semesters — the index POST /v1/plan/edit needs.
  index: number;
  semesterOptions: SemesterOption[];
  busy?: boolean;
  onMoveCourse: (courseCode: string, targetSemester: number) => void;
  onRemoveCourse: (courseCode: string) => void;
  onAddCourse: (courseCode: string, targetSemester: number) => void;
};

export default function SemesterCard({
  title,
  credits,
  warnings,
  courses,
  index,
  semesterOptions,
  busy = false,
  onMoveCourse,
  onRemoveCourse,
  onAddCourse,
}: Props) {
  return (
    <div className="card p-6">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="kicker">Semester {index + 1}</p>
          <h2 className="font-display mt-0.5 text-3xl font-semibold uppercase tracking-wide text-[var(--ink)]">
            {title}
          </h2>
        </div>
        <span className="rounded-lg border border-[var(--stroke-strong)] bg-[rgba(207,185,145,0.08)] px-3 py-1.5 text-sm font-semibold text-[var(--gold)]">
          {credits} cr
        </span>
      </div>

      {warnings.length > 0 ? (
        <div className="mb-4 space-y-1.5">
          {warnings.map((warning) => (
            <div
              key={warning}
              className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-1.5 text-xs font-medium text-amber-200"
            >
              ⚠ {warning}
            </div>
          ))}
        </div>
      ) : (
        <div className="mb-4 inline-flex items-center gap-1.5 rounded-full border border-emerald-500/25 bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-300">
          ✓ On track
        </div>
      )}

      <div className="space-y-2.5">
        {courses.map((course) => (
          <CoursePill
            key={course.code}
            code={course.code}
            name={course.name}
            semesterIndex={index}
            semesterOptions={semesterOptions}
            disabled={busy}
            onMove={(target) => onMoveCourse(course.code, target)}
            onRemove={() => onRemoveCourse(course.code)}
          />
        ))}

        {courses.length === 0 && (
          <p className="rounded-lg border border-dashed border-[var(--stroke)] px-4 py-6 text-center text-sm text-[var(--muted)]">
            No courses scheduled.
          </p>
        )}

        <AddCourseSearch
          disabled={busy}
          onAdd={(courseCode) => onAddCourse(courseCode, index)}
        />
      </div>
    </div>
  );
}
