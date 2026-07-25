"use client";

import { useState } from "react";
import type { RequirementBlockDetail, RequirementCourseOption } from "@/lib/api";

type Props = {
  blocks: RequirementBlockDetail[];
  // When set, shows the top-line "X of Y requirements met" bar — pass ProgramAudit's rollup.
  // Omit for the plain read-only catalog reference view (no student, nothing to total).
  totalRequirements?: number;
  satisfiedRequirements?: number;
};

// Renders a program's requirement tree: blocks -> rules -> courses. Doubles as both views:
//   - plain catalog reference (AcademicProgramDetail): no course carries `satisfied`, so every
//     status affordance below quietly no-ops and it reads as a neutral checklist template.
//   - personal degree audit (ProgramAudit): courses carry `satisfied`/`satisfied_by`, so the
//     same markup lights up as a MyPurduePlan-style "what's done, what's left" view.
// One component for both keeps them from silently drifting into two different renderings of
// the same requirement data.
// Programs commonly carry a dozen-plus blocks and 50-100+ course options (a real degree
// easily has that many total requirement rows) — expanding everything by default turns the
// page into a multi-thousand-pixel wall of text instead of a scannable checklist. So blocks
// start collapsed; the header alone (title, credit total, Met/In progress badge) already says
// enough to scan the whole tree at a glance, and "Expand all" is one click away when you
// actually want to read every course.
export default function RequirementTree({
  blocks,
  totalRequirements,
  satisfiedRequirements,
}: Props) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  function toggle(blockId: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(blockId)) next.delete(blockId);
      else next.add(blockId);
      return next;
    });
  }

  const showRollup =
    totalRequirements !== undefined && satisfiedRequirements !== undefined && totalRequirements > 0;

  return (
    <div className="space-y-4">
      {showRollup && (
        <div className="card p-4">
          <div className="flex items-center justify-between text-sm">
            <span className="font-semibold text-[var(--ink)]">
              {satisfiedRequirements} of {totalRequirements} requirements met
            </span>
            <span className="text-[var(--muted)]">
              {Math.round(((satisfiedRequirements ?? 0) / (totalRequirements || 1)) * 100)}%
            </span>
          </div>
          <div className="mt-2 h-2 overflow-hidden rounded-full bg-black/40">
            <div
              className="h-full rounded-full bg-gradient-to-r from-[var(--gold-soft)] to-emerald-400 transition-all"
              style={{
                width: `${Math.min(100, ((satisfiedRequirements ?? 0) / (totalRequirements || 1)) * 100)}%`,
              }}
            />
          </div>
        </div>
      )}

      {blocks.length > 1 && (
        <div className="flex justify-end gap-4 text-xs font-semibold">
          <button
            type="button"
            onClick={() => setExpanded(new Set(blocks.map((b) => b.id)))}
            className="text-[var(--muted)] hover:text-[var(--gold)]"
          >
            Expand all
          </button>
          <button
            type="button"
            onClick={() => setExpanded(new Set())}
            className="text-[var(--muted)] hover:text-[var(--gold)]"
          >
            Collapse all
          </button>
        </div>
      )}

      {blocks.length === 0 && (
        <p className="rounded-xl border border-[var(--stroke)] bg-black/20 p-4 text-sm text-[var(--muted)]">
          No requirement blocks on file for this program yet.
        </p>
      )}

      {blocks.map((block) => {
        const isCollapsed = !expanded.has(block.id);
        return (
          <div key={block.id} className="card overflow-hidden">
            <button
              type="button"
              onClick={() => toggle(block.id)}
              className="flex w-full items-center justify-between gap-3 px-5 py-4 text-left"
            >
              <div className="flex items-center gap-3">
                <span
                  className={`font-display text-xs transition-transform ${isCollapsed ? "" : "rotate-90"}`}
                  aria-hidden
                >
                  ▶
                </span>
                <span className="font-display text-lg font-semibold uppercase tracking-wide text-[var(--ink)]">
                  {block.title ?? "Requirements"}
                </span>
                {block.credits_text && (
                  <span className="rounded-full border border-[var(--stroke-strong)] px-2 py-0.5 text-xs font-semibold text-[var(--gold)]">
                    {block.credits_text} cr
                  </span>
                )}
              </div>
              <StatusBadge satisfied={block.satisfied} />
            </button>

            {!isCollapsed && (
              <div className="space-y-3 border-t border-[var(--stroke)] px-5 py-4">
                {block.rules.map((rule) => (
                  <RuleRow key={rule.id} rule={rule} />
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function RuleRow({ rule }: { rule: RequirementBlockDetail["rules"][number] }) {
  const courses = rule.options.flatMap((option) => option.courses);
  const isChoose = rule.rule_type === "choose";
  const label =
    isChoose && rule.credits_min
      ? `Choose ${rule.credits_min} credits from:`
      : isChoose
        ? "Choose one:"
        : null;

  // A narrative rule (GPA minimums, policy text) carries no courses — render its text plainly,
  // no checklist affordance, since there's nothing concrete to mark satisfied.
  if (courses.length === 0) {
    return (
      <p className="rounded-lg bg-black/20 px-3 py-2 text-sm text-[var(--muted)]">
        {rule.raw_text ?? "—"}
      </p>
    );
  }

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">
          {label ?? "Required"}
        </p>
        <StatusBadge satisfied={rule.satisfied} compact />
      </div>
      <ul className="space-y-1.5">
        {courses.map((course) => (
          <CourseRow key={course.id} course={course} />
        ))}
      </ul>
    </div>
  );
}

function CourseRow({ course }: { course: RequirementCourseOption }) {
  const hasProgress = course.satisfied !== undefined;
  return (
    <li
      className={`flex items-center gap-2.5 rounded-lg border px-3 py-2 text-sm ${
        hasProgress && course.satisfied
          ? "border-emerald-500/25 bg-emerald-500/[0.06]"
          : "border-[var(--stroke)] bg-black/20"
      }`}
    >
      {hasProgress && <CourseCheck satisfiedBy={course.satisfied_by} />}
      <span className="font-semibold text-[var(--gold)]">
        {course.course_code_text || "—"}
      </span>
      <span className="flex-1 text-[var(--ink)]">{course.course_title ?? course.raw_text}</span>
      {course.credits_text && (
        <span className="whitespace-nowrap text-xs text-[var(--muted)]">
          {course.credits_text} cr
        </span>
      )}
    </li>
  );
}

function CourseCheck({ satisfiedBy }: { satisfiedBy?: "completed" | "planned" | null }) {
  if (satisfiedBy === "completed") {
    return (
      <span
        title="Completed"
        className="flex h-4 w-4 flex-none items-center justify-center rounded-full bg-emerald-400 text-[0.6rem] font-bold text-black"
      >
        ✓
      </span>
    );
  }
  if (satisfiedBy === "planned") {
    return (
      <span
        title="Already in your plan"
        className="h-4 w-4 flex-none rounded-full border-2 border-dashed border-sky-400"
      />
    );
  }
  return <span className="h-4 w-4 flex-none rounded-full border border-[var(--stroke-strong)]" />;
}

function StatusBadge({ satisfied, compact = false }: { satisfied?: boolean | null; compact?: boolean }) {
  if (satisfied === undefined || satisfied === null) return null;
  const className = satisfied
    ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
    : "border-amber-500/30 bg-amber-500/10 text-amber-300";
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[0.65rem] font-semibold uppercase tracking-wide ${className}`}
    >
      {satisfied ? "Met" : compact ? "Remaining" : "In progress"}
    </span>
  );
}
