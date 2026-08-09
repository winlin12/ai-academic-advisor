"use client";

import { useState } from "react";
import type {
  ComputedRequirementView,
  RequirementGroupProgress,
  RequirementProgressResponse,
  RequirementSlot,
  SlotOption,
  UnresolvedRequirementView,
  UnresolvedScope,
} from "@/lib/api";

type Props = {
  progress: RequirementProgressResponse;
  busy: boolean;
  // Puts a chosen course into a chosen semester. Goes through the deterministic edit route, so
  // the plan comes back re-validated.
  onFill: (courseCode: string, semesterIndex: number) => void;
};

// THE DEGREE AS A CHECKLIST — the view that answers "am I going to graduate", which the
// semester grid cannot. A plan of eight tidy semesters that quietly misses a 3-credit selective
// looks fine right up until it isn't.
//
// The colour IS the information, so it carries no other meaning anywhere in this component:
//
//   gold   done — on the transcript already
//   ink    planned — scheduled, not yet taken. Provisional, and deliberately quiet.
//   red    unfilled — nothing satisfies it, and it is the student's call what does
//
// Colour is never the ONLY signal: every slot also carries a glyph and a word, because roughly
// one man in twelve cannot reliably separate the red from the gold.

const STATUS_STYLE: Record<RequirementSlot["status"], string> = {
  completed: "border-[var(--stroke-strong)] bg-[rgba(207,185,145,0.14)] text-[var(--gold)]",
  planned: "border-[var(--stroke)] bg-black/25 text-[var(--ink)]",
  unfilled: "border-red-500/40 bg-red-500/[0.07] text-red-300",
};

const STATUS_GLYPH: Record<RequirementSlot["status"], string> = {
  completed: "✓",
  planned: "•",
  unfilled: "!",
};

const STATUS_WORD: Record<RequirementSlot["status"], string> = {
  completed: "Completed",
  planned: "Planned",
  unfilled: "Not filled",
};

// Groups the "can't be checked" pile by who else has it. A Women's, Gender & Sexuality
// Studies student and a Computer Science student see the exact same "University Core
// Requirements" paragraph — filing it under WGSS as if the department wrote it would be its
// own small lie. Order is broadest first: university, then college, then this program alone.
const SCOPE_ORDER: UnresolvedScope[] = ["university", "college", "program"];

const SCOPE_META: Record<UnresolvedScope, { heading: string; blurb: string }> = {
  university: {
    heading: "University-wide",
    blurb: "Every Purdue undergraduate has these — they don't come from this major.",
  },
  college: {
    heading: "College-wide",
    blurb: "Everyone in your college has these, whatever their major.",
  },
  program: {
    heading: "Specific to this program",
    blurb: "Stated in the catalog as prose rather than a list of courses.",
  },
};

export default function RequirementChecklist({ progress, busy, onFill }: Props) {
  const done = progress.groups_satisfied;
  const total = progress.groups_total;

  return (
    <div className="space-y-4">
      <div className="card p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="kicker">Degree requirements</p>
            <p className="mt-1 text-sm text-[var(--muted)]">
              {progress.program_title} · {progress.catalog_year}
            </p>
          </div>
          <span
            className={`rounded-full border px-3 py-1 text-xs font-semibold ${
              done === total
                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                : "border-amber-500/30 bg-amber-500/10 text-amber-300"
            }`}
          >
            {done}/{total} checkable groups filled
          </span>
        </div>
        {progress.unresolved.length > 0 && (
          <p className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/[0.07] px-3 py-2 text-xs text-amber-200/90">
            <span className="font-semibold">
              {progress.unresolved.length} more requirement
              {progress.unresolved.length === 1 ? "" : "s"} can&rsquo;t be checked automatically.
            </span>{" "}
            The catalog states {progress.unresolved.length === 1 ? "it" : "them"} in prose
            rather than as a list of courses, so this app can neither plan nor verify{" "}
            {progress.unresolved.length === 1 ? "it" : "them"} — they&rsquo;re listed at the
            bottom, with a best-effort guess at courses that might fit.
          </p>
        )}
        {progress.computed.length > 0 && (
          <div className="mt-3 space-y-2">
            {progress.computed.map((item) => (
              <ComputedBar key={item.id} item={item} />
            ))}
          </div>
        )}
        <div className="mt-4 flex flex-wrap gap-x-6 gap-y-1 text-xs text-[var(--muted)]">
          <span>
            <span className="font-semibold text-[var(--gold)]">
              {progress.credits_completed}
            </span>{" "}
            credits completed
          </span>
          <span>
            <span className="font-semibold text-[var(--ink)]">{progress.credits_planned}</span>{" "}
            credits planned
          </span>
        </div>
        <div className="mt-4 flex flex-wrap gap-3 border-t border-[var(--stroke)] pt-3 text-xs">
          {(["completed", "planned", "unfilled"] as const).map((status) => (
            <span
              key={status}
              className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 ${STATUS_STYLE[status]}`}
            >
              <span aria-hidden>{STATUS_GLYPH[status]}</span>
              {STATUS_WORD[status]}
            </span>
          ))}
        </div>
      </div>

      {progress.groups.map((group) => (
        <GroupCard
          key={group.id}
          group={group}
          labels={progress.semester_labels}
          busy={busy}
          onFill={onFill}
        />
      ))}

      {progress.unresolved.length > 0 && (
        <div className="card border-amber-500/25 p-5">
          <p className="kicker !text-amber-400">Can&rsquo;t be checked automatically</p>
          <p className="mt-2 text-sm text-[var(--muted)]">
            Your program lists {progress.unresolved.length} more requirement
            {progress.unresolved.length === 1 ? "" : "s"} that the catalog describes in words
            rather than as a set of courses — a rule over disciplines, a list kept on another
            site, or a choice of world language. This app can&rsquo;t verify these against the
            catalog rule itself, so it shows them as the catalog writes them, with a
            best-effort guess at courses that might fit, and leaves them out of the count
            above. Grouped below by who else has the exact same requirement.
          </p>
          <div className="mt-4 space-y-5">
            {SCOPE_ORDER.map((scope) => {
              const items = progress.unresolved.filter((item) => item.scope === scope);
              if (items.length === 0) return null;
              const meta = SCOPE_META[scope];
              return (
                <div key={scope}>
                  <p className="text-xs font-semibold uppercase tracking-wide text-amber-300/90">
                    {meta.heading}
                  </p>
                  <p className="mt-0.5 text-xs text-[var(--muted)]">{meta.blurb}</p>
                  <ul className="mt-2 space-y-2">
                    {items.map((item) => (
                      <li key={item.id}>
                        <UnresolvedCard
                          item={item}
                          labels={progress.semester_labels}
                          busy={busy}
                          onFill={onFill}
                        />
                      </li>
                    ))}
                  </ul>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// A requirement stated as a credit threshold rather than a course list ("Upper Level
// Requirement": at least 32 credits from courses numbered 30000+) — genuinely checkable, so
// it gets a real progress bar, not a "can't be checked" card. Deliberately no "add" action:
// nothing needs picking here, it either already qualifies or it doesn't, and it updates the
// moment a qualifying course lands anywhere else in the plan.
function ComputedBar({ item }: { item: ComputedRequirementView }) {
  const [open, setOpen] = useState(false);
  const pct = item.credits_required > 0
    ? Math.min(100, (item.credits_filled / item.credits_required) * 100)
    : 100;

  return (
    <div className="rounded-lg border border-[var(--stroke)] bg-black/20 px-3 py-2 text-xs">
      <div className="flex items-center gap-3">
        <span
          aria-hidden
          className={`w-3 text-center font-bold ${item.satisfied ? "text-emerald-400" : "text-[var(--muted)]"}`}
        >
          {item.satisfied ? "✓" : "•"}
        </span>
        <span className="flex-1 font-semibold text-[var(--ink)]">{item.name}</span>
        <span className="whitespace-nowrap text-[var(--muted)]">
          {item.credits_filled}/{item.credits_required} cr
        </span>
        {item.contributing_courses.length > 0 && (
          <button
            onClick={() => setOpen((v) => !v)}
            className="whitespace-nowrap font-semibold text-[var(--muted)] underline-offset-2 hover:underline"
          >
            {open ? "Hide" : "What counts?"}
          </button>
        )}
      </div>
      <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-black/30">
        <div
          className={`h-full rounded-full ${item.satisfied ? "bg-emerald-400" : "bg-[var(--gold)]"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {open && (
        <ul className="mt-2 space-y-1 border-t border-[var(--stroke)] pt-2">
          {item.contributing_courses.map((course) => (
            <li key={course.code} className="flex items-center gap-2">
              <span
                className={course.status === "completed" ? "text-[var(--gold)]" : "text-[var(--ink)]"}
              >
                {course.status === "completed" ? "✓" : "•"}
              </span>
              <span className="font-semibold">{course.code}</span>
              <span className="min-w-0 flex-1 truncate text-[var(--muted)]">{course.title}</span>
              <span className="whitespace-nowrap text-[var(--muted)]">{course.credits} cr</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function UnresolvedCard({
  item,
  labels,
  busy,
  onFill,
}: {
  item: UnresolvedRequirementView;
  labels: string[];
  busy: boolean;
  onFill: (courseCode: string, semesterIndex: number) => void;
}) {
  const [open, setOpen] = useState(false);
  const text = item.raw_text ?? "";
  const long = text.length > 190;

  return (
    <div className="rounded-lg border border-amber-500/30 bg-amber-500/[0.06] px-3 py-2 text-sm">
      <div className="flex items-start gap-3">
        <span aria-hidden className="mt-0.5 w-3 text-center font-bold text-amber-300">
          ?
        </span>
        <span className="sr-only">Cannot be checked:</span>
        <span className="flex-1 font-semibold text-amber-200">{item.name}</span>
        {item.credits_min ? (
          <span className="whitespace-nowrap text-xs text-amber-200/70">
            {item.credits_min} cr
          </span>
        ) : null}
      </div>
      {text && (
        <div className="mt-1.5 pl-6">
          <p className="whitespace-pre-line text-xs leading-relaxed text-[var(--muted)]">
            {open || !long ? text : `${text.slice(0, 190)}…`}
          </p>
          {long && (
            <button
              onClick={() => setOpen((v) => !v)}
              className="mt-1 text-xs font-semibold text-amber-300/80 underline-offset-2 hover:underline"
            >
              {open ? "Show less" : "Show the full catalog text"}
            </button>
          )}
        </div>
      )}
      {item.suggested_courses.length > 0 && (
        <div className="mt-2 pl-6">
          <p className="text-xs text-amber-200/60">
            Might be a fit — guessed from course descriptions, not checked against this rule.
            Scheduling still goes through the same prerequisite/term check as everything else.
          </p>
          <ul className="mt-1.5 space-y-1.5">
            {item.suggested_courses.map((option) => (
              <li key={option.code}>
                <OptionRow option={option} labels={labels} busy={busy} onFill={onFill} />
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function GroupCard({
  group,
  labels,
  busy,
  onFill,
}: {
  group: RequirementGroupProgress;
  labels: string[];
  busy: boolean;
  onFill: (courseCode: string, semesterIndex: number) => void;
}) {
  return (
    <div className={`card p-5 ${group.satisfied ? "" : "border-red-500/20"}`}>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-display text-lg font-semibold uppercase tracking-wide text-[var(--ink)]">
          {group.satisfied && <span className="mr-2 text-emerald-400">✓</span>}
          {group.name}
        </h3>
        <span className="whitespace-nowrap text-xs text-[var(--muted)]">
          {group.credits_required
            ? `${group.credits_filled}/${group.credits_required} cr`
            : `${group.credits_filled} cr`}
          {group.selective && " · choose from list"}
        </span>
      </div>

      <ul className="mt-3 space-y-2">
        {group.slots.map((slot, index) => (
          <li key={`${group.id}-${index}`}>
            <SlotRow slot={slot} labels={labels} busy={busy} onFill={onFill} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function SlotRow({
  slot,
  labels,
  busy,
  onFill,
}: {
  slot: RequirementSlot;
  labels: string[];
  busy: boolean;
  onFill: (courseCode: string, semesterIndex: number) => void;
}) {
  const [open, setOpen] = useState(false);

  if (slot.status !== "unfilled") {
    return (
      <div
        className={`flex items-center gap-3 rounded-lg border px-3 py-2 text-sm ${STATUS_STYLE[slot.status]}`}
      >
        <span aria-hidden className="w-3 text-center font-bold">
          {STATUS_GLYPH[slot.status]}
        </span>
        <span className="sr-only">{STATUS_WORD[slot.status]}:</span>
        <span className="font-semibold">{slot.filled_by}</span>
        <span className="flex-1 truncate text-[var(--muted)]">{slot.title}</span>
        <span className="whitespace-nowrap text-xs text-[var(--muted)]">
          {slot.credits} cr
          {slot.status === "planned" && slot.semester_index != null && labels[slot.semester_index]
            ? ` · ${labels[slot.semester_index]}`
            : ""}
          {slot.status === "completed" ? " · done" : ""}
        </span>
      </div>
    );
  }

  const fillable = slot.options.filter((o) => o.legal_semesters.length > 0);
  return (
    <div className={`rounded-lg border px-3 py-2 text-sm ${STATUS_STYLE.unfilled}`}>
      <div className="flex items-center gap-3">
        <span aria-hidden className="w-3 text-center font-bold">
          {STATUS_GLYPH.unfilled}
        </span>
        <span className="sr-only">Not filled:</span>
        <span className="flex-1 font-semibold">{slot.label ?? "Not filled"}</span>
        {slot.options.length > 0 && (
          <button
            onClick={() => setOpen((v) => !v)}
            disabled={busy}
            className="whitespace-nowrap rounded-md border border-red-500/40 px-2.5 py-1 text-xs font-semibold transition hover:bg-red-500/15 disabled:opacity-40"
          >
            {open ? "Close" : `Choose (${slot.options.length})`}
          </button>
        )}
      </div>

      {slot.options.length === 0 && (
        <p className="mt-1 pl-6 text-xs text-red-300/80">
          Nothing in the catalog satisfies this yet.
        </p>
      )}

      {open && (
        <ul className="mt-2 space-y-1.5 border-t border-red-500/20 pt-2">
          {slot.options.map((option) => (
            <li key={option.code}>
              <OptionRow option={option} labels={labels} busy={busy} onFill={onFill} />
            </li>
          ))}
          {fillable.length === 0 && (
            <li className="pl-1 text-xs text-red-300/80">
              None of these fit yet — they depend on something else in this list being
              scheduled first. Fill an earlier requirement and they will open up.
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

function OptionRow({
  option,
  labels,
  busy,
  onFill,
}: {
  option: SlotOption;
  labels: string[];
  busy: boolean;
  onFill: (courseCode: string, semesterIndex: number) => void;
}) {
  // Defaults to the EARLIEST legal term — the answer most students want, and a sane default
  // beats an empty dropdown on a list that can run to seventeen options.
  const [semester, setSemester] = useState<number | null>(
    option.legal_semesters.length > 0 ? option.legal_semesters[0] : null,
  );
  const blocked = option.legal_semesters.length === 0;

  return (
    <div className="flex flex-wrap items-center gap-2 rounded-md bg-black/25 px-2.5 py-1.5">
      <span className="font-semibold text-[var(--gold)]">{option.code}</span>
      <span className="min-w-0 flex-1 truncate text-xs text-[var(--muted)]">{option.title}</span>
      <span className="whitespace-nowrap text-xs text-[var(--muted)]">{option.credits} cr</span>

      {blocked ? (
        <span
          className="whitespace-nowrap text-xs text-red-300/80"
          title="Its prerequisites aren't scheduled yet, or every term it runs in is full."
        >
          no term available yet
        </span>
      ) : (
        <>
          <select
            value={semester ?? option.legal_semesters[0]}
            disabled={busy}
            onChange={(e) => setSemester(Number(e.target.value))}
            className="rounded-md border border-[var(--stroke)] bg-black/40 px-2 py-1 text-xs text-[var(--ink)]"
          >
            {option.legal_semesters.map((index) => (
              <option key={index} value={index}>
                {labels[index] ?? `Semester ${index + 1}`}
              </option>
            ))}
          </select>
          <button
            onClick={() => onFill(option.code, semester ?? option.legal_semesters[0])}
            disabled={busy}
            className="btn-gold whitespace-nowrap !rounded-md px-2.5 py-1 text-xs disabled:opacity-40"
          >
            Add
          </button>
        </>
      )}
    </div>
  );
}
