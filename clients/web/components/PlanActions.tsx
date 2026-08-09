"use client";

import type { RefineMode } from "@/lib/api";

type Props = {
  // Fill has nothing to do on a plan that already covers the degree and schedules everything.
  // Saying so up front beats spending thirty seconds to be told the same by a model.
  canFill: boolean;
  busy: boolean;
  running: RefineMode | null;
  onRefine: (mode: RefineMode) => void;
};

// MODE C, as three buttons. They are not three names for "try again" — they are the three
// policies `model_eval/harness/convergence.py` measured separately and never averaged, because
// they answer different questions about a plan you have just read:
//
//   Fill        this is close, finish it        freeze what validates, model fills the gaps
//   Regenerate  this has problems, fix them     model is told what is wrong and must repair it
//   Start over  this isn't what I wanted        fresh sample, told nothing about this one
//
// The distinction is worth surfacing because it changes what the student KEEPS. Fill cannot
// disturb a semester they were happy with; the other two can rewrite the whole plan. Anyone
// who has watched a chatbot destroy a good answer while "improving" it knows why that matters.
const ACTIONS: Array<{
  mode: RefineMode;
  label: string;
  running: string;
  blurb: string;
  primary?: boolean;
}> = [
  {
    mode: "fill",
    label: "Fill the gaps",
    running: "Filling…",
    blurb:
      "Keeps every semester that checks out and asks the model only for what is still missing.",
    primary: true,
  },
  {
    mode: "regenerate",
    label: "Regenerate",
    running: "Regenerating…",
    blurb:
      "Tells the model exactly what is wrong with this plan and makes it fix it. Can move anything.",
  },
  {
    mode: "start-over",
    label: "Start over",
    running: "Starting over…",
    blurb: "A completely fresh plan, with no knowledge of this one.",
  },
];

export default function PlanActions({ canFill, busy, running, onRefine }: Props) {
  return (
    <div className="card p-5">
      <p className="kicker">Not happy with this plan?</p>
      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
        {ACTIONS.map((action) => {
          const disabled = busy || (action.mode === "fill" && !canFill);
          const isRunning = running === action.mode;
          return (
            <div key={action.mode}>
              <button
                onClick={() => onRefine(action.mode)}
                disabled={disabled}
                title={
                  action.mode === "fill" && !canFill
                    ? "Nothing to fill — every requirement this app can check is covered."
                    : action.blurb
                }
                className={
                  (action.primary
                    ? "btn-gold w-full px-4 py-2.5 text-sm"
                    : "btn-ghost w-full px-4 py-2.5 text-sm") +
                  (disabled ? " cursor-not-allowed opacity-40" : "")
                }
              >
                {isRunning ? action.running : action.label}
              </button>
              <p className="mt-1.5 text-xs leading-snug text-[var(--muted)]">
                {action.mode === "fill" && !canFill
                  ? "Every checkable requirement is covered. Anything left needs an advisor."
                  : action.blurb}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
