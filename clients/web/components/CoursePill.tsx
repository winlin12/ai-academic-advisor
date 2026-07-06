"use client";

export type SemesterOption = {
  value: number; // zero-based index into plan.semesters — what POST /v1/plan/edit expects
  label: string; // "Fall 2026"
};

type Props = {
  code: string;
  name: string;
  // Edit affordances are optional: omit them and the pill renders read-only, exactly as
  // before (used anywhere a plan is displayed but not editable).
  semesterIndex?: number;
  semesterOptions?: SemesterOption[];
  disabled?: boolean;
  onMove?: (targetSemester: number) => void;
  onRemove?: () => void;
};

export default function CoursePill({
  code,
  name,
  semesterIndex,
  semesterOptions,
  disabled = false,
  onMove,
  onRemove,
}: Props) {
  const canMove =
    onMove !== undefined &&
    semesterOptions !== undefined &&
    semesterIndex !== undefined;

  return (
    <div className="group rounded-xl border border-[var(--stroke)] bg-black/30 p-4 transition hover:border-[var(--stroke-strong)]">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="font-display text-lg font-semibold uppercase tracking-wide text-[var(--gold)]">
            {code}
          </p>
          <p className="text-sm text-[var(--muted)] group-hover:text-[var(--ink)]">
            {name}
          </p>
        </div>

        {onRemove && (
          <button
            type="button"
            onClick={onRemove}
            disabled={disabled}
            aria-label={`Remove ${code} from this plan`}
            title="Remove from plan"
            className="rounded-full border border-red-500/20 bg-red-500/10 px-2 text-sm font-semibold leading-6 text-red-400 transition hover:bg-red-500/20 disabled:cursor-not-allowed disabled:opacity-40"
          >
            ×
          </button>
        )}
      </div>

      {canMove && (
        <label className="mt-3 flex items-center gap-2 text-xs text-[var(--muted)]">
          <span className="font-semibold uppercase tracking-wide">Move to</span>
          <select
            value={semesterIndex}
            disabled={disabled}
            aria-label={`Move ${code} to another semester`}
            onChange={(event) => {
              const target = Number(event.target.value);
              if (target !== semesterIndex) onMove(target);
            }}
            className="field flex-1 !py-1 text-xs"
          >
            {semesterOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      )}
    </div>
  );
}
