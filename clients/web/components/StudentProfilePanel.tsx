"use client";

import { StudentProfile } from "@/lib/api";

type Props = {
  profile: StudentProfile;
  busy?: boolean;
  onRegenerate: () => void;
  onEditProfile: () => void;
  onStartOver: () => void;
};

export default function StudentProfilePanel({
  profile,
  busy = false,
  onRegenerate,
  onEditProfile,
  onStartOver,
}: Props) {
  const targetGraduation = profile.preferences?.target_graduation;

  const fields: Array<{ label: string; value: string }> = [
    { label: "Name", value: profile.name },
    { label: "Degree", value: profile.degree_program },
    {
      label: "Starts",
      value: `${capitalize(profile.start_term)} ${profile.start_year}`,
    },
    {
      label: "Credit cap",
      value: `${profile.max_credits_per_semester} / semester`,
    },
    {
      label: "Completed",
      value: `${profile.completed_courses.length} course${
        profile.completed_courses.length === 1 ? "" : "s"
      }`,
    },
    ...(targetGraduation
      ? [{ label: "Target graduation", value: String(targetGraduation) }]
      : []),
  ];

  return (
    <div className="card card-accent p-6">
      <p className="kicker">Student</p>
      <h2 className="font-display mt-1 text-2xl font-semibold uppercase tracking-wide text-[var(--ink)]">
        Profile
      </h2>

      <dl className="mt-5 space-y-3">
        {fields.map((field) => (
          <div
            key={field.label}
            className="flex items-baseline justify-between gap-3 border-b border-[var(--stroke)] pb-2 last:border-b-0"
          >
            <dt className="text-xs uppercase tracking-wide text-[var(--muted)]">
              {field.label}
            </dt>
            <dd className="text-right text-sm font-semibold text-[var(--ink)]">
              {field.value}
            </dd>
          </div>
        ))}
      </dl>

      <div className="mt-6 space-y-2">
        <button
          onClick={onRegenerate}
          disabled={busy}
          className="btn-gold w-full px-4 py-2.5 text-sm"
        >
          Regenerate plan
        </button>
        <button
          onClick={onEditProfile}
          disabled={busy}
          className="btn-ghost w-full px-4 py-2.5 text-sm"
        >
          Edit profile
        </button>
        <button
          onClick={onStartOver}
          disabled={busy}
          className="w-full rounded-lg px-4 py-2 text-xs font-medium text-[var(--muted)] transition hover:text-red-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Start over
        </button>
      </div>
    </div>
  );
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
