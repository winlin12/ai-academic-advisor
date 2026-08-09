"use client";

import { StudentProfile } from "@/lib/api";

type Props = {
  profile: StudentProfile;
  // Whether the plan on screen came from the model or was built empty for manual filling —
  // shown here because it's otherwise easy to lose track of mid-session, especially after
  // navigating away from the Requirements tab where it's obvious.
  aiMode: boolean;
  busy?: boolean;
  onEditProfile: () => void;
  // Discards the student entirely and returns to onboarding. NOT the plan-level "Start over",
  // which resamples the schedule and keeps the person — the two used to share a name and one
  // of them throws away everything the student typed.
  onNewProfile: () => void;
  // Downloads the profile + plan as a JSON file — the only backup that exists, now that
  // nothing is kept server-side. See lib/api's `downloadProfileFile`.
  onExportProfile: () => void;
  // Replaces the active profile with one loaded from a previously exported file.
  onImportProfile: (file: File) => void;
};

export default function StudentProfilePanel({
  profile,
  aiMode,
  busy = false,
  onEditProfile,
  onNewProfile,
  onExportProfile,
  onImportProfile,
}: Props) {
  const targetGraduation = profile.preferences?.target_graduation;

  const fields: Array<{ label: string; value: string }> = [
    // Omitted entirely when blank, rather than showing an empty row — most students will
    // never set this, and there is no default to fall back to.
    ...(profile.profile_label ? [{ label: "Profile name", value: profile.profile_label }] : []),
    { label: "Mode", value: aiMode ? "AI-assisted" : "Manual (no AI)" },
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
    // Just the subject code, not the display name ("Spanish") — that mapping only exists
    // where the program's catalog is loaded (ProfileSetup, mid-edit), and this panel is a
    // cheap, synchronous summary with no API access of its own. The code is truthful and
    // unambiguous even if less friendly; "Edit profile" shows the friendly picker.
    ...(profile.language_of_interest
      ? [{
          label: "Language",
          value: profile.language_proficiency
            ? `${profile.language_of_interest} (level ${profile.language_proficiency})`
            : profile.language_of_interest,
        }]
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
          onClick={onEditProfile}
          disabled={busy}
          className="btn-ghost w-full px-4 py-2.5 text-sm"
        >
          Edit profile
        </button>
        <button
          onClick={onExportProfile}
          disabled={busy}
          className="btn-ghost w-full px-4 py-2.5 text-sm"
        >
          Save profile to file
        </button>
        <label className="btn-ghost flex w-full cursor-pointer justify-center px-4 py-2.5 text-sm">
          Load profile from file
          <input
            type="file"
            accept="application/json"
            className="hidden"
            disabled={busy}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) onImportProfile(file);
              event.target.value = "";
            }}
          />
        </label>
        <button
          onClick={onNewProfile}
          disabled={busy}
          className="w-full rounded-lg px-4 py-2 text-xs font-medium text-[var(--muted)] transition hover:text-red-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Discard profile
        </button>
      </div>
      <p className="mt-3 text-xs leading-relaxed text-[var(--muted)]">
        Nothing here is sent to or stored on a server — this profile lives in this browser
        and in whatever file you save.
      </p>
    </div>
  );
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
