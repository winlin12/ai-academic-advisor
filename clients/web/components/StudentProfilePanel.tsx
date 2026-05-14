type Props = {
  profile: {
    name: string;
    degree: string;
    targetGraduation: string;
  };
  onRegenerate?: () => void;
};

export default function StudentProfilePanel({ profile, onRegenerate }: Props) {
  return (
    <div className="card p-6 text-center">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-orange-700">
        Student
      </p>
      <h2 className="mb-5 mt-2 text-2xl font-bold text-stone-900">Profile</h2>

      <div className="space-y-4">
        <div className="rounded-xl border border-orange-900/10 bg-white/70 p-3">
          <p className="text-xs uppercase tracking-wide text-stone-500">Name</p>
          <p className="mt-1 font-semibold text-stone-900">{profile.name}</p>
        </div>

        <div className="rounded-xl border border-orange-900/10 bg-white/70 p-3">
          <p className="text-xs uppercase tracking-wide text-stone-500">Degree</p>
          <p className="mt-1 font-semibold text-stone-900">{profile.degree}</p>
        </div>

        <div className="rounded-xl border border-orange-900/10 bg-white/70 p-3">
          <p className="text-xs uppercase tracking-wide text-stone-500">Target Graduation</p>
          <p className="mt-1 font-semibold text-stone-900">{profile.targetGraduation}</p>
        </div>
      </div>

      <button
        onClick={onRegenerate}
        className="mt-6 w-full rounded-xl bg-gradient-to-r from-orange-600 to-orange-500 px-4 py-3 font-semibold text-white shadow-[0_10px_20px_rgba(234,88,12,0.3)] transition hover:brightness-110"
      >
        Regenerate Plan
      </button>
    </div>
  );
}
