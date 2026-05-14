import CoursePill from "./CoursePill";

type Course = {
  code: string;
  name: string;
};

type Props = {
  title: string;
  credits: number;
  warning: string;
  courses: Course[];
};

export default function SemesterCard({
  title,
  credits,
  warning,
  courses,
}: Props) {
  return (
    <div className="card p-6 text-center">
      <div className="mb-4 flex flex-col items-center gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.15em] text-orange-700">
            Semester Plan
          </p>
          <h2 className="text-2xl font-bold text-stone-900">{title}</h2>
          <p className="text-sm text-stone-600">{credits} credits</p>
        </div>

        {warning ? (
          <div className="rounded-full border border-amber-700/30 bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-900">
            {warning}
          </div>
        ) : (
          <div className="rounded-full border border-emerald-700/25 bg-emerald-100 px-3 py-1 text-xs font-semibold text-emerald-900">
            On track
          </div>
        )}
      </div>

      <div className="space-y-3">
        {courses.map((course) => (
          <CoursePill
            key={course.code}
            code={course.code}
            name={course.name}
          />
        ))}
      </div>
    </div>
  );
}
