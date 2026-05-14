"use client";

import { useEffect, useState } from "react";
import SemesterCard from "@/components/SemesterCard";
import StudentProfilePanel from "@/components/StudentProfilePanel";
import AdvisorChat from "@/components/AdvisorChat";
import { generatePlan, PlanResponse, StudentProfile } from "@/lib/api";

const demoProfile: StudentProfile = {
  name: "Past Winston",
  degree_program: "MS Computer Science",
  completed_courses: ["CS251"],
  remaining_courses: [
    "CS381",
    "CS502",
    "CS536",
    "CS541",
    "CS551",
    "CS577",
    "CS587",
  ],
  start_term: "fall",
  start_year: 2026,
  semesters_to_plan: 5,
  max_credits_per_semester: 6,
  preferences: {
    avoid_too_many_project_heavy_courses: true,
    target_graduation: "Spring 2028",
  },
};

export default function HomePage() {
  const [plan, setPlan] = useState<PlanResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadPlan() {
    try {
      setLoading(true);
      setError(null);
      const generatedPlan = await generatePlan(demoProfile);
      setPlan(generatedPlan);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadPlan();
  }, []);

  return (
    <main className="min-h-screen px-4 py-8 text-center md:px-8">
      <div className="mx-auto max-w-7xl">
        <div className="mb-8 rounded-3xl border border-orange-900/10 bg-white/70 p-7 shadow-[0_16px_50px_rgba(120,53,15,0.16)] backdrop-blur">
          <p className="text-sm font-semibold uppercase tracking-[0.2em] text-orange-700">
            Academic Planning Studio
          </p>
          <h1 className="mt-2 text-4xl font-bold leading-tight text-stone-900 md:text-5xl">
            AI Academic Advisor
          </h1>
          <p className="mt-3 max-w-2xl text-stone-700">
            Build semester-by-semester plans, inspect blockers, and ask targeted
            questions about your degree trajectory.
          </p>
        </div>
      </div>

      {loading && (
        <div className="card p-6 text-stone-700">Generating academic plan...</div>
      )}

      {error && (
        <div className="rounded-2xl border border-red-400/40 bg-red-100/70 p-6 text-red-900">
          {error}
        </div>
      )}

      {plan && (
        <div className="mx-auto grid max-w-7xl grid-cols-1 gap-6 lg:grid-cols-12">
          <div className="lg:col-span-3">
            <div className="lg:sticky lg:top-8">
              <StudentProfilePanel
                profile={{
                  name: plan.student_name,
                  degree: plan.degree_program,
                  targetGraduation:
                    String(demoProfile.preferences.target_graduation) ??
                    "Not set",
                }}
                onRegenerate={loadPlan}
              />
            </div>
          </div>

          <div className="space-y-6 text-center lg:col-span-6">
            {plan.semesters.map((semester) => (
              <SemesterCard
                key={`${semester.term}-${semester.year}`}
                title={`${capitalize(semester.term)} ${semester.year}`}
                credits={semester.total_credits}
                warning={semester.warnings[0] ?? ""}
                courses={semester.courses.map((course) => ({
                  code: course.code,
                  name: course.title,
                }))}
              />
            ))}

            {plan.warnings.length > 0 && (
              <div className="rounded-2xl border border-amber-700/30 bg-amber-100/70 p-6 text-amber-900 shadow-[0_10px_25px_rgba(120,53,15,0.12)]">
                <h2 className="mb-3 text-xl font-bold">Plan Warnings</h2>
                <ul className="space-y-2 text-sm leading-relaxed">
                  {plan.warnings.map((warning) => (
                    <li key={warning}>• {warning}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <div className="lg:col-span-3">
            <div className="lg:sticky lg:top-8">
              <AdvisorChat plan={plan} />
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
