"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import RequirementTree from "@/components/RequirementTree";
import { AcademicProgramDetail, ApiError, fetchProgramDetail } from "@/lib/api";

// The plain catalog reference view for one degree — "pull a degree up and see its
// requirements," no student context required (TODO Priority 3 step 1). The personalized,
// progress-annotated version of this same tree lives on the planner page once a student has
// linked this program to their profile (see the "Degree Requirements" panel in app/page.tsx).
export default function ProgramDetailPage() {
  const params = useParams<{ id: string }>();
  const programId = params.id;

  const [program, setProgram] = useState<AcademicProgramDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchProgramDetail(programId)
      .then((detail) => {
        if (!cancelled) setProgram(detail);
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 404) {
          setError("This program couldn't be found — it may belong to an older catalog year.");
        } else {
          setError(err instanceof Error ? err.message : "Failed to load this program.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [programId]);

  return (
    <main className="mx-auto max-w-4xl px-4 py-10 md:px-8">
      <Link
        href="/programs"
        className="mb-6 inline-block text-sm font-semibold text-[var(--muted)] hover:text-[var(--gold)]"
      >
        ← Back to degree catalog
      </Link>

      {loading && <p className="text-sm text-[var(--muted)]">Loading program…</p>}

      {error && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
          {error}
        </div>
      )}

      {program && (
        <>
          <div className="mb-8">
            <p className="kicker">{program.degree_code ?? "Program"}</p>
            <h1 className="font-display mt-1 text-4xl font-bold uppercase tracking-wide text-[var(--ink)] md:text-5xl">
              {program.program_title}
            </h1>
            <p className="mt-2 text-sm text-[var(--muted)]">
              {program.school ?? "College not on file yet"} · Catalog {program.catalog_year}–
              {program.catalog_year + 1}
              {program.variant ? ` · ${program.variant}` : ""}
            </p>
            {program.source_url && (
              <a
                href={program.source_url}
                target="_blank"
                rel="noreferrer"
                className="mt-2 inline-block text-xs font-semibold text-[var(--gold)] hover:underline"
              >
                View on Purdue&apos;s official catalog ↗
              </a>
            )}
          </div>

          <div className="mb-6 rounded-xl border border-[var(--stroke)] bg-black/20 p-4 text-sm text-[var(--muted)]">
            Planning to pursue this degree?{" "}
            <Link href="/" className="font-semibold text-[var(--gold)] hover:underline">
              Set it up in the planner
            </Link>{" "}
            to generate a semester-by-semester plan and track your progress against every
            requirement below.
          </div>

          <RequirementTree blocks={program.blocks} />
        </>
      )}
    </main>
  );
}
