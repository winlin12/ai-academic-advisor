"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  AcademicFacets,
  AcademicProgramSummary,
  ApiError,
  fetchAcademicFacets,
  searchPrograms,
} from "@/lib/api";

const DEBOUNCE_MS = 300;

// The "look up a college/department/degree/major like a search" surface (TODO Priority 3
// step 1): pure catalog reference, no student context. Colleges/majors/degrees all live in
// the same `programs` table (a major is a program; its college is `program.school`), so one
// search+filter view covers all of it — there's no separate college/department entity in the
// data yet (see the note below), but the school filter is wired up and will fill in the moment
// that data exists.
export default function ProgramsPage() {
  const [facets, setFacets] = useState<AcademicFacets | null>(null);
  const [query, setQuery] = useState("");
  const [catalogYear, setCatalogYear] = useState<number | "">("");
  const [school, setSchool] = useState("");
  const [results, setResults] = useState<AcademicProgramSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const requestSeq = useRef(0);

  useEffect(() => {
    fetchAcademicFacets()
      .then(setFacets)
      .catch(() => setFacets({ catalog_years: [], schools: [], subjects: [] }));
  }, []);

  useEffect(() => {
    const timer = setTimeout(async () => {
      const seq = ++requestSeq.current;
      try {
        setLoading(true);
        setError(null);
        const programs = await searchPrograms({
          query,
          catalogYear: catalogYear || undefined,
          school: school || undefined,
          limit: 100,
        });
        if (seq === requestSeq.current) setResults(programs);
      } catch (err) {
        if (seq === requestSeq.current) {
          setError(
            err instanceof ApiError
              ? err.message
              : "Failed to load the degree catalog — is the database reachable?",
          );
          setResults([]);
        }
      } finally {
        if (seq === requestSeq.current) setLoading(false);
      }
    }, DEBOUNCE_MS);

    return () => clearTimeout(timer);
  }, [query, catalogYear, school]);

  return (
    <main className="mx-auto max-w-7xl px-4 py-10 md:px-8">
      <div className="mb-8">
        <p className="kicker">Colleges · Departments · Degrees · Majors</p>
        <h1 className="font-display mt-1 text-4xl font-bold uppercase tracking-wide text-[var(--ink)] md:text-5xl">
          Degree <span className="text-[var(--gold)]">Catalog</span>
        </h1>
        <p className="mt-3 max-w-2xl text-sm text-[var(--muted)]">
          Search every degree and major in the catalog and open one to see its full
          requirement breakdown. Programs are added continuously as the catalog crawl runs —
          if your major isn&apos;t here yet, check back soon, or enter it manually during
          profile setup.
        </p>
      </div>

      <div className="card card-accent mb-6 grid grid-cols-1 gap-4 p-5 md:grid-cols-3">
        <label className="block md:col-span-1">
          <span className="mb-1 block text-sm font-semibold">Search</span>
          <input
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Degree, major, or school name…"
            className="field text-sm"
          />
        </label>

        <label className="block">
          <span className="mb-1 block text-sm font-semibold">Catalog year</span>
          <select
            value={catalogYear}
            onChange={(event) =>
              setCatalogYear(event.target.value ? Number(event.target.value) : "")
            }
            className="field text-sm"
          >
            <option value="">Any year</option>
            {facets?.catalog_years.map((year) => (
              <option key={year} value={year}>
                {year}–{year + 1}
              </option>
            ))}
          </select>
        </label>

        <label className="block">
          <span className="mb-1 block text-sm font-semibold">
            College{" "}
            {facets && facets.schools.length === 0 && (
              <span className="font-normal text-[var(--muted)]">(not on file yet)</span>
            )}
          </span>
          <select
            value={school}
            disabled={!facets || facets.schools.length === 0}
            onChange={(event) => setSchool(event.target.value)}
            className="field text-sm"
          >
            <option value="">Any college</option>
            {facets?.schools.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
      </div>

      {error && (
        <div className="mb-6 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
          {error}
        </div>
      )}

      {loading && results.length === 0 && !error && (
        <p className="text-sm text-[var(--muted)]">Searching…</p>
      )}

      {!loading && !error && results.length === 0 && (
        <div className="card p-8 text-center text-sm text-[var(--muted)]">
          No programs matched yet. The catalog crawl is still filling in — try a broader
          search, or a different catalog year.
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {results.map((program) => (
          <Link
            key={program.id}
            href={`/programs/${program.id}`}
            className="card block p-5 transition hover:border-[var(--gold)]"
          >
            <p className="kicker">{program.degree_code ?? "Program"}</p>
            <h2 className="font-display mt-1 text-xl font-semibold uppercase tracking-wide text-[var(--ink)]">
              {program.program_title}
            </h2>
            <p className="mt-1 text-xs text-[var(--muted)]">
              {program.school ?? "College not on file yet"} · {program.catalog_year}–
              {program.catalog_year + 1}
            </p>
            <p className="mt-3 text-xs text-[var(--muted)]">
              {program.block_count} requirement block{program.block_count === 1 ? "" : "s"} ·{" "}
              {program.course_count} course option{program.course_count === 1 ? "" : "s"}
            </p>
          </Link>
        ))}
      </div>
    </main>
  );
}
