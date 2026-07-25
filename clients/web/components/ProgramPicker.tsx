"use client";

import { useEffect, useRef, useState } from "react";
import { AcademicProgramSummary, searchPrograms } from "@/lib/api";

type Props = {
  label: string;
  hint?: string;
  programId: string | null;
  // The human-readable degree name — doubles as the free-text fallback value when no
  // program_id is set, and as the display label once one is picked.
  degreeText: string;
  onChange: (programId: string | null, degreeText: string) => void;
  disabled?: boolean;
};

const DEBOUNCE_MS = 300;
const MIN_QUERY_LENGTH = 2;

function formatProgramLabel(program: AcademicProgramSummary): string {
  const parts = [program.program_title];
  if (program.degree_code) parts.push(`(${program.degree_code})`);
  return parts.join(" ");
}

// Search-and-select for a real degree program (sets `program_id`, which is what actually
// unlocks program-driven planning and the degree audit view), with a manual free-text
// fallback — the program catalog is populated by a slow, ongoing crawl (TODO.md), so onboarding
// must never dead-end just because a given major hasn't been crawled yet.
export default function ProgramPicker({
  label,
  hint,
  programId,
  degreeText,
  onChange,
  disabled = false,
}: Props) {
  const [manualMode, setManualMode] = useState(!programId && Boolean(degreeText));
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<AcademicProgramSummary[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState(false);
  const requestSeq = useRef(0);

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < MIN_QUERY_LENGTH) {
      setResults([]);
      setSearching(false);
      return;
    }

    const timer = setTimeout(async () => {
      const seq = ++requestSeq.current;
      try {
        setSearching(true);
        setSearchError(false);
        const programs = await searchPrograms({ query: trimmed, limit: 15 });
        if (seq === requestSeq.current) setResults(programs);
      } catch {
        if (seq === requestSeq.current) {
          setResults([]);
          setSearchError(true);
        }
      } finally {
        if (seq === requestSeq.current) setSearching(false);
      }
    }, DEBOUNCE_MS);

    return () => clearTimeout(timer);
  }, [query]);

  function select(program: AcademicProgramSummary) {
    onChange(program.id, formatProgramLabel(program));
    setQuery("");
    setResults([]);
  }

  function clearSelection() {
    onChange(null, "");
    setManualMode(false);
  }

  return (
    <div>
      <span className="mb-1 block text-sm font-semibold">{label}</span>
      {hint && <p className="mb-2 text-xs text-[var(--muted)]">{hint}</p>}

      {programId ? (
        <div className="flex items-center justify-between gap-3 rounded-lg border border-[var(--stroke-strong)] bg-[rgba(207,185,145,0.08)] px-3 py-2.5">
          <span className="text-sm font-semibold text-[var(--gold)]">{degreeText}</span>
          <button
            type="button"
            disabled={disabled}
            onClick={clearSelection}
            className="whitespace-nowrap text-xs font-semibold text-[var(--muted)] underline-offset-2 hover:text-[var(--ink)] hover:underline disabled:cursor-not-allowed"
          >
            Change
          </button>
        </div>
      ) : manualMode ? (
        <div>
          <input
            type="text"
            value={degreeText}
            disabled={disabled}
            onChange={(event) => onChange(null, event.target.value)}
            placeholder="e.g. MS Computer Science"
            className="field text-sm"
          />
          <button
            type="button"
            disabled={disabled}
            onClick={() => setManualMode(false)}
            className="mt-1.5 text-xs font-semibold text-[var(--muted)] underline-offset-2 hover:text-[var(--ink)] hover:underline disabled:cursor-not-allowed"
          >
            Search the degree catalog instead
          </button>
        </div>
      ) : (
        <div>
          <input
            type="text"
            value={query}
            disabled={disabled}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search degrees, e.g. &ldquo;Computer Science&rdquo;…"
            className="field text-sm"
          />

          {searching && <p className="mt-1 text-xs text-[var(--muted)]">Searching…</p>}
          {searchError && (
            <p className="mt-1 text-xs text-amber-300">
              Couldn&apos;t reach the degree catalog — you can still enter it manually below.
            </p>
          )}

          {results.length > 0 && (
            <ul className="mt-2 max-h-60 space-y-1 overflow-y-auto scroll-dark">
              {results.map((program) => (
                <li key={program.id}>
                  <button
                    type="button"
                    disabled={disabled}
                    onClick={() => select(program)}
                    className="w-full rounded-lg border border-[var(--stroke)] bg-black/30 px-3 py-2 text-left text-xs transition hover:border-[var(--gold)] hover:bg-[rgba(207,185,145,0.08)] disabled:cursor-not-allowed"
                  >
                    <span className="font-semibold text-[var(--gold)]">
                      {program.program_title}
                    </span>{" "}
                    {program.degree_code && (
                      <span className="text-[var(--ink)]">{program.degree_code}</span>
                    )}
                    {program.school && (
                      <span className="text-[var(--muted)]"> · {program.school}</span>
                    )}
                    <span className="text-[var(--muted)]"> · {program.catalog_year}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}

          <button
            type="button"
            disabled={disabled}
            onClick={() => setManualMode(true)}
            className="mt-1.5 text-xs font-semibold text-[var(--muted)] underline-offset-2 hover:text-[var(--ink)] hover:underline disabled:cursor-not-allowed"
          >
            Can&apos;t find it? Enter it manually
          </button>
        </div>
      )}
    </div>
  );
}
