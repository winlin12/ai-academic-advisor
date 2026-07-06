"use client";

import { useEffect, useRef, useState } from "react";
import { AcademicCourseResult, searchCourses } from "@/lib/api";

type Props = {
  label: string;
  hint?: string;
  codes: string[];
  onChange: (codes: string[]) => void;
  disabled?: boolean;
};

const DEBOUNCE_MS = 300;
const MIN_QUERY_LENGTH = 2;

// Chip-list editor for a set of course codes, fed by the same debounced catalog search
// the semester cards use. Enter adds the raw typed code verbatim, so profiles can still
// be built when the catalog DB (and therefore search) is down.
export default function CourseChipInput({
  label,
  hint,
  codes,
  onChange,
  disabled = false,
}: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<AcademicCourseResult[]>([]);
  const [searching, setSearching] = useState(false);
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
        const courses = await searchCourses(trimmed, 6);
        if (seq === requestSeq.current) setResults(courses);
      } catch {
        // Search being down isn't fatal here — Enter still adds raw codes.
        if (seq === requestSeq.current) setResults([]);
      } finally {
        if (seq === requestSeq.current) setSearching(false);
      }
    }, DEBOUNCE_MS);

    return () => clearTimeout(timer);
  }, [query]);

  function add(code: string) {
    const normalized = code.replace(/\s+/g, "").toUpperCase();
    if (!normalized) return;
    if (!codes.includes(normalized)) onChange([...codes, normalized]);
    setQuery("");
    setResults([]);
  }

  function remove(code: string) {
    onChange(codes.filter((existing) => existing !== code));
  }

  return (
    <div>
      <p className="mb-1 text-sm font-semibold text-[var(--ink)]">{label}</p>
      {hint && <p className="mb-2 text-xs text-[var(--muted)]">{hint}</p>}

      {codes.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-2">
          {codes.map((code) => (
            <span
              key={code}
              className="inline-flex items-center gap-1.5 rounded-full border border-[var(--stroke-strong)] bg-[rgba(207,185,145,0.1)] px-3 py-1 text-xs font-semibold text-[var(--gold)]"
            >
              {code}
              <button
                type="button"
                disabled={disabled}
                onClick={() => remove(code)}
                aria-label={`Remove ${code}`}
                className="text-[var(--muted)] transition hover:text-red-400 disabled:cursor-not-allowed"
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}

      <input
        type="text"
        value={query}
        disabled={disabled}
        onChange={(event) => setQuery(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            add(results[0]?.code ?? query);
          }
        }}
        placeholder="Search the catalog, or type a code and press Enter…"
        className="field text-sm"
      />

      {searching && (
        <p className="mt-1 text-xs text-[var(--muted)]">Searching catalog…</p>
      )}

      {results.length > 0 && (
        <ul className="mt-2 space-y-1">
          {results.map((course) => (
            <li key={course.id}>
              <button
                type="button"
                disabled={disabled}
                onClick={() => add(course.code)}
                className="w-full rounded-lg border border-[var(--stroke)] bg-black/30 px-3 py-2 text-left text-xs transition hover:border-[var(--gold)] hover:bg-[rgba(207,185,145,0.08)] disabled:cursor-not-allowed"
              >
                <span className="font-semibold text-[var(--gold)]">
                  {course.code}
                </span>{" "}
                <span className="text-[var(--ink)]">{course.title}</span>
                {course.credit_hours !== null && (
                  <span className="text-[var(--muted)]">
                    {" "}
                    · {course.credit_hours} cr
                  </span>
                )}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
