"use client";

import { useEffect, useRef, useState } from "react";
import { AcademicCourseResult, searchCourses } from "@/lib/api";

type Props = {
  onAdd: (courseCode: string) => void;
  disabled?: boolean;
};

const DEBOUNCE_MS = 300;
const MIN_QUERY_LENGTH = 2;

// Debounced search over GET /v1/academic/courses/search so a student can hand-add a course
// to a semester. Debouncing matters doubly on a local-first stack: every keystroke saved is
// a Postgres ILIKE query the laptop doesn't run while Ollama may be holding the RAM.
export default function AddCourseSearch({ onAdd, disabled = false }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<AcademicCourseResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Monotonic request id: a slow response that lands after a newer query fired is dropped
  // instead of clobbering the fresher results.
  const requestSeq = useRef(0);

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < MIN_QUERY_LENGTH) {
      setResults([]);
      setError(null);
      setSearching(false);
      return;
    }

    const timer = setTimeout(async () => {
      const seq = ++requestSeq.current;
      try {
        setSearching(true);
        setError(null);
        const courses = await searchCourses(trimmed, 6);
        if (seq === requestSeq.current) setResults(courses);
      } catch (err) {
        if (seq === requestSeq.current) {
          setResults([]);
          setError(err instanceof Error ? err.message : "Search failed");
        }
      } finally {
        if (seq === requestSeq.current) setSearching(false);
      }
    }, DEBOUNCE_MS);

    return () => clearTimeout(timer);
  }, [query]);

  function handleAdd(course: AcademicCourseResult) {
    setQuery("");
    setResults([]);
    onAdd(course.code);
  }

  return (
    <div>
      <input
        type="search"
        value={query}
        disabled={disabled}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Add a course (search title or code)…"
        aria-label="Search the catalog to add a course to this semester"
        className="field text-sm"
      />

      {searching && (
        <p className="mt-2 text-xs text-[var(--muted)]">Searching catalog…</p>
      )}

      {error && <p className="mt-2 text-xs text-red-400">{error}</p>}

      {results.length > 0 && (
        <ul className="mt-2 space-y-1">
          {results.map((course) => (
            <li key={course.id}>
              <button
                type="button"
                disabled={disabled}
                onClick={() => handleAdd(course)}
                className="w-full rounded-lg border border-[var(--stroke)] bg-black/30 px-3 py-2 text-left text-xs transition hover:border-[var(--gold)] hover:bg-[rgba(207,185,145,0.08)] disabled:cursor-not-allowed disabled:opacity-40"
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
