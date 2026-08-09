"use client";

import { useEffect, useRef, useState } from "react";
import { fetchModelStatus, ModelStatus, selectModel } from "@/lib/api";

// A GLOBAL setting, not a per-profile one — which local model answers chat and writes plans
// is a property of the running server, so it lives in the nav bar rather than the student
// profile panel. Switching is a real 30-90s wait (the backend stops and relaunches
// llama-server — see backend/app/services/model_manager.py, only one model fits the box's
// VRAM at a time), which is why this shows a "Switching…" state rather than pretending it's
// instant.
export default function ModelPicker() {
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [open, setOpen] = useState(false);
  const [switching, setSwitching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchModelStatus()
      .then(setStatus)
      .catch(() => setStatus(null));
  }, []);

  useEffect(() => {
    function onClickOutside(event: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  async function handleSelect(name: string) {
    if (!status || name === status.current || switching) return;
    setOpen(false);
    setSwitching(true);
    setError(null);
    try {
      const next = await selectModel(name);
      setStatus(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to switch models");
      // The backend's own status is the source of truth after a failed switch (it may have
      // fallen back to "no model running" rather than staying on the old one) — refetch
      // rather than assume.
      fetchModelStatus().then(setStatus).catch(() => {});
    } finally {
      setSwitching(false);
    }
  }

  if (!status) {
    return null;
  }

  const current = status.available.find((m) => m.name === status.current);
  const label = switching
    ? "Switching…"
    : current
      ? current.label
      : "No model running";

  return (
    <div ref={rootRef} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        disabled={switching}
        title={error ?? undefined}
        className={
          "flex items-center gap-1.5 rounded-lg border px-3 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-70 " +
          (error
            ? "border-red-500/30 text-red-300"
            : "border-[var(--stroke)] text-[var(--muted)] hover:text-[var(--ink)]")
        }
      >
        {switching && (
          <span
            aria-hidden
            className="h-3 w-3 animate-spin rounded-full border-2 border-[var(--gold)] border-t-transparent"
          />
        )}
        <span className="max-w-[10rem] truncate">{label}</span>
        <span aria-hidden className="text-xs">▾</span>
      </button>

      {open && (
        <div className="absolute right-0 z-30 mt-2 w-72 rounded-xl border border-[var(--stroke)] bg-black/95 p-2 shadow-xl backdrop-blur">
          <p className="px-2 pb-1.5 pt-1 text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">
            Local model
          </p>
          <ul className="space-y-1">
            {status.available.map((option) => {
              const active = option.name === status.current;
              return (
                <li key={option.name}>
                  <button
                    onClick={() => handleSelect(option.name)}
                    disabled={switching}
                    className={
                      "w-full rounded-lg px-2.5 py-2 text-left text-sm transition disabled:cursor-not-allowed disabled:opacity-50 " +
                      (active
                        ? "bg-[rgba(207,185,145,0.14)] text-[var(--gold)]"
                        : "text-[var(--ink)] hover:bg-white/5")
                    }
                  >
                    <span className="flex items-center gap-1.5 font-semibold">
                      {active && <span aria-hidden>✓</span>}
                      {option.label}
                    </span>
                    <span className="mt-0.5 block text-xs text-[var(--muted)]">
                      {option.blurb}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
          <p className="mt-2 border-t border-[var(--stroke)] px-2 pt-2 text-xs text-[var(--muted)]">
            Switching stops and relaunches the model — usually 30-90s, and only one model runs
            at a time.
          </p>
        </div>
      )}
    </div>
  );
}
