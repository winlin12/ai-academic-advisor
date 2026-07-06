"use client";

import { useEffect, useState } from "react";
import {
  AdminTableInfo,
  AdminTableRows,
  fetchAdminTables,
  fetchAdminTableRows,
} from "@/lib/api";

const PAGE_SIZE = 25;

// Read-only database browser (TODO Priority 6): table row counts + paged row contents,
// via GET /v1/admin/*. For editing rows, use Adminer (make adminer → localhost:8081).
export default function AdminPage() {
  const [tables, setTables] = useState<AdminTableInfo[]>([]);
  const [tablesError, setTablesError] = useState<string | null>(null);
  const [loadingTables, setLoadingTables] = useState(true);

  const [selected, setSelected] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [rows, setRows] = useState<AdminTableRows | null>(null);
  const [rowsError, setRowsError] = useState<string | null>(null);
  const [loadingRows, setLoadingRows] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        setLoadingTables(true);
        setTablesError(null);
        const infos = await fetchAdminTables();
        setTables(infos);
        if (infos.length > 0) setSelected((current) => current ?? infos[0].name);
      } catch (err) {
        setTablesError(
          err instanceof Error ? err.message : "Failed to list tables",
        );
      } finally {
        setLoadingTables(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (!selected) return;
    (async () => {
      try {
        setLoadingRows(true);
        setRowsError(null);
        const data = await fetchAdminTableRows(selected, PAGE_SIZE, page * PAGE_SIZE);
        setRows(data);
      } catch (err) {
        setRows(null);
        setRowsError(
          err instanceof Error ? err.message : "Failed to load rows",
        );
      } finally {
        setLoadingRows(false);
      }
    })();
  }, [selected, page]);

  const totalPages = rows ? Math.max(1, Math.ceil(rows.total / PAGE_SIZE)) : 1;

  return (
    <main className="mx-auto max-w-7xl px-4 py-10 md:px-8">
      <div className="mb-8">
        <p className="kicker">Admin</p>
        <h1 className="font-display mt-1 text-4xl font-bold uppercase tracking-wide text-[var(--ink)] md:text-5xl">
          Database <span className="text-[var(--gold)]">Browser</span>
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-[var(--muted)]">
          Read-only view of the catalog database. For editing, run{" "}
          <code className="rounded bg-black/40 px-1.5 py-0.5 text-[var(--gold)]">
            make adminer
          </code>{" "}
          and open{" "}
          <a
            href="http://localhost:8081"
            target="_blank"
            rel="noreferrer"
            className="text-[var(--gold)] underline decoration-[var(--gold-aged)] underline-offset-2 hover:text-[var(--gold-soft)]"
          >
            Adminer
          </a>
          .
        </p>
      </div>

      {tablesError && (
        <div className="mb-6 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
          {tablesError}
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="lg:col-span-3">
          <div className="card p-4 lg:sticky lg:top-24">
            <p className="kicker mb-3">Tables</p>
            {loadingTables && (
              <p className="px-2 py-1 text-sm text-[var(--muted)]">Loading…</p>
            )}
            <ul className="space-y-1">
              {tables.map((table) => {
                const active = table.name === selected;
                return (
                  <li key={table.name}>
                    <button
                      onClick={() => {
                        setSelected(table.name);
                        setPage(0);
                      }}
                      className={
                        active
                          ? "flex w-full items-center justify-between rounded-lg bg-[rgba(207,185,145,0.14)] px-3 py-2 text-sm font-semibold text-[var(--gold)]"
                          : "flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm text-[var(--muted)] transition hover:bg-[rgba(207,185,145,0.07)] hover:text-[var(--ink)]"
                      }
                    >
                      <span className="font-mono">{table.name}</span>
                      <span className="text-xs tabular-nums">
                        {table.row_count.toLocaleString()}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        </div>

        <div className="lg:col-span-9">
          <div className="card p-5">
            {!selected && !loadingTables && (
              <p className="p-4 text-sm text-[var(--muted)]">
                No browsable tables found.
              </p>
            )}

            {selected && (
              <>
                <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                  <h2 className="font-display text-2xl font-semibold uppercase tracking-wide text-[var(--ink)]">
                    <span className="font-mono normal-case text-[var(--gold)]">
                      {selected}
                    </span>
                  </h2>
                  {rows && (
                    <div className="flex items-center gap-2 text-xs text-[var(--muted)]">
                      <button
                        onClick={() => setPage((current) => Math.max(0, current - 1))}
                        disabled={page === 0 || loadingRows}
                        className="btn-ghost px-3 py-1.5"
                      >
                        ← Prev
                      </button>
                      <span className="tabular-nums">
                        Page {page + 1} / {totalPages} · {rows.total.toLocaleString()} rows
                      </span>
                      <button
                        onClick={() =>
                          setPage((current) => Math.min(totalPages - 1, current + 1))
                        }
                        disabled={page >= totalPages - 1 || loadingRows}
                        className="btn-ghost px-3 py-1.5"
                      >
                        Next →
                      </button>
                    </div>
                  )}
                </div>

                {loadingRows && (
                  <p className="p-4 text-sm text-[var(--muted)]">Loading rows…</p>
                )}

                {rowsError && (
                  <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
                    {rowsError}
                  </div>
                )}

                {rows && !loadingRows && rows.rows.length === 0 && (
                  <p className="rounded-lg border border-dashed border-[var(--stroke)] p-8 text-center text-sm text-[var(--muted)]">
                    Table is empty.
                  </p>
                )}

                {rows && !loadingRows && rows.rows.length > 0 && (
                  <div className="scroll-dark overflow-x-auto rounded-lg border border-[var(--stroke)]">
                    <table className="w-full border-collapse text-left text-xs">
                      <thead>
                        <tr className="border-b border-[var(--stroke-strong)] bg-black/40">
                          {rows.columns.map((column) => (
                            <th
                              key={column}
                              className="whitespace-nowrap px-3 py-2 font-mono font-semibold text-[var(--gold)]"
                            >
                              {column}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {rows.rows.map((row, rowIndex) => (
                          <tr
                            key={rowIndex}
                            className="border-b border-[var(--stroke)] transition last:border-b-0 hover:bg-[rgba(207,185,145,0.05)]"
                          >
                            {rows.columns.map((column) => (
                              <td
                                key={column}
                                className="max-w-[22rem] truncate px-3 py-2 align-top font-mono text-[var(--muted)]"
                                title={formatCell(row[column])}
                              >
                                {formatCell(row[column])}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "∅";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
