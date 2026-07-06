"use client";

import { useEffect, useState } from "react";
import SemesterCard from "@/components/SemesterCard";
import StudentProfilePanel from "@/components/StudentProfilePanel";
import AdvisorChat from "@/components/AdvisorChat";
import ProfileSetup from "@/components/ProfileSetup";
import {
  ApiError,
  createStudent,
  editPlan,
  generatePlan,
  getStudent,
  PlanEditOperation,
  PlanResponse,
  savePlan,
  StudentProfile,
} from "@/lib/api";

// The saved student row's id — the only client-side state that survives a reload.
// Everything else (profile, latest plan) is re-fetched from the backend on mount.
const STUDENT_ID_KEY = "boileradvisor.studentId";

type SaveState = "saved" | "saving" | "save-failed" | "local-only";

export default function HomePage() {
  const [initializing, setInitializing] = useState(true);
  const [studentId, setStudentId] = useState<string | null>(null);
  const [profile, setProfile] = useState<StudentProfile | null>(null);
  const [plan, setPlan] = useState<PlanResponse | null>(null);
  const [editingProfile, setEditingProfile] = useState(false);

  const [working, setWorking] = useState(false);
  const [saveState, setSaveState] = useState<SaveState>("saved");
  const [error, setError] = useState<string | null>(null);
  const [editError, setEditError] = useState<string | null>(null);

  // On mount: restore the saved student (if any) and their newest saved plan. There is
  // deliberately no default profile — first-time visitors land on onboarding.
  useEffect(() => {
    const remembered = localStorage.getItem(STUDENT_ID_KEY);
    if (!remembered) {
      setInitializing(false);
      return;
    }
    (async () => {
      try {
        const student = await getStudent(remembered);
        setStudentId(student.id);
        setProfile(student.profile);
        setPlan(student.plans[0]?.plan ?? null);
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) {
          // The row is gone (DB reset, etc.) — forget it and start fresh.
          localStorage.removeItem(STUDENT_ID_KEY);
        } else {
          setError(
            err instanceof Error ? err.message : "Failed to load saved profile",
          );
        }
      } finally {
        setInitializing(false);
      }
    })();
  }, []);

  // Fire-and-forget persistence: the UI already holds the server-validated plan, so a
  // failed save downgrades the status chip instead of blocking the edit.
  async function persistPlan(
    nextPlan: PlanResponse,
    feedback: string,
    id: string | null = studentId,
  ) {
    if (!id) {
      setSaveState("local-only");
      return;
    }
    try {
      setSaveState("saving");
      await savePlan(id, nextPlan, feedback);
      setSaveState("saved");
    } catch {
      setSaveState("save-failed");
    }
  }

  // Onboarding and profile editing share this path. Editing re-creates the student row
  // (there's no update route yet) and re-points localStorage at the new id.
  async function handleProfileSubmit(nextProfile: StudentProfile) {
    try {
      setWorking(true);
      setError(null);

      let id: string | null = null;
      try {
        const record = await createStudent(nextProfile);
        id = record.id;
        localStorage.setItem(STUDENT_ID_KEY, record.id);
      } catch {
        // DB down — keep going. The plan still generates (fixture fallback); it just
        // won't survive a reload, which the status chip makes visible.
        id = null;
        localStorage.removeItem(STUDENT_ID_KEY);
      }

      const generated = await generatePlan(nextProfile);
      setStudentId(id);
      setProfile(nextProfile);
      setPlan(generated);
      setEditingProfile(false);
      await persistPlan(generated, editingProfile ? "profile edited" : "initial plan", id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate plan");
    } finally {
      setWorking(false);
    }
  }

  async function handleRegenerate() {
    if (!profile) return;
    try {
      setWorking(true);
      setError(null);
      setEditError(null);
      const generated = await generatePlan(profile);
      setPlan(generated);
      await persistPlan(generated, "regenerated");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate plan");
    } finally {
      setWorking(false);
    }
  }

  function handleStartOver() {
    if (!window.confirm("Start over with a new profile? Your saved plans stay in the database.")) {
      return;
    }
    localStorage.removeItem(STUDENT_ID_KEY);
    setStudentId(null);
    setProfile(null);
    setPlan(null);
    setEditingProfile(false);
    setError(null);
    setEditError(null);
    setSaveState("saved");
  }

  // One deterministic move/add/remove round-tripped through POST /v1/plan/edit. The
  // backend re-validates the whole layout, so whatever comes back simply replaces the
  // plan state — warnings and credit totals are always the server's, never recomputed
  // client-side. A rejected edit (unknown course, duplicate add) leaves the plan as-is
  // and surfaces the backend's `detail` message. Accepted edits are autosaved.
  async function handlePlanEdit(
    operation: PlanEditOperation,
    courseCode: string,
    targetSemester: number | null = null,
  ) {
    if (!plan || !profile || working) return;
    try {
      setWorking(true);
      setEditError(null);
      const editedPlan = await editPlan(
        plan,
        operation,
        courseCode,
        targetSemester,
        profile,
      );
      setPlan(editedPlan);
      await persistPlan(editedPlan, `manual edit: ${operation} ${courseCode}`);
    } catch (err) {
      setEditError(err instanceof Error ? err.message : "Failed to edit plan");
    } finally {
      setWorking(false);
    }
  }

  async function handlePlanRevised(revisedPlan: PlanResponse) {
    setPlan(revisedPlan);
    await persistPlan(revisedPlan, "advisor revision");
  }

  const semesterOptions =
    plan?.semesters.map((semester, index) => ({
      value: index,
      label: `${capitalize(semester.term)} ${semester.year}`,
    })) ?? [];

  if (initializing) {
    return (
      <main className="mx-auto max-w-7xl px-4 py-16 md:px-8">
        <div className="card mx-auto max-w-md p-8 text-center text-[var(--muted)]">
          Loading your saved plan…
        </div>
      </main>
    );
  }

  // --- Onboarding: no profile yet (or the user chose to start over) -------------------
  if (!profile || editingProfile) {
    return (
      <main className="mx-auto max-w-7xl px-4 py-12 md:px-8">
        {!editingProfile && (
          <section className="mx-auto mb-12 max-w-3xl text-center">
            <p className="kicker">Boiler up. Plan ahead.</p>
            <h1 className="font-display mt-3 text-5xl font-bold uppercase leading-none tracking-wide text-[var(--ink)] md:text-7xl">
              Plan your <span className="text-[var(--gold)]">Purdue</span> degree
            </h1>
            <p className="mx-auto mt-5 max-w-xl text-[var(--muted)]">
              Semester-by-semester plans validated against real prerequisites,
              term offerings, and credit caps — with a local AI advisor that
              answers from the actual course catalog.
            </p>
          </section>
        )}

        {error && (
          <div className="mx-auto mb-6 max-w-3xl rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
            {error}
          </div>
        )}

        <div className="mx-auto max-w-3xl">
          <ProfileSetup
            initial={editingProfile && profile ? profile : undefined}
            submitLabel={
              editingProfile ? "Save & regenerate plan" : "Create profile & generate plan"
            }
            busy={working}
            onSubmit={handleProfileSubmit}
            onCancel={
              editingProfile ? () => setEditingProfile(false) : undefined
            }
          />
        </div>
      </main>
    );
  }

  // --- Main planner view ---------------------------------------------------------------
  return (
    <main className="mx-auto max-w-7xl px-4 py-10 md:px-8">
      <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="kicker">Degree Plan</p>
          <h1 className="font-display mt-1 text-4xl font-bold uppercase tracking-wide text-[var(--ink)] md:text-5xl">
            {plan?.student_name ?? profile.name}
            <span className="text-[var(--gold)]"> · {profile.degree_program}</span>
          </h1>
        </div>
        <SaveStatusChip state={saveState} />
      </div>

      {error && (
        <div className="mb-6 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="lg:col-span-3">
          <div className="lg:sticky lg:top-24">
            <StudentProfilePanel
              profile={profile}
              busy={working}
              onRegenerate={handleRegenerate}
              onEditProfile={() => setEditingProfile(true)}
              onStartOver={handleStartOver}
            />
          </div>
        </div>

        <div className="space-y-6 lg:col-span-6">
          {editError && (
            <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
              {editError}
            </div>
          )}

          {!plan && (
            <div className="card p-8 text-center">
              <p className="text-[var(--muted)]">
                No plan yet for this profile.
              </p>
              <button
                onClick={handleRegenerate}
                disabled={working}
                className="btn-gold mt-4 px-6 py-3"
              >
                {working ? "Generating…" : "Generate plan"}
              </button>
            </div>
          )}

          {plan?.semesters.map((semester, index) => (
            <SemesterCard
              key={`${semester.term}-${semester.year}`}
              title={`${capitalize(semester.term)} ${semester.year}`}
              credits={semester.total_credits}
              warnings={semester.warnings}
              courses={semester.courses.map((course) => ({
                code: course.code,
                name: course.title,
              }))}
              index={index}
              semesterOptions={semesterOptions}
              busy={working}
              onMoveCourse={(courseCode, targetSemester) =>
                handlePlanEdit("move", courseCode, targetSemester)
              }
              onRemoveCourse={(courseCode) =>
                handlePlanEdit("remove", courseCode)
              }
              onAddCourse={(courseCode, targetSemester) =>
                handlePlanEdit("add", courseCode, targetSemester)
              }
            />
          ))}

          {plan && plan.warnings.length > 0 && (
            <div className="card border-amber-500/30 p-6">
              <p className="kicker !text-amber-400">Plan warnings</p>
              <ul className="mt-3 space-y-2 text-sm leading-relaxed text-amber-200/90">
                {plan.warnings.map((warning) => (
                  <li key={warning}>• {warning}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="lg:col-span-3">
          <div className="lg:sticky lg:top-24">
            <AdvisorChat
              profile={profile}
              plan={plan}
              onPlanRevised={handlePlanRevised}
            />
          </div>
        </div>
      </div>
    </main>
  );
}

function SaveStatusChip({ state }: { state: SaveState }) {
  const styles: Record<SaveState, { label: string; className: string }> = {
    saved: {
      label: "Saved",
      className: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
    },
    saving: {
      label: "Saving…",
      className: "border-[var(--stroke-strong)] bg-[rgba(207,185,145,0.1)] text-[var(--gold)]",
    },
    "save-failed": {
      label: "Save failed — edits live only in this tab",
      className: "border-red-500/30 bg-red-500/10 text-red-300",
    },
    "local-only": {
      label: "Not saved — database unavailable",
      className: "border-amber-500/30 bg-amber-500/10 text-amber-300",
    },
  };
  const { label, className } = styles[state];
  return (
    <span
      className={`inline-flex items-center rounded-full border px-3 py-1.5 text-xs font-semibold ${className}`}
    >
      {label}
    </span>
  );
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
