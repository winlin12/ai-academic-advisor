"use client";

import { useEffect, useState } from "react";
import SemesterCard from "@/components/SemesterCard";
import StudentProfilePanel from "@/components/StudentProfilePanel";
import AdvisorChat from "@/components/AdvisorChat";
import ProfileSetup from "@/components/ProfileSetup";
import PlanActions from "@/components/PlanActions";
import RequirementChecklist from "@/components/RequirementChecklist";
import {
  aiGeneratePlan,
  downloadProfileFile,
  editPlan,
  fetchRequirements,
  generatePlan,
  PlanEditOperation,
  PlanProvenance,
  PlanResponse,
  readProfileFile,
  RefineMode,
  refinePlan,
  RequirementProgressResponse,
  StudentProfile,
} from "@/lib/api";

// Everything that survives a reload lives here, in the browser, and nowhere else. See
// `serializeProfileFile` in lib/api for why: no server-side row backs this any more.
const PROFILE_STORAGE_KEY = "boileradvisor.profile";

type SaveState = "saved" | "save-failed";

// A fresh seed per generation. Without it the model returns the same plan every time, and
// "regenerate" would look broken rather than deterministic.
function newSeed() {
  return Math.floor(Math.random() * 2_000_000_000);
}

export default function HomePage() {
  const [initializing, setInitializing] = useState(true);
  const [profile, setProfile] = useState<StudentProfile | null>(null);
  const [plan, setPlan] = useState<PlanResponse | null>(null);
  // OFF means no model ever ran and nothing was auto-placed — the plan on screen is an empty
  // shell (correct term/year calendar, zero courses) the student fills in by hand from the
  // Requirements tab. Gates the AI-only actions (PlanActions' Fill/Regenerate/Start over):
  // none of them make sense to offer over a plan the student deliberately chose to build
  // themselves.
  const [aiMode, setAiMode] = useState(true);
  const [editingProfile, setEditingProfile] = useState(false);

  const [working, setWorking] = useState(false);
  // Separate from `working` because it is the only operation slow enough to need explaining:
  // a 26B model reading an ~11k-token catalog export takes tens of seconds, and a button that
  // just says "Working…" for half a minute reads as broken.
  const [planning, setPlanning] = useState(false);
  const [provenance, setProvenance] = useState<PlanProvenance | null>(null);
  // Which MODE C action is in flight, so the button that was pressed says so rather than every
  // button greying out identically.
  const [refining, setRefining] = useState<RefineMode | null>(null);
  // Counts "Start over" presses. The backend creeps the temperature with it, so pressing it
  // repeatedly explores instead of redrawing the same plan at the same sampling settings.
  const [startOvers, setStartOvers] = useState(1);
  const [refineNote, setRefineNote] = useState<string | null>(null);
  // The degree checklist, refetched whenever the plan changes. Nothing is auto-filled any more,
  // so this is where the holes become something the student can act on.
  const [requirements, setRequirements] = useState<RequirementProgressResponse | null>(null);
  const [view, setView] = useState<"plan" | "requirements">("plan");
  const [saveState, setSaveState] = useState<SaveState>("saved");
  const [error, setError] = useState<string | null>(null);
  const [editError, setEditError] = useState<string | null>(null);

  // On mount: restore whatever this browser saved locally, if anything. There is
  // deliberately no default profile — first-time visitors land on onboarding. No network
  // round trip here any more — the save is the localStorage read itself.
  useEffect(() => {
    const raw = localStorage.getItem(PROFILE_STORAGE_KEY);
    if (raw) {
      try {
        const stored = JSON.parse(raw) as {
          profile: StudentProfile;
          plan: PlanResponse | null;
          provenance: PlanProvenance | null;
          ai_mode?: boolean;
        };
        setProfile(stored.profile);
        setPlan(stored.plan);
        setProvenance(stored.provenance);
        setAiMode(stored.ai_mode ?? true);
      } catch {
        // Corrupt entry from an older shape — drop it rather than crash onboarding.
        localStorage.removeItem(PROFILE_STORAGE_KEY);
      }
    }
    setInitializing(false);
  }, []);

  // Requirements follow the plan. Deterministic and fast (no LLM), so it can simply re-run on
  // every change rather than being invalidated by hand from six call sites.
  useEffect(() => {
    if (!profile?.program_id || !plan) {
      setRequirements(null);
      return;
    }
    let cancelled = false;
    fetchRequirements(profile, plan)
      .then((result) => {
        if (!cancelled) setRequirements(result);
      })
      .catch(() => {
        if (!cancelled) setRequirements(null);
      });
    return () => {
      cancelled = true;
    };
  }, [profile, plan]);

  // The only "save" there is: a synchronous write to this browser's localStorage. Takes
  // explicit values rather than reading state, because callers just called `setProfile` /
  // `setPlan` and React hasn't re-rendered yet — reading `profile`/`plan` here would save
  // the PREVIOUS values.
  function saveLocal(
    nextProfile: StudentProfile,
    nextPlan: PlanResponse | null,
    nextProvenance: PlanProvenance | null,
    nextAiMode: boolean = aiMode,
  ) {
    try {
      localStorage.setItem(
        PROFILE_STORAGE_KEY,
        JSON.stringify({
          profile: nextProfile, plan: nextPlan, provenance: nextProvenance,
          ai_mode: nextAiMode,
        }),
      );
      setSaveState("saved");
    } catch {
      // Private browsing with storage disabled, or quota exceeded. The plan is still on
      // screen and still exportable by hand — only "survives a reload" is lost.
      setSaveState("save-failed");
    }
  }

  // AI MODE OFF: no model runs, nothing is auto-placed — not even by the deterministic
  // planner. `generatePlan` is still called once, but only to borrow its correct term/year
  // calendar (fall/spring alternation, optional summer, the student's own start term/year);
  // every course it placed is stripped immediately after. The student fills the result in by
  // hand from the Requirements tab, through the same validated /plan/edit route as any other
  // manual placement.
  async function buildBlankPlan(forProfile: StudentProfile): Promise<PlanResponse> {
    const shaped = await generatePlan(forProfile);
    return {
      ...shaped,
      semesters: shaped.semesters.map((semester) => ({
        ...semester, courses: [], total_credits: 0, warnings: [],
      })),
      unplanned_courses: [],
      warnings: [],
    };
  }

  // THE INITIAL PLAN OF STUDY comes from the AI planner when a major was chosen, because that
  // is the only path with the program's requirement groups, prerequisite edges and observed
  // term offerings behind it. Profiles with a hand-typed course list get the deterministic
  // planner — the AI planner refuses without a program, and rightly so.
  //
  // Never throws for a downed model: /plan/ai-generate falls back server-side and reports
  // `used_model: false`, which the provenance panel shows.
  async function buildPlan(
    forProfile: StudentProfile,
    aiModeForBuild: boolean,
    seed?: number,
  ): Promise<{ plan: PlanResponse; provenance: PlanProvenance | null }> {
    if (!aiModeForBuild) {
      return { plan: await buildBlankPlan(forProfile), provenance: null };
    }
    if (!forProfile.program_id) {
      return { plan: await generatePlan(forProfile), provenance: null };
    }
    const result = await aiGeneratePlan(forProfile, "", seed ?? newSeed());
    return {
      plan: result.plan,
      provenance: {
        model: result.model,
        used_model: result.used_model,
        layout: result.layout,
        model_placed: result.model_placed,
        removed: result.removed,
        backfilled: result.backfilled,
        requirement_coverage: result.requirement_coverage,
        missing_requirements: result.missing_requirements,
      },
    };
  }

  // Onboarding and profile editing share this path. Editing re-creates the student row
  // (there's no update route yet) and re-points localStorage at the new id.
  async function handleProfileSubmit(nextProfile: StudentProfile, nextAiMode: boolean) {
    try {
      setWorking(true);
      setPlanning(nextAiMode && Boolean(nextProfile.program_id));
      setError(null);

      const built = await buildPlan(nextProfile, nextAiMode);
      setProfile(nextProfile);
      setPlan(built.plan);
      setProvenance(built.provenance);
      setAiMode(nextAiMode);
      setEditingProfile(false);
      setPlanning(false);
      // Manual mode has nothing for the Schedule tab to show yet — land where there's
      // something to do.
      setView(nextAiMode ? "plan" : "requirements");
      saveLocal(nextProfile, built.plan, built.provenance, nextAiMode);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate plan");
    } finally {
      setWorking(false);
      setPlanning(false);
    }
  }

  // MODE C. Fill / Regenerate / Start over — three different policies over the plan on screen,
  // not three names for the same retry. See components/PlanActions.
  async function handleRefine(mode: RefineMode) {
    if (!profile || !plan || working) return;
    try {
      setWorking(true);
      setRefining(mode);
      setRefineNote(null);
      setError(null);
      setEditError(null);

      // Seed policy mirrors the backend's, and the reason is the same: "regenerate" holds the
      // seed fixed so a better plan is attributable to the feedback rather than to a luckier
      // sample, while "start over" moves it because resampling is the entire point.
      const result = await refinePlan(profile, plan, mode, {
        seed: mode === "start-over" ? newSeed() : undefined,
        attempt: mode === "start-over" ? startOvers : 1,
      });
      if (mode === "start-over") setStartOvers((n) => n + 1);

      const nextProvenance: PlanProvenance = {
        model: result.model,
        used_model: result.used_model,
        layout: result.layout,
        model_placed: result.model_placed,
        removed: result.removed,
        backfilled: result.backfilled,
        requirement_coverage: result.requirement_coverage,
        missing_requirements: result.missing_requirements,
      };
      setPlan(result.plan);
      setProvenance(nextProvenance);
      if (result.note === "nothing-to-fill") {
        setRefineNote(
          "Nothing to fill — this plan already covers every requirement in your program.",
        );
      } else if (result.note === "model-unavailable") {
        setRefineNote(
          "The model could not be reached, so your plan is unchanged apart from the checks " +
            "that run without it.",
        );
      } else if (mode === "fill") {
        setRefineNote(
          `Kept ${result.kept.length} placement(s) exactly where they were and filled in ` +
            `${result.backfilled.length}.`,
        );
      }
      setRefining(null);
      if (profile) saveLocal(profile, result.plan, nextProvenance);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to refine the plan");
    } finally {
      setWorking(false);
      setRefining(null);
    }
  }

  // A NEW seed every time, so this is a genuinely different plan rather than the same one
  // redrawn. The model has real latitude here — which requirement options it picks and how it
  // balances the terms — so a student who does not like a layout can simply ask again.
  async function handleRegenerate() {
    if (!profile) return;
    try {
      setWorking(true);
      setPlanning(aiMode && Boolean(profile.program_id));
      setError(null);
      setEditError(null);
      const built = await buildPlan(profile, aiMode, newSeed());
      setPlan(built.plan);
      setProvenance(built.provenance);
      setPlanning(false);
      saveLocal(profile, built.plan, built.provenance);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate plan");
    } finally {
      setWorking(false);
      setPlanning(false);
    }
  }

  function handleNewProfile() {
    if (
      !window.confirm(
        "Discard this profile and start again? This only clears it from this browser — " +
          "save it to a file first if you want to keep it.",
      )
    ) {
      return;
    }
    localStorage.removeItem(PROFILE_STORAGE_KEY);
    setProfile(null);
    setPlan(null);
    setProvenance(null);
    setAiMode(true);
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
      // The provenance panel described how the AI built this layout. One hand edit later that
      // is no longer what is on screen, so it goes rather than quietly misdescribing the plan.
      setProvenance(null);
      saveLocal(profile, editedPlan, null);
    } catch (err) {
      setEditError(err instanceof Error ? err.message : "Failed to edit plan");
    } finally {
      setWorking(false);
    }
  }

  // Filling a red slot by hand. Deliberately the SAME route as dragging a course between
  // semesters — the checklist is a different view of the plan, not a different way to write it,
  // so the edit is re-validated exactly like every other one.
  async function handleFillRequirement(courseCode: string, semesterIndex: number) {
    await handlePlanEdit("add", courseCode, semesterIndex);
  }

  async function handlePlanRevised(revisedPlan: PlanResponse) {
    setPlan(revisedPlan);
    setProvenance(null);
    if (profile) saveLocal(profile, revisedPlan, null);
  }

  function handleExportProfile() {
    if (profile) downloadProfileFile(profile, plan, provenance, aiMode);
  }

  // Loading a file is the portability story: it works with no server involved, so it is the
  // one path that survives moving to a different browser or a different computer entirely.
  async function handleImportProfile(file: File) {
    try {
      setError(null);
      const loaded = await readProfileFile(file);
      setProfile(loaded.profile);
      setPlan(loaded.plan);
      setProvenance(loaded.provenance);
      setAiMode(loaded.ai_mode);
      setEditingProfile(false);
      saveLocal(loaded.profile, loaded.plan, loaded.provenance, loaded.ai_mode);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load that file");
    }
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

        {!editingProfile && (
          <div className="mx-auto mb-6 max-w-3xl text-center">
            <label className="btn-ghost inline-flex cursor-pointer px-4 py-2 text-sm">
              Load a saved profile file
              <input
                type="file"
                accept="application/json"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void handleImportProfile(file);
                  event.target.value = "";
                }}
              />
            </label>
            <p className="mt-1.5 text-xs text-[var(--muted)]">
              Saved a profile on another browser or computer? Load its file here — nothing
              is stored on our end to fetch it from.
            </p>
          </div>
        )}

        <div className="mx-auto max-w-3xl">
          <ProfileSetup
            initial={editingProfile && profile ? profile : undefined}
            initialAiMode={editingProfile ? aiMode : undefined}
            submitLabel={editingProfile ? "Save & rebuild plan" : "Create profile & build my plan"}
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
            {(plan?.profile_label || profile.profile_label) && (
              <>
                {plan?.profile_label || profile.profile_label}
                <span className="text-[var(--gold)]"> · </span>
              </>
            )}
            <span className={plan?.profile_label || profile.profile_label ? "text-[var(--gold)]" : ""}>
              {profile.degree_program}
            </span>
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
              aiMode={aiMode}
              busy={working}
              onEditProfile={() => setEditingProfile(true)}
              onNewProfile={handleNewProfile}
              onExportProfile={handleExportProfile}
              onImportProfile={handleImportProfile}
            />
          </div>
        </div>

        <div className="space-y-6 lg:col-span-6">
          {editError && (
            <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
              {editError}
            </div>
          )}

          {planning && (
            <div className="card border-[var(--stroke-strong)] p-6">
              <p className="kicker">Writing your plan</p>
              <p className="mt-2 text-sm text-[var(--muted)]">
                The local model is reading your program&rsquo;s full catalog — every course,
                prerequisite chain and term offering — and laying out the semesters. This takes
                a few seconds; every placement is then re-checked against the same rules before
                you see it.
              </p>
              <div className="mt-4 h-1 w-full overflow-hidden rounded-full bg-black/40">
                <div className="h-full w-1/3 animate-pulse rounded-full bg-[var(--gold)]" />
              </div>
            </div>
          )}

          {!plan && !planning && (
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

          {plan && provenance && (
            <ProvenancePanel provenance={provenance} />
          )}

          {plan && !provenance && !aiMode && (
            <div className="card p-5">
              <p className="kicker">Built manually — no model involved</p>
              <p className="mt-2 text-sm text-[var(--muted)]">
                Every semester below started empty. Add courses from the Requirements tab (or
                drag them into a semester here) — each placement is checked against
                prerequisites, term offerings and your credit caps the same way an AI-written
                one would be, it just starts from nothing rather than a first draft.
              </p>
            </div>
          )}

          {plan && profile?.program_id && aiMode && (
            <PlanActions
              canFill={
                !provenance ||
                provenance.requirement_coverage < 1 ||
                plan.unplanned_courses.length > 0
              }
              busy={working}
              running={refining}
              onRefine={handleRefine}
            />
          )}

          {refineNote && (
            <div className="rounded-xl border border-[var(--stroke-strong)] bg-[rgba(207,185,145,0.08)] p-4 text-sm text-[var(--ink)]">
              {refineNote}
            </div>
          )}

          {plan && requirements && (
            <div className="flex rounded-full border border-[var(--stroke)] bg-black/30 p-1 text-xs font-semibold">
              {(["plan", "requirements"] as const).map((value) => {
                const open = requirements.groups_total - requirements.groups_satisfied;
                return (
                  <button
                    key={value}
                    onClick={() => setView(value)}
                    className={
                      view === value
                        ? "btn-gold flex-1 !rounded-full px-3 py-1.5"
                        : "flex-1 rounded-full px-3 py-1.5 text-[var(--muted)] transition hover:text-[var(--ink)]"
                    }
                  >
                    {value === "plan" ? "Schedule" : "Requirements"}
                    {value === "requirements" && open > 0 && (
                      <span
                        className={
                          view === value
                            ? "ml-1.5"
                            : "ml-1.5 text-red-300"
                        }
                      >
                        ({open} open)
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          )}

          {view === "requirements" && requirements && (
            <RequirementChecklist
              progress={requirements}
              busy={working}
              onFill={handleFillRequirement}
            />
          )}

          {view === "plan" && plan?.semesters.map((semester, index) => (
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

          {view === "plan" && plan && plan.warnings.length > 0 && (
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

// WHERE THIS PLAN CAME FROM, stated plainly. The model drafts the schedule and the app repairs
// it, so "the AI made this" is only partly true and the student is entitled to the specifics:
// how much of the degree it covers, what was dropped as unschedulable, what was added back.
// Anything hidden here is something a student would find out later, from a registrar.
function ProvenancePanel({ provenance }: { provenance: PlanProvenance }) {
  const coverage = Math.round(provenance.requirement_coverage * 100);
  const complete = coverage >= 100;

  return (
    <div className="card p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="kicker">
          {!provenance.used_model
            ? "Written by the planner"
            : provenance.layout === "model"
              ? "Written by the local model"
              : "Ordered by the local model, laid out by the planner"}
        </p>
        <span
          className={`rounded-full border px-3 py-1 text-xs font-semibold ${
            complete
              ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
              : "border-amber-500/30 bg-amber-500/10 text-amber-300"
          }`}
        >
          {coverage}% of requirement groups
        </span>
      </div>

      {!provenance.used_model && (
        <p className="mt-3 text-sm text-amber-200/90">
          The AI planner wasn&rsquo;t reachable, so this came from the deterministic planner
          instead. It&rsquo;s a legal plan — it just won&rsquo;t reflect any preferences you
          described.
        </p>
      )}

      <dl className="mt-3 space-y-1.5 text-xs text-[var(--muted)]">
        {provenance.used_model && (
          <div>
            <dt className="inline font-semibold text-[var(--ink)]">Model: </dt>
            <dd className="inline">{provenance.model}</dd>
          </div>
        )}
        {provenance.removed.length > 0 && (
          <div>
            <dt className="inline font-semibold text-[var(--ink)]">
              {provenance.layout === "model"
                ? "Dropped as unschedulable: "
                : "In the draft but not this layout: "}
            </dt>
            <dd className="inline">{provenance.removed.join(", ")}</dd>
          </div>
        )}
        {provenance.backfilled.length > 0 && (
          <div>
            <dt className="inline font-semibold text-[var(--ink)]">
              {provenance.layout === "model"
                ? "Added by the planner: "
                : "Chosen by the planner: "}
            </dt>
            <dd className="inline">{provenance.backfilled.join(", ")}</dd>
          </div>
        )}
      </dl>

      {provenance.missing_requirements.length > 0 && (
        <div className="mt-3 border-t border-[var(--stroke)] pt-3">
          <p className="text-xs font-semibold text-amber-300">
            Not covered within these semesters
          </p>
          <ul className="mt-1.5 space-y-1 text-xs text-[var(--muted)]">
            {provenance.missing_requirements.map((item) => (
              <li key={item}>• {item}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function SaveStatusChip({ state }: { state: SaveState }) {
  const styles: Record<SaveState, { label: string; className: string }> = {
    saved: {
      label: "Saved on this device",
      className: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
    },
    "save-failed": {
      label: "Not saved — save to a file instead",
      className: "border-red-500/30 bg-red-500/10 text-red-300",
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
