const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

// All product routes live under /v1 (health probes stay unversioned on the backend).
const API_V1_URL = `${API_BASE_URL}/v1`;

export type StudentProfile = {
  // A LABEL for telling saved profiles apart on this device — optional, never required, and
  // never sent to the model. Not the student's name.
  profile_label: string;
  degree_program: string;
  // The chosen major. This is what makes the plan real: with it set, the backend derives
  // `remaining_courses` from that program's requirement rows in Postgres and ignores whatever
  // the client listed, and the AI planner refuses to run without it.
  program_id?: string | null;
  completed_courses: string[];
  remaining_courses: string[];
  start_term: string;
  start_year: number;
  semesters_to_plan: number;
  max_credits_per_semester: number;
  // A second cap, on how many of the major's own courses land in one term. Credits alone
  // cannot express it: four 4-credit CS courses and one CS course plus three gen-eds are both
  // 16 credits and nothing like the same semester.
  major_subject?: string;
  max_major_courses_per_semester?: number;
  preferred_major_courses_per_semester?: number;
  // A world-language subject code ("SPAN") the student wants to study — narrows a detected
  // language-sequence requirement down to that language instead of every language the
  // catalog offers. Populate the picker from `RequirementProgressResponse.available_languages`
  // for the student's chosen program, not a hardcoded list.
  language_of_interest?: string | null;
  // Roughly 1-3, matching Purdue's own "Level I/II/III" scheme. null/undefined = unknown,
  // treated as level 1 (start from the beginning).
  language_proficiency?: number | null;
  preferences: Record<string, unknown>;
};

// --- The degree catalog, as the programs browser reads it ------------------------------------
//
// Mirrors the backend's requirement models. SATISFACTION IS THREE-VALUED and the optional
// `satisfied` fields are why: `undefined` on a plain catalog view (nothing was cross-referenced
// against a student), `null` from an audit that could not DECIDE — a rule whose text the parser
// never resolved into course options, like "complete the College of Science core". Rendering
// "not decidable" as an unmet requirement would tell a student they are behind when the truth
// is that we don't know.

export type AcademicFacets = {
  catalog_years: number[];
  schools: string[];
  subjects: string[];
};

export type RequirementCourseOption = {
  id: string;
  sort_order: number;
  course_code_text: string;
  course_id?: string | null;
  course_title?: string | null;
  credits_text?: string | null;
  raw_text?: string | null;
  satisfied?: boolean;
  // Which list matched. A course completed for real always wins over merely being planned, so
  // the two are never both true — hence one field rather than two booleans.
  satisfied_by?: "completed" | "planned" | null;
};

export type RequirementRuleOption = {
  id: string;
  option_index: number;
  sort_order: number;
  label?: string | null;
  courses: RequirementCourseOption[];
  satisfied?: boolean | null;
};

export type RequirementRuleDetail = {
  id: string;
  sort_order: number;
  rule_type: string;
  choose_count?: number | null;
  credits_min?: number | null;
  raw_text?: string | null;
  options: RequirementRuleOption[];
  satisfied?: boolean | null;
};

export type RequirementBlockDetail = {
  id: string;
  sort_order: number;
  title?: string | null;
  credits_text?: string | null;
  rules: RequirementRuleDetail[];
  satisfied?: boolean | null;
};

export type AcademicProgramDetail = {
  id: string;
  catalog_year: number;
  school?: string | null;
  program_title: string;
  degree_code?: string | null;
  variant?: string | null;
  source_url: string;
  parser_status: string;
  blocks: RequirementBlockDetail[];
};

export async function fetchAcademicFacets(): Promise<AcademicFacets> {
  const response = await fetch(`${API_V1_URL}/academic/facets`);

  if (!response.ok) {
    await throwApiError(response, "Failed to load catalog facets");
  }

  return response.json();
}

export async function fetchProgramDetail(
  programId: string,
): Promise<AcademicProgramDetail> {
  const response = await fetch(`${API_V1_URL}/academic/programs/${programId}`);

  if (!response.ok) {
    await throwApiError(response, "Failed to load the program");
  }

  return response.json();
}

// One degree program from GET /v1/academic/programs — the major picker's row type.
export type AcademicProgramSummary = {
  id: string;
  catalog_year: number;
  school: string | null;
  program_title: string;
  degree_code: string | null;
  variant: string | null;
  source_url: string;
  parser_status: string;
  block_count: number;
  course_count: number;
  linked_course_count: number;
};

export type PlannedCourse = {
  code: string;
  title: string;
  credits: number;
};

export type SemesterPlan = {
  term: string;
  year: number;
  courses: PlannedCourse[];
  total_credits: number;
  warnings: string[];
};

export type PlanResponse = {
  profile_label: string;
  degree_program: string;
  semesters: SemesterPlan[];
  unplanned_courses: string[];
  warnings: string[];
};

// Mirrors the backend's PlanEditOperation Literal — one deterministic manipulation.
export type PlanEditOperation = "move" | "add" | "remove";

// One row from GET /v1/academic/courses/search (the real catalog DB).
export type AcademicCourseResult = {
  id: string;
  subject: string;
  number: string;
  code: string;
  title: string;
  credit_hours: number | null;
  description: string | null;
};

// A structured edit the local model proposes in response to free-text feedback.
export type PlanEditProposal = {
  rationale: string;
  reorder: string[];
  defer: string[];
  avoid_tags: string[];
  max_credits_per_semester: number | null;
};

export type RevisePlanResponse = {
  plan: PlanResponse;
  rationale: string;
  proposal: PlanEditProposal;
  iterations: number;
  // Which path built the revised schedule. "ai" re-runs the AI planner with the proposal folded
  // in, so the revision looks like the plan it revised; "deterministic" hands the proposal to
  // the greedy planner. A silent fallback from one to the other is visible here.
  planner: "ai" | "deterministic";
  // See AiPlanResponse.layout.
  layout: "model" | "planner";
  removed: string[];
  backfilled: string[];
  requirement_coverage: number;
  missing_requirements: string[];
};

// One catalog chunk retrieved to ground the answer (cosine similarity + stored tags).
export type AdvisorSource = {
  id: number;
  similarity: number;
  metadata: Record<string, unknown>;
  content: string;
};

export type AdvisorAskResponse = {
  answer: string;
  model: string;
  context_char_count: number;
  sources: AdvisorSource[];
};

// Carries the HTTP status so callers can branch on it — e.g. a 404 on the remembered
// student id means "start over with onboarding", while a 503 means "DB down, keep the id".
export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

// FastAPI errors carry a human-readable `detail` ("'CS999' was not found in the course
// catalog"); surface it instead of a bare status code so edit rejections read as advice.
async function throwApiError(response: Response, fallback: string): Promise<never> {
  let detail: string | null = null;
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") detail = body.detail;
  } catch {
    // Non-JSON error body; fall through to the generic message.
  }
  throw new ApiError(detail ?? `${fallback}: ${response.status}`, response.status);
}

export async function generatePlan(
  profile: StudentProfile,
): Promise<PlanResponse> {
  const response = await fetch(`${API_V1_URL}/plan/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(profile),
  });

  if (!response.ok) {
    await throwApiError(response, "Failed to generate plan");
  }

  return response.json();
}

// One deterministic move/add/remove applied to the current plan. Bypasses llama.cpp entirely:
// the backend mutates the layout and re-validates it against the planner's rules, so this
// round-trips in milliseconds with zero model compute. `targetSemester` is a zero-based
// index into plan.semesters (required for move/add, ignored for remove). Passing `profile`
// lets the validator check prerequisites against completed courses and enforce the
// per-semester credit cap.
export async function editPlan(
  plan: PlanResponse,
  operation: PlanEditOperation,
  courseCode: string,
  targetSemester: number | null = null,
  profile?: StudentProfile,
): Promise<PlanResponse> {
  const response = await fetch(`${API_V1_URL}/plan/edit`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      plan,
      operation,
      course_code: courseCode,
      target_semester: targetSemester,
      profile: profile ?? null,
    }),
  });

  if (!response.ok) {
    await throwApiError(response, "Failed to edit plan");
  }

  return response.json();
}

// Search the real degree catalog (951 crawled programs). Callers debounce; `query` matches the
// program title, and omitting it lists the first `limit` programs for the given filters.
export async function searchPrograms(params: {
  query?: string;
  catalogYear?: number;
  school?: string;
  limit?: number;
}): Promise<AcademicProgramSummary[]> {
  const search = new URLSearchParams();
  if (params.query) search.set("query", params.query);
  if (params.catalogYear != null) search.set("catalog_year", String(params.catalogYear));
  if (params.school) search.set("school", params.school);
  search.set("limit", String(params.limit ?? 40));

  const response = await fetch(`${API_V1_URL}/academic/programs?${search.toString()}`);

  if (!response.ok) {
    await throwApiError(response, "Failed to search degree programs");
  }

  return response.json();
}

// The AI-written plan of study (MODE B). Slow — a 26B model reading an ~11k-token catalog
// export takes tens of seconds — so callers must show a progress state rather than a spinner
// that looks stuck. `seed` is what makes "regenerate" produce a different plan instead of the
// same one: at the server's temperature an identical request is very nearly deterministic.
export type AiPlanResponse = {
  plan: PlanResponse;
  rationale: string;
  model: string;
  // False when llama-server was unreachable and the deterministic planner answered instead.
  // The UI says so rather than passing it off as an AI plan.
  used_model: boolean;
  // "model" — the model's own semester layout survived; "planner" — the deterministic planner
  // laid the semesters out from the model's course ordering because that covered more of the
  // degree. It decides the wording: a course missing from a "planner" layout was not
  // unschedulable, it just wasn't the option this layout picked.
  layout: "model" | "planner";
  model_placed: string[];
  removed: string[];
  backfilled: string[];
  violations: string[];
  requirement_coverage: number;
  missing_requirements: string[];
  seed: number | null;
};

// What the AI planner reported about how a plan was made: which courses it placed, what was
// dropped as unschedulable, what the deterministic planner added. Kept separate from
// `PlanResponse` because it describes how the plan was MADE and stops being true the moment
// the student edits a course by hand — see the `setProvenance(null)` calls in page.tsx.
export type PlanProvenance = Pick<
  AiPlanResponse,
  "model" | "used_model" | "layout" | "model_placed" | "removed" | "backfilled" |
  "requirement_coverage" | "missing_requirements"
>;

export async function aiGeneratePlan(
  profile: StudentProfile,
  request = "",
  seed?: number,
): Promise<AiPlanResponse> {
  const response = await fetch(`${API_V1_URL}/plan/ai-generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile, request, seed: seed ?? null }),
  });

  if (!response.ok) {
    await throwApiError(response, "Failed to generate a plan");
  }

  return response.json();
}

// Ask the model to explain an already-generated plan. Grounded on the structured plan alone —
// it cannot invent courses, because it is only ever shown the ones in the plan.
// --- The degree as a checklist (myPurduePlan-style) -----------------------------------------
//
// THREE STATES, and the distinction is the whole point:
//   completed  already on the transcript — nothing more to do. Gold.
//   planned    scheduled in the plan on screen, not yet taken. Provisional.
//   unfilled   nothing satisfies it yet. Red, and carries the options that could fill it.
//
// "unfilled" is deliberately not "missing": it is a decision the student has not made, not a
// mistake they have made. Nothing is auto-filled any more, so these holes are real.
export type SlotStatus = "completed" | "planned" | "unfilled";

export type SlotOption = {
  code: string;
  title: string;
  credits: number;
  // Semester indices where adding this course creates no new violation, checked with the same
  // validator the edit route uses — so an option offered here cannot be rejected on click.
  // Empty means "nowhere yet", which usually means another requirement has to be filled first.
  legal_semesters: number[];
};

export type RequirementSlot = {
  status: SlotStatus;
  filled_by: string | null;
  title: string | null;
  credits: number;
  semester_index: number | null;
  options: SlotOption[];
  label: string | null;
};

export type RequirementGroupProgress = {
  id: string;
  name: string;
  selective: boolean;
  credits_required: number | null;
  credits_filled: number;
  satisfied: boolean;
  slots: RequirementSlot[];
};

// A requirement the catalog states as a CREDIT THRESHOLD over a course-number range ("at
// least 32 semester hours... junior-level (30000+)") rather than as any specific course
// list — genuinely checkable, just not against a fixed option list.
// DOUBLE-COUNTING IS CORRECT HERE: a course already satisfying a major
// requirement still counts toward this too (Purdue's own catalog text says so), so this is
// computed independently over every completed/planned course, never subtracted elsewhere.
export type ComputedCourseView = {
  code: string;
  title: string;
  credits: number;
  status: SlotStatus;
};

export type ComputedRequirementView = {
  id: string;
  name: string;
  credits_required: number;
  credits_filled: number;
  satisfied: boolean;
  contributing_courses: ComputedCourseView[];
};

export type RequirementProgressResponse = {
  program_title: string;
  catalog_year: string;
  groups: RequirementGroupProgress[];
  groups_satisfied: number;
  groups_total: number;
  credits_completed: number;
  credits_planned: number;
  semester_labels: string[];
  // Requirements stated as a credit threshold rather than a course list — see
  // ComputedRequirementView. A second bucket, deliberately not in `groups`/`groups_satisfied`:
  // no fixed option list, so it has no slots and would change that fraction's denominator.
  computed: ComputedRequirementView[];
  // Subject code -> display name ("SPAN" -> "Spanish") for every language-sequence
  // requirement this program has, regardless of what the student already picked. Empty for a
  // program with no detected language-sequence requirement. Source for the language picker
  // in ProfileSetup — never hardcode a language list.
  available_languages: Record<string, string>;
};

export async function fetchRequirements(
  profile: StudentProfile,
  plan: PlanResponse,
): Promise<RequirementProgressResponse> {
  const response = await fetch(`${API_V1_URL}/plan/requirements`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile, plan }),
  });

  if (!response.ok) {
    await throwApiError(response, "Failed to load your requirements");
  }

  return response.json();
}

// An empty plan is enough — `available_languages` is computed from the program's catalog,
// not from what's scheduled, so this is the only way to ask "what languages could I pick"
// BEFORE a plan exists (during profile setup). `fetchRequirements` would work identically
// with this same empty plan; a dedicated helper just keeps callers from having to construct
// a throwaway PlanResponse inline every time they only want this one field.
export async function fetchAvailableLanguages(
  profile: StudentProfile,
): Promise<Record<string, string>> {
  const emptyPlan: PlanResponse = {
    profile_label: profile.profile_label,
    degree_program: profile.degree_program,
    semesters: [],
    unplanned_courses: [],
    warnings: [],
  };
  const result = await fetchRequirements(profile, emptyPlan);
  return result.available_languages;
}

// MODE C — the three things a student does after READING a plan. They are different policies,
// not three names for "try again", which is why they are one parameter rather than three
// endpoints: `fill` freezes what already validates and asks the model only for the gaps;
// `regenerate` tells the model exactly what is wrong and holds the seed fixed so the fix is
// attributable to the feedback; `start-over` resamples from scratch with a moving seed.
export type RefineMode = "fill" | "regenerate" | "start-over";

export type RefinePlanResponse = AiPlanResponse & {
  mode: RefineMode;
  // `fill` only: what survived the checker and was frozen against further change.
  kept: string[];
  // "nothing-to-fill" | "model-unavailable" | "" — set when the model was never called, so the
  // UI can explain an unchanged plan instead of looking like it did nothing.
  note: string;
};

export async function refinePlan(
  profile: StudentProfile,
  plan: PlanResponse,
  mode: RefineMode,
  options: { seed?: number; attempt?: number } = {},
): Promise<RefinePlanResponse> {
  const response = await fetch(`${API_V1_URL}/plan/refine`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      profile,
      plan,
      mode,
      seed: options.seed ?? null,
      attempt: options.attempt ?? 1,
    }),
  });

  if (!response.ok) {
    await throwApiError(response, "Failed to refine the plan");
  }

  return response.json();
}

export async function explainPlan(
  question: string,
  plan: PlanResponse,
  seed?: number,
): Promise<{ answer: string }> {
  const response = await fetch(`${API_V1_URL}/advisor/explain-plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, plan, seed: seed ?? null }),
  });

  if (!response.ok) {
    await throwApiError(response, "Failed to explain the plan");
  }

  return response.json();
}

// Substring search over the real catalog DB (title/description/code). Callers debounce.
export async function searchCourses(
  query: string,
  limit = 8,
): Promise<AcademicCourseResult[]> {
  const params = new URLSearchParams({ query, limit: String(limit) });
  const response = await fetch(
    `${API_V1_URL}/academic/courses/search?${params.toString()}`,
  );

  if (!response.ok) {
    await throwApiError(response, "Failed to search courses");
  }

  const data: { courses: AcademicCourseResult[] } = await response.json();
  return data.courses;
}

// Free-text feedback ("less theory-heavy", "cap me at 6 credits") that the local LFM2 agent
// turns into a proposal, which the deterministic planner re-validates into a revised plan.
export async function revisePlan(
  profile: StudentProfile,
  feedback: string,
  currentPlan?: PlanResponse,
  options: { planner?: "ai" | "deterministic"; seed?: number } = {},
): Promise<RevisePlanResponse> {
  const response = await fetch(`${API_V1_URL}/advisor/revise-plan`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      profile,
      feedback,
      current_plan: currentPlan ?? null,
      planner: options.planner ?? "ai",
      seed: options.seed ?? null,
    }),
  });

  if (!response.ok) {
    await throwApiError(response, "Failed to revise plan");
  }

  return response.json();
}

// --- Local-only profile persistence --------------------------------------------------
// NOTHING ABOUT A STUDENT IS SENT TO OR STORED ON THE SERVER. There used to be a
// `students`/`plans` pair of tables doing exactly that — a Postgres row per profile, a row
// per saved plan, keyed by a browser-remembered id. It was removed on purpose: a degree
// plan carries completed courses and a program of study, which reads as an education
// record, and a tool a prospective student is trying out has no business accumulating those
// server-side before anyone has decided the tool is trustworthy. Persistence is now entirely
// client-side — the browser's own localStorage for "reload the page and it's still there",
// and an exportable JSON file for "move it to another browser" or "keep an offline backup".

const SAVE_FILE_VERSION = 1;

export type SavedProfileFile = {
  version: number;
  profile: StudentProfile;
  plan: PlanResponse | null;
  // Optional: only present when the plan on screen still reflects how the model built it.
  // Dropped on any hand edit (see the `setProvenance(null)` calls in page.tsx), so a file
  // saved after an edit legitimately has none — that is not a bug in the export.
  provenance: PlanProvenance | null;
  // Whether the plan above was AI-generated or built empty for manual filling — restoring a
  // file should put the student back in the mode they were in, not silently default to AI.
  ai_mode: boolean;
};

// Serialises the current profile/plan/provenance into the download the "Save profile"
// button offers. A plain object, not a Blob — the caller decides how to turn it into a file
// (see `downloadProfileFile` below) so this stays testable without a DOM.
export function serializeProfileFile(
  profile: StudentProfile,
  plan: PlanResponse | null,
  provenance: PlanProvenance | null = null,
  aiMode = true,
): SavedProfileFile {
  return { version: SAVE_FILE_VERSION, profile, plan, provenance, ai_mode: aiMode };
}

// Triggers a browser download of the profile as a `.json` file the student keeps themselves —
// on their own disk, not in this app's database, because after this change there isn't one.
export function downloadProfileFile(
  profile: StudentProfile,
  plan: PlanResponse | null,
  provenance: PlanProvenance | null = null,
  aiMode = true,
): void {
  const payload = serializeProfileFile(profile, plan, provenance, aiMode);
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const slug = (profile.profile_label || profile.degree_program || "profile")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  link.href = url;
  link.download = `boileradvisor-${slug || "profile"}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

// Reads a file the student picked (from a previous export) back into a profile/plan pair.
// Only shape-checked, never trusted further than that — it round-trips through the same
// `/v1/plan/*` validation as anything else once the app uses it.
export async function readProfileFile(file: File): Promise<SavedProfileFile> {
  const text = await file.text();
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("That file isn't valid JSON.");
  }
  const candidate = parsed as Partial<SavedProfileFile>;
  if (!candidate || typeof candidate !== "object" || !candidate.profile) {
    throw new Error("That file doesn't look like a BoilerAdvisor profile.");
  }
  return {
    version: candidate.version ?? SAVE_FILE_VERSION,
    profile: candidate.profile as StudentProfile,
    plan: candidate.plan ?? null,
    provenance: candidate.provenance ?? null,
    // Files saved before AI Mode existed have no `ai_mode` key — they were all AI-generated
    // plans (it's the only thing that existed then), so `true` is the correct default, not
    // just a safe one.
    ai_mode: candidate.ai_mode ?? true,
  };
}

// --- Admin visibility (GET /v1/admin/*) — read-only database browsing ------------------

export type AdminTableInfo = {
  name: string;
  row_count: number;
};

export type AdminTableRows = {
  table: string;
  columns: string[];
  rows: Record<string, unknown>[];
  total: number;
  limit: number;
  offset: number;
};

export async function fetchAdminTables(): Promise<AdminTableInfo[]> {
  const response = await fetch(`${API_V1_URL}/admin/tables`);

  if (!response.ok) {
    await throwApiError(response, "Failed to list tables");
  }

  const data: { tables: AdminTableInfo[] } = await response.json();
  return data.tables;
}

export async function fetchAdminTableRows(
  table: string,
  limit = 25,
  offset = 0,
): Promise<AdminTableRows> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  const response = await fetch(
    `${API_V1_URL}/admin/tables/${table}?${params.toString()}`,
  );

  if (!response.ok) {
    await throwApiError(response, "Failed to load table rows");
  }

  return response.json();
}

// Free-text question to the local advisor model. The backend attaches a fixed catalog
// context (all degrees + requirements + referenced courses) to every question, so the
// answer is grounded in the real database regardless of what the student asks.
export async function askAdvisor(
  question: string,
  seed?: number,
): Promise<AdvisorAskResponse> {
  const response = await fetch(`${API_V1_URL}/advisor/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    // `seed` re-rolls only the wording. Retrieval stays deterministic, so a regenerated answer
    // is the same evidence explained differently — not a different set of rules.
    body: JSON.stringify({ question, seed: seed ?? null }),
  });

  if (!response.ok) {
    await throwApiError(response, "Failed to get advisor response");
  }

  return response.json();
}

// --- Local model selection (GET /v1/models, POST /v1/models/{name}/select) -------------------
//
// The backend owns llama-server's process lifecycle (see backend/app/services/model_manager.py)
// so it can stop and relaunch it with a different model. Only one 26-35B-class model fits in
// the box's VRAM at a time, so a switch is a real 30-90s wait, not an instant toggle — the
// picker component is expected to show that as a loading state.

export type ModelOption = {
  name: string;
  label: string;
  blurb: string;
};

export type ModelStatus = {
  current: string | null;
  switching_to: string | null;
  last_error: string | null;
  available: ModelOption[];
};

export async function fetchModelStatus(): Promise<ModelStatus> {
  const response = await fetch(`${API_V1_URL}/models`);

  if (!response.ok) {
    await throwApiError(response, "Failed to load model status");
  }

  return response.json();
}

// Resolves once the new model is loaded and healthy (or rejects) — no polling needed, but the
// caller should expect this to take up to llamacpp_startup_timeout_s (180s server-side default).
export async function selectModel(name: string): Promise<ModelStatus> {
  const response = await fetch(`${API_V1_URL}/models/${encodeURIComponent(name)}/select`, {
    method: "POST",
  });

  if (!response.ok) {
    await throwApiError(response, "Failed to switch models");
  }

  return response.json();
}
