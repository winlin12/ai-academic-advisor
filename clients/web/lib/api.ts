const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

// All product routes live under /v1 (health probes stay unversioned on the backend).
const API_V1_URL = `${API_BASE_URL}/v1`;

export type StudentProfile = {
  name: string;
  degree_program: string;
  program_id?: string | null;
  completed_courses: string[];
  remaining_courses: string[];
  start_term: string;
  start_year: number;
  semesters_to_plan: number;
  max_credits_per_semester: number;
  preferences: Record<string, unknown>;
};

export type PlannedCourse = {
  code: string;
  title: string;
  credits: number;
  workload_score: number;
};

export type SemesterPlan = {
  term: string;
  year: number;
  courses: PlannedCourse[];
  total_credits: number;
  average_workload: number;
  warnings: string[];
};

export type PlanResponse = {
  student_name: string;
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
    }),
  });

  if (!response.ok) {
    await throwApiError(response, "Failed to revise plan");
  }

  return response.json();
}

// --- Student/plan persistence (POST /v1/students, /v1/students/{id}/plans) -----------
// The student row is the durable identity: the web client remembers its id in
// localStorage and reloads the newest saved plan on mount, so edits survive reloads.

export type StudentRecord = {
  id: string;
  name: string;
  profile: StudentProfile;
  created_at: string;
};

export type SavedPlan = {
  id: string;
  feedback: string | null;
  plan: PlanResponse;
  created_at: string;
};

// GET /v1/students/{id} — plans come back newest-first, so plans[0] is the live one.
export type StudentDetail = StudentRecord & { plans: SavedPlan[] };

export async function createStudent(
  profile: StudentProfile,
): Promise<StudentRecord> {
  const response = await fetch(`${API_V1_URL}/students`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(profile),
  });

  if (!response.ok) {
    await throwApiError(response, "Failed to save profile");
  }

  return response.json();
}

export async function getStudent(studentId: string): Promise<StudentDetail> {
  const response = await fetch(`${API_V1_URL}/students/${studentId}`);

  if (!response.ok) {
    await throwApiError(response, "Failed to load saved profile");
  }

  return response.json();
}

// Every accepted edit/revision/regeneration is appended as a new plan row (history is
// kept server-side); `feedback` records what prompted the save.
export async function savePlan(
  studentId: string,
  plan: PlanResponse,
  feedback?: string,
): Promise<SavedPlan> {
  const response = await fetch(`${API_V1_URL}/students/${studentId}/plans`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plan, feedback: feedback ?? null }),
  });

  if (!response.ok) {
    await throwApiError(response, "Failed to save plan");
  }

  return response.json();
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
): Promise<AdvisorAskResponse> {
  const response = await fetch(`${API_V1_URL}/advisor/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    await throwApiError(response, "Failed to get advisor response");
  }

  return response.json();
}
