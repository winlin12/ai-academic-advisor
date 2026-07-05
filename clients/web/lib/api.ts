const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type StudentProfile = {
  name: string;
  degree_program: string;
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

export type ExplainPlanResponse = {
  answer: string;
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

export async function generatePlan(
  profile: StudentProfile,
): Promise<PlanResponse> {
  const response = await fetch(`${API_BASE_URL}/plan/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(profile),
  });

  if (!response.ok) {
    throw new Error(`Failed to generate plan: ${response.status}`);
  }

  return response.json();
}

export async function explainPlan(
  question: string,
  plan: PlanResponse,
): Promise<ExplainPlanResponse> {
  const response = await fetch(`${API_BASE_URL}/advisor/explain-plan`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question, plan }),
  });

  if (!response.ok) {
    throw new Error(`Failed to get advisor response: ${response.status}`);
  }

  return response.json();
}

// Free-text question to the local advisor model. The backend attaches a fixed catalog
// context (all degrees + requirements + referenced courses) to every question, so the
// answer is grounded in the real database regardless of what the student asks.
export async function askAdvisor(
  question: string,
): Promise<AdvisorAskResponse> {
  const response = await fetch(`${API_BASE_URL}/advisor/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    throw new Error(`Failed to get advisor response: ${response.status}`);
  }

  return response.json();
}
