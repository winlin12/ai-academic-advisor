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
