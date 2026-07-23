# qwen3-8b · plan_mode_a · mi-ai-early · run 0

**Verdict:** grounded=True viable=True assertions={'scheduled_before:CS 37300<2': True, 'any_scheduled_before:CS 47100|CS 47300<2': True}

## System prompt

```
You are an assistant that tunes a college course plan from a student's feedback. You are NOT an official advisor and you must not invent courses, prerequisites, or requirements. A deterministic planner owns legality (prerequisites, term offerings, credit caps); you only express preferences over the courses already listed. Only use course codes and tags that appear in the context. Leave a list empty if it does not apply. Never put a course in both reorder and defer.
```

## User prompt

```
STUDENT: Riley — Computer Science, BS — Machine Intelligence
Current credit cap: 16 per semester.

COURSES THAT CAN BE MOVED:
- CS 25200 "Systems Programming" (4 cr; tags: required, cs-core, systems, project-heavy)
- CS 38100 "Introduction To The Analysis Of Algorithms" (3 cr; tags: required, cs-core, theory-heavy)
- CS 37300 "Data Mining And Machine Learning" (3 cr; tags: required, machine-intelligence, ai, project-heavy)
- MA 26500 "Linear Algebra" (3 cr; tags: required, mathematics)
- PHYS 17200 "Modern Mechanics" (4 cr; tags: required, science)
- PHYS 24100 "Electricity And Optics" (3 cr; tags: required, science)
- COM 11400 "Fundamentals Of Speech Communication" (3 cr; tags: required, gen-ed, oral-communication)
- SCLA 10100 "Transformative Texts: Critical Thinking And Communication I" (3 cr; tags: required, gen-ed, humanities)
- CS 47100 "Introduction To Artificial Intelligence" (3 cr; tags: machine-intelligence, ai)
- CS 43900 "Introduction To Data Visualization" (3 cr; tags: machine-intelligence, ai, project-heavy)
- CS 44800 "Introduction To Relational Database Systems" (3 cr; tags: machine-intelligence, systems, project-heavy)
- CS 35400 "Operating Systems" (3 cr; tags: cs-elective, systems, project-heavy)
- CS 49000 "Topics In Computer Science For Undergraduates" (3 cr; tags: cs-elective)
- PHIL 15000 "Principles Of Logic" (3 cr; tags: gen-ed, humanities)
- ECON 25100 "Microeconomics" (3 cr; tags: gen-ed, social-science)

CURRENT PLAN:
- fall 2026: CS 25200, CS 38100, CS 37300, MA 26500, COM 11400 [16 cr]
- spring 2027: PHYS 17200, SCLA 10100, CS 47100, CS 44800, CS 35400 [16 cr]
- fall 2027: PHYS 24100, CS 43900, CS 49000, PHIL 15000, ECON 25100 [15 cr]

STUDENT FEEDBACK:
I've finished the intro sequence and I want CS 37300 and the AI course in my first two semesters if the prerequisites are already satisfied.
```

## Model output

```
{
  "rationale": "The student wants to complete CS 37300 and an AI course in their first two semesters if prerequisites are satisfied. CS 37300 is already in Fall 2026, which is appropriate. The AI course could be CS 47100, which is in Spring 2027. No changes are needed to the current plan to satisfy this request.",
  "reorder": [],
  "defer": [],
  "avoid_tags": []
}
```
