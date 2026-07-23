# qwen3-8b · plan_mode_a · mi-fresh-start · run 0

**Verdict:** grounded=True viable=True assertions={'scheduled_before:CS 37300<6': True, 'max_credits_at_most:16': True}

## System prompt

```
You are an assistant that tunes a college course plan from a student's feedback. You are NOT an official advisor and you must not invent courses, prerequisites, or requirements. A deterministic planner owns legality (prerequisites, term offerings, credit caps); you only express preferences over the courses already listed. Only use course codes and tags that appear in the context. Leave a list empty if it does not apply. Never put a course in both reorder and defer.
```

## User prompt

```
STUDENT: Avery — Computer Science, BS — Machine Intelligence
Current credit cap: 16 per semester.

COURSES THAT CAN BE MOVED:
- CS 18000 "Problem Solving And Object-Oriented Programming" (4 cr; tags: required, cs-core, programming)
- CS 18200 "Foundations Of Computer Science" (3 cr; tags: required, cs-core, theory-heavy)
- CS 24000 "Programming In C" (3 cr; tags: required, cs-core, programming)
- CS 25000 "Computer Architecture" (4 cr; tags: required, cs-core, systems)
- CS 25100 "Data Structures And Algorithms" (3 cr; tags: required, cs-core, theory-heavy)
- CS 25200 "Systems Programming" (4 cr; tags: required, cs-core, systems, project-heavy)
- CS 38100 "Introduction To The Analysis Of Algorithms" (3 cr; tags: required, cs-core, theory-heavy)
- CS 37300 "Data Mining And Machine Learning" (3 cr; tags: required, machine-intelligence, ai, project-heavy)
- MA 16100 "Plane Analytic Geometry And Calculus I" (5 cr; tags: required, mathematics)
- MA 16200 "Plane Analytic Geometry And Calculus II" (5 cr; tags: required, mathematics)
- MA 26100 "Multivariate Calculus" (4 cr; tags: required, mathematics)
- MA 26500 "Linear Algebra" (3 cr; tags: required, mathematics)
- STAT 35000 "Introduction To Statistics" (3 cr; tags: required, mathematics, statistics)
- PHYS 17200 "Modern Mechanics" (4 cr; tags: required, science)
- PHYS 24100 "Electricity And Optics" (3 cr; tags: required, science)
- ENGL 10600 "First-Year Composition" (4 cr; tags: required, gen-ed, writing)
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
- fall 2026: CS 18000, MA 16100, ENGL 10600, COM 11400 [16 cr]
- spring 2027: CS 18200, CS 24000, MA 16200, PHYS 17200 [15 cr]
- fall 2027: CS 25000, CS 25100, MA 26100, MA 26500 [14 cr]
- spring 2028: CS 25200, CS 38100, STAT 35000, PHYS 24100, SCLA 10100 [16 cr]
- fall 2028: CS 37300, CS 47100, CS 43900, CS 44800, CS 35400 [15 cr]
- spring 2029: CS 49000, PHIL 15000, ECON 25100 [9 cr]

STUDENT FEEDBACK:
I'm starting from scratch and I want to concentrate in machine intelligence. Please get me into the AI and machine learning courses as early as the prerequisites allow, and keep every semester at 16 credits or under.
```

## Model output

```
{
  "rationale": "The plan has been adjusted to prioritize machine intelligence courses as early as prerequisites allow, while maintaining the 16-credit cap per semester.",
  "reorder": [
    "CS 37300",
    "CS 47100",
    "CS 43900",
    "CS 44800",
    "CS 35400"
  ],
  "defer": [
    "CS 38100"
  ],
  "avoid_tags": [
    "gen-ed"
  ]
}
```
