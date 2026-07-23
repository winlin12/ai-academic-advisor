# qwen3-8b · plan_mode_b · mi-sophomore-catchup · run 0

**Verdict:** viable=False coverage=89% violations={'credit_cap_violation': 2, 'prereq_violation': 1}

## System prompt

```
You are an academic planning assistant building a semester-by-semester plan of study for an
undergraduate student. You are not an official advisor.

Build the plan ONLY from the course catalog given to you. Every rule below is hard:

- Never schedule a course that is not in the catalog. Never invent a course code.
- A course may only be scheduled in a semester whose term appears in its "terms" list.
- Every prerequisite of a course must be scheduled in an EARLIER semester, or already be in
  the student's completed list. Taking a course in the same semester as its prerequisite does
  not satisfy it.
- No semester may exceed the student's credit limit.
- Never schedule a course the student has already completed, and never schedule the same
  course twice.
- Only use the terms listed in the student's calendar below. Summer is NOT available for
  planning even when a course is offered in it.
- Cover every degree requirement listed. For a "choose" requirement, schedule enough of its
  options to reach the stated credits.

If the requirements cannot all fit in the number of semesters available, still produce the
best legal plan you can and list what did not fit in "unplanned". A short legal plan is
better than a long illegal one.

COURSE CATALOG:
- COM 11400 "Fundamentals Of Speech Communication" | 3 cr | terms: fall/spring/summer | prereqs: none | tags: required, gen-ed, oral-communication
- CS 18000 "Problem Solving And Object-Oriented Programming" | 4 cr | terms: fall/spring/summer | prereqs: none | tags: required, cs-core, programming
- CS 18200 "Foundations Of Computer Science" | 3 cr | terms: fall/spring/summer | prereqs: CS 18000, MA 16100 | tags: required, cs-core, theory-heavy
- CS 24000 "Programming In C" | 3 cr | terms: fall/spring/summer | prereqs: CS 18000 | tags: required, cs-core, programming
- CS 25000 "Computer Architecture" | 4 cr | terms: fall/spring/summer | prereqs: CS 18200, CS 24000 | tags: required, cs-core, systems
- CS 25100 "Data Structures And Algorithms" | 3 cr | terms: fall/spring/summer | prereqs: CS 18200, CS 24000 | tags: required, cs-core, theory-heavy
- CS 25200 "Systems Programming" | 4 cr | terms: fall/spring/summer | prereqs: CS 25000, CS 25100 | tags: required, cs-core, systems, project-heavy
- CS 35200 "Compilers: Principles And Practice" | 3 cr | terms: fall/spring | prereqs: CS 25200 | tags: cs-elective, systems, project-heavy
- CS 35400 "Operating Systems" | 3 cr | terms: fall/spring | prereqs: CS 25200 | tags: cs-elective, systems, project-heavy
- CS 37300 "Data Mining And Machine Learning" | 3 cr | terms: fall/spring | prereqs: CS 25100, STAT 35000 | tags: required, machine-intelligence, ai, project-heavy
- CS 38100 "Introduction To The Analysis Of Algorithms" | 3 cr | terms: fall/spring | prereqs: CS 25100, MA 26100 | tags: required, cs-core, theory-heavy
- CS 42200 "Computer Networks" | 3 cr | terms: fall/spring | prereqs: CS 25200 | tags: cs-elective, systems
- CS 42600 "Computer Security" | 3 cr | terms: fall/spring | prereqs: CS 25200 | tags: cs-elective, systems
- CS 43900 "Introduction To Data Visualization" | 3 cr | terms: fall | prereqs: CS 25100 | tags: machine-intelligence, ai, project-heavy
- CS 44000 "Large Scale Data Analytics" | 3 cr | terms: fall/spring | prereqs: CS 37300 | tags: cs-elective, ai, project-heavy
- CS 44800 "Introduction To Relational Database Systems" | 3 cr | terms: fall/spring | prereqs: CS 25100 | tags: machine-intelligence, systems, project-heavy
- CS 45600 "Programming Languages" | 3 cr | terms: fall/spring | prereqs: CS 25200 | tags: machine-intelligence, theory-heavy
- CS 45800 "Introduction To Robotics" | 3 cr | terms: fall/spring | prereqs: CS 25100, MA 26500 | tags: machine-intelligence, ai, project-heavy
- CS 47100 "Introduction To Artificial Intelligence" | 3 cr | terms: fall/spring | prereqs: CS 25100 | tags: machine-intelligence, ai
- CS 47300 "Web Information Search And Management" | 3 cr | terms: fall | prereqs: CS 25100 | tags: machine-intelligence, ai
- CS 48300 "Introduction To The Theory Of Computation" | 3 cr | terms: fall/spring | prereqs: CS 38100 | tags: machine-intelligence, theory-heavy
- CS 49000 "Topics In Computer Science For Undergraduates" | 3 cr | terms: fall/spring/summer | prereqs: CS 25200 | tags: cs-elective
- ECON 25100 "Microeconomics" | 3 cr | terms: fall/spring/summer | prereqs: none | tags: gen-ed, social-science
- ENGL 10600 "First-Year Composition" | 4 cr | terms: fall/spring/summer | prereqs: none | tags: required, gen-ed, writing
- MA 16100 "Plane Analytic Geometry And Calculus I" | 5 cr | terms: fall/spring/summer | prereqs: none | tags: required, mathematics
- MA 16200 "Plane Analytic Geometry And Calculus II" | 5 cr | terms: fall/spring/summer | prereqs: MA 16100 | tags: required, mathematics
- MA 26100 "Multivariate Calculus" | 4 cr | terms: fall/spring/summer | prereqs: MA 16200 | tags: required, mathematics
- MA 26500 "Linear Algebra" | 3 cr | terms: fall/spring/summer | prereqs: MA 16200 | tags: required, mathematics
- PHIL 15000 "Principles Of Logic" | 3 cr | terms: fall/spring/summer | prereqs: none | tags: gen-ed, humanities
- PHYS 17200 "Modern Mechanics" | 4 cr | terms: fall/spring/summer | prereqs: MA 16100 | tags: required, science
- PHYS 24100 "Electricity And Optics" | 3 cr | terms: fall/spring/summer | prereqs: PHYS 17200, MA 16200 | tags: required, science
- PSY 12000 "Elementary Psychology" | 3 cr | terms: fall/spring/summer | prereqs: none | tags: gen-ed, social-science
- SCLA 10100 "Transformative Texts: Critical Thinking And Communication I" | 3 cr | terms: fall/spring/summer | prereqs: none | tags: required, gen-ed, humanities
- STAT 35000 "Introduction To Statistics" | 3 cr | terms: fall/spring/summer | prereqs: MA 16200 | tags: required, mathematics, statistics

DEGREE REQUIREMENTS:
- Computer Science Core: take ALL of CS 18000, CS 18200, CS 24000, CS 25000, CS 25100, CS 25200, CS 38100
- Machine Intelligence — required: take ALL of CS 37300
- Machine Intelligence — AI course (choose one): choose 3 credits from CS 47100, CS 47300
- Machine Intelligence — selectives (choose two): choose 6 credits from CS 43900, CS 44800, CS 45600, CS 45800, CS 48300, CS 47100, CS 47300
- Additional CS courses at the 400 level: choose 6 credits from CS 35400, CS 49000, CS 42200, CS 42600, CS 44000, CS 35200
- Mathematics And Statistics: take ALL of MA 16100, MA 16200, MA 26100, MA 26500, STAT 35000
- Laboratory Science Sequence: take ALL of PHYS 17200, PHYS 24100
- University Core — written, oral, humanities: take ALL of ENGL 10600, COM 11400, SCLA 10100
- University Core — additional selectives: choose 6 credits from PHIL 15000, ECON 25100, PSY 12000
```

## User prompt

```
STUDENT: Jordan — Computer Science, BS — Machine Intelligence
Already completed: CS 18000, CS 18200, CS 24000, MA 16100, MA 16200, ENGL 10600
Credit limit per semester: 15
Semesters available (6), in order:
  fall 2026 -> spring 2027 -> fall 2027 -> spring 2028 -> fall 2028 -> spring 2029

STUDENT'S REQUEST:
I just switched into the machine intelligence concentration and I have three years left. I care most about finishing the CS core on time so the AI electives aren't blocked, and I'd rather not stack two project-heavy courses in the same semester.

Produce the plan of study.
```

## Model output

```
{
  "semesters": [
    {
      "term": "fall",
      "year": 2026,
      "courses": [
        "CS 25000",
        "CS 25100",
        "MA 26100",
        "PHYS 17200",
        "SCLA 10100"
      ]
    },
    {
      "term": "spring",
      "year": 2027,
      "courses": [
        "CS 25200",
        "CS 38100",
        "MA 26500",
        "PHYS 24100",
        "ECON 25100"
      ]
    },
    {
      "term": "fall",
      "year": 2027,
      "courses": [
        "CS 37300",
        "STAT 35000",
        "PHIL 15000",
        "PSY 12000"
      ]
    },
    {
      "term": "spring",
      "year": 2028,
      "courses": [
        "CS 45600",
        "CS 45800",
        "CS 47100"
      ]
    },
    {
      "term": "fall",
      "year": 2028,
      "courses": [
        "CS 44000",
        "CS 42200",
        "CS 42600"
      ]
    },
    {
      "term": "spring",
      "year": 2029,
      "courses": [
        "CS 49000"
      ]
    }
  ],
  "unplanned": [
    "CS 43900",
    "CS 44800",
    "CS 48300"
  ]
}
```
