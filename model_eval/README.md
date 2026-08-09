# Model Evaluation Harness

Answers one question with numbers instead of vibes:

> **Which local model should BoilerAdvisor run, given that generating a viable plan of study
> is the thing students will actually use it for?**

Everything else this harness measures — grounded QA, plan explanation, structured-output
reliability — is secondary to that. The report is ordered accordingly.

The harness depends on nothing in the app (no FastAPI, no pydantic, no Postgres). It needs
Python 3.10+, PyYAML, and llama.cpp. It starts and stops `llama-server` itself.

---

## What changed, and why (read this before comparing to old results)

This used to be an Ollama text-to-SQL benchmark. Both halves of that are obsolete:

| Was | Is | Why |
|---|---|---|
| Ollama HTTP API (`/api/generate`) | llama.cpp `llama-server` (`/v1/chat/completions`) | The app moved to llama.cpp on 2026-07-21 (`backend/app/services/llamacpp_client.py`). Ollama's nanosecond timing fields, `think:` flag and `journalctl` offload parsing have no equivalent. |
| Text-to-SQL against a stub SQLite schema | Plan of study + grounded QA | **The app never asks a model for SQL.** `services/rag/pipeline.py` does retrieval itself (exact course-code lookup + pgvector cosine search) and hands the model only the chunks it already found. Measuring SQL was measuring a call site that does not exist. |
| `num_ctx` sent per request | `--ctx-size` fixed at launch | llama.cpp fixes context, KV type, GPU layers and reasoning mode when the process starts. The harness therefore owns the process — otherwise "every model saw identical settings" depends on whoever typed the last command line. |
| Offload from `journalctl -u ollama` | Offload from llama-server's own stderr | Better data: build b10083 names the device for *every layer*, so offload is counted, not inferred. Requires `-lv 5`; at default verbosity llama-server prints no offload info at all. |

The old SQL question set, stub schema and Ollama client are preserved under
`results/archive_ollama_sql/`. Old `runs_*.jsonl` files are **not** comparable to new ones —
different tasks, different prompts, different hashes.

---

## Quick start

```bash
cd model_eval

python run.py doctor              # local llama-server setup: binary, models, GPU (once, first)
python run.py refresh-offerings   # pull real term offerings into the fixture (see below)
python run.py check               # fixture validity, prompt sizes, gguf files, GPU visible
python run.py run        # every model, every task  (hours — see "Cost" below)
python run.py report     # results/report.md

# narrower runs
python run.py run --brackets 8gb,coder            # deployment candidates only
python run.py run --models qwen3.6-27b --tasks plan_b
python run.py run --mitigate --models <champion>  # the free fixes, for before/after
```

### Requirements

- Python 3.10+, `pip install pyyaml` (only dependency)
- `llama-server` built from `../llama.cpp` (`cmake -B build -DGGML_CUDA=on && cmake --build
  build --config Release -j`; lands at `llama.cpp/build/bin/llama-server`), GGUFs under
  `../models/`
- `nvidia-smi` (WSL ships it at `/usr/lib/wsl/lib/nvidia-smi`; absent = VRAM recorded as null)
- Models on **SSD** — an HDD cold-load pollutes latency and can blow the startup timeout

`llama-server` runs as a native Linux process directly in this WSL box — no Windows interop,
no cross-VM networking, so it binds plain loopback (`127.0.0.1`). `python run.py doctor`
checks the binary, the models directory, GPU visibility and whether the configured port is
already in use.

Note the eval port is **8099**, not llama.cpp's default 8080 — the `purdueio-api` container
already owns 8080 on this box.

### WSL memory (lives outside this repo)

Because the server is a native Linux process, model weights are WSL's memory, and WSL2 is slow
to hand memory back to Windows after a sweep. Two settings bound it, neither of them in this
repo:

- **`C:\Users\<you>\.wslconfig`** — `memory=40GB` caps the VM (was 56 GiB, which left Windows
  8 GiB of the box's 64 GB and bounded nothing), and `[experimental] autoMemoryReclaim=gradual`
  is what actually shrinks the VM back down once a model exits. The file carries its own
  reasoning. Editing it needs `wsl --shutdown` from PowerShell — **between** runs, not during
  one. The cap is sized against the largest GGUF in the sweep; a bigger model needs it raised
  or `mlock: false` on its config entry, or `--mlock` fails.
- **`--no-mmap`**, always passed (see `build_argv`). Weights land in anonymous memory that is
  freed at process exit rather than in a page-cache mapping that lingers, so memory comes back
  between models without a `wsl --shutdown`. The tradeoff is a real read of the GGUF on every
  launch — which is what we want anyway, since a warm-mapped model would post load times no
  deployment ever sees.

---

## Term offerings are observed, not invented

Purdue's Acalog catalog publishes no term offerings, so the fixture's `offered_terms` started
out hand-written — and were wrong for **17 of 34 courses**. CS 47300 was recorded spring-only
when it is observed fall-only; CS 47100 fall-only when it runs both. Models were being charged
term-offering violations for schedules that were legal.

PurdueIO has no "offered in" field either, but it has `Classes` — a (Course, Term) pair that
actually ran. So offerings are *inferred from sightings*:

```bash
cd ../backend && python -m app.services.offerings.sync --terms 12   # 55 terms available
cd ../model_eval && python run.py refresh-offerings
```

That populates `advisor.course_offerings` (raw observations) and
`advisor.course_offering_patterns` (per-course rollup) in the purdueio database, then rewrites
the fixture. Each row carries its observation count: **9 terms is a pattern, 3 is a hint, and
absence is the weakest signal of all** — a course seen twice says nothing reliable about the
terms it was not seen in.

Effect on the eval: term-offering violations fell from ~4.2 to 0.8 per plan, and prerequisite
ordering became the dominant failure — which is the real problem, and the one the fixture
still cannot state authoritatively (see threat 2).

## Summer is not planned into

`run.planning_terms: [fall, spring]` (mirroring the backend's `PLANNER_INCLUDE_SUMMER=false`).
Summer is a real term with real offerings, but summer enrolment has cost, aid and residency
consequences a course planner cannot reason about, so it is not recommended unprompted. This
constrains **scheduling only** — the offerings data still records summer availability, and a
course whose only offering is summer goes unplanned with a warning rather than silently
vanishing. Mode B's response grammar restricts the term enum too, so a
model cannot emit a summer semester at all. Add `summer` back to both settings when summer
planning lands.

## The plan-of-study eval

The scoring authority is `plan_fixtures/cs_machine_intelligence.yaml`: a CS BS +
Machine Intelligence catalog (43 courses with prereq edges, term offerings and credits),
9 requirement groups, and 9 student scenarios. Nothing else decides a score.

### Two modes, kept separate on purpose

**Mode A — the app's real path.** `advisor_agent.revise_plan` as shipped: the model emits a
`PlanEditProposal` (reorder / defer / avoid_tags / credit cap) under grammar-constrained
decoding, and the deterministic planner rebuilds the schedule from it. The resulting plan is
**legal by construction**, so viability is not the discriminator here. What is:

| Metric | What it catches |
|---|---|
| Proposal parsed | Did grammar-constrained decoding produce usable JSON at all |
| Touched anything | An empty proposal is a silent no-op that still returns a working plan |
| **Grounded** | Do the course codes and tags exist? The app drops unknown ones — so a model that writes prose into `reorder` looks successful and changes nothing |
| Ask honoured | Machine-checkable assertions per scenario (did the 12-credit cap actually get set, did CS 37300 actually move earlier) |
| Plan not worse | Did the revision strand more courses than the baseline |

**Mode B — the model builds the whole schedule.** No production call site does this. It is
the discriminating measurement, and it answers *what would we gain, or risk, by trusting the
model with sequencing?*

**Mode C was removed on 2026-07-30.** It ran twice: first as Mode B plus the department's
published sample plan, then as Mode B with the scorer's findings fed back for up to 3 retries.
The second version worked — it took gemma4-26b from 22% to 61% viable at 2 median iterations —
but at ~2x Mode B's latency, which does not survive being a web server with concurrent users.
Both live in git history if the tradeoff changes. The published sample plan and the
`template_slot_match` diagnostic that measured how much of it a model reproduced from memory
were removed with it on 2026-08-07 — Purdue's catalog sits behind a WAF that blocks automated
fetches, so keeping the diagnostic alive per major meant hand-pasting a page from a browser
every time, for a number that never decided a score.

### Mode C — retry to convergence (added 2026-07-31)

Modes A and B score one attempt. Mode C asks a different question: **given repeated attempts,
how many does a model need — and how long — to land a plan with no errors in it?** That is
closer to how the product is used; a student iterates rather than accepting or binning a single
plan.

Folded into `run`/`report` like every other mode — `--tasks converge` picks it out on its own,
and it is in `run.default_tasks` by default, so a plain `python run.py run` includes it.

```bash
python run.py run --tasks converge --preflight     # validate the locked slots, spend no GPU time
python run.py run --tasks converge                 # every model, all variants, all 9 scenarios
python run.py run --tasks converge --variants feedback --models qwen3.5-9b
python run.py report                                # results/report.md, Mode C section included
```

**It is Mode C, not Mode C, and that is not cosmetic.** `results_old8/runs_*.jsonl` on this box
still contains `"stage": "plan_mode_c"` records from the removed reference arm, which measured
something else entirely (Mode B plus the department's sample plan, later Mode B with scorer
feedback). The code is gone; the data is not, so the label is burned — reusing it would let two
incompatible experiments pool under one name in any analysis that globs the results
directories. That is exactly what `static_hash` and `fixture_hash` exist to prevent everywhere
else here.

**Three variants, never averaged.** They are different capabilities and one column mixing them
would describe neither:

| Variant | What changes between attempts | Measures |
|---|---|---|
| `repair` | The **frozen set** — the harness deletes violating placements itself and locks everything that validates. Seed and temperature fixed. | How many rounds of "fill the gaps" to finish a degree-complete plan. |
| `feedback` | The context — the validator's violations are appended. Seed and temperature held **fixed**. | Self-correction: can the model read a structured error and fix it? |
| `blind` | The sample — seed moves, temperature climbs. No information about the failure. | Raw variance: how often does a good plan fall out by luck? |

Holding seed fixed in the feedback arm is the point. If feedback also reseeded, self-correction
and resampling luck would be confounded, and that single comparison is the mode's reason to
exist.

**Why `repair` exists.** The `feedback` arm asks the model to delete a course the validator has
*already* identified by name, semester and reason. That is not a reasoning task — it is a
deletion the harness can do itself, correctly, in microseconds. And it costs more than a round
trip: the first live case (gemma4-e4b, `mi-fresh-start`) went `17 → 4 → 10 → 14` violations,
rewriting the whole plan every attempt and discarding good placements, because nothing anchored
what it had already got right. `repair` splits the job where the split actually is — the harness
removes, the model fills — and validated placements are frozen into the locked set so progress
ratchets instead of being re-rolled.

**The ratchet is conditional, and that correction cost a run.** Freezing on "no violation" is
too weak a test: a placement can be perfectly legal where it sits and still make the rest of the
degree unreachable. First live repair run, gemma4-e4b / `mi-fresh-start`: `CS 25200` validated at
semester 7 and froze there. It gates the whole `cs-elective` group, so every course behind it —
CS 30700, CS 35200, CS 42200, CS 42600, CS 45600, CS 49000 — had nowhere legal to go and was
deleted on each pass; `STAT 35000` froze at semester 8 and did the same to `CS 37300`. Coverage
sat at 77.8% for eight attempts and the case was recorded as the model failing. It was not —
**the harness had built the trap and then scored the model for being in it**, which is the same
class of error as a wrong fixture edge inventing a capability difference.

So when a clean plan stops improving its coverage for `repair_stall_patience` attempts, the
frozen courses standing in front of the unmet requirements are released — latest position first,
since a late prerequisite blocks the most — and the model re-places them. Releases follow the
whole prerequisite chain (freeing `CS 25200` also frees the `CS 25000` behind it), which is why
`frozen_by_attempt` may legitimately fall. The report's ratchet guard therefore reads it against
`released_by_attempt`: a fall **with** a release is the deadlock-breaker working, a fall
**without** one is lost work and a bug. Student pins are never released; if the deadlock traces
to one, the honest outcome is a plan that cannot cover everything.

**`repair` is scored strictly, and that is not a detail.** Its repair step guarantees a clean
plan by construction, so under the loose criterion every model would converge on attempt 1 and
the column would be worthless. It is scored on clean **and** 100% requirement coverage instead.
Its `attempts p50` is therefore a *different bar* from the other two variants' — do not read
them as one ranking.

**Repair never overrides the student.** Locked slots, and everything they transitively depend
on, are protected from removal. If the only way to clear a violation would be to move a pinned
course, the harness refuses and reports it in `repair_unrepairable` rather than quietly planning
over the person it is planning for.

**Locked slots.** A locked slot is a `(semester, course)` pair the student fixed. It is an input
to every attempt and the model may never move it; only the slots the model fills itself are
revisable. They live in `plan_fixtures/*.locked_slots.yaml` — deliberately **outside**
`fixture_hash`, because Mode A/B never see them and folding them in would invalidate those
records for a change they did not observe. Each is lifted
from the deterministic planner's own output, so the set is demonstrably satisfiable, and
`--preflight` re-verifies that every pin is legal where it sits.

**Censoring is the statistics.** A case that runs out of clock at attempt 7 has not shown that
the model needs 7 attempts — it has shown it needs *more than* 7. Dropping it biases the median
down (slow cases are exactly the ones that get censored); scoring it as 7 biases it up. So every
case carries `censored` / `censor_reason`, and the report uses a **Kaplan-Meier** estimator.
When too few cases converge for the survival curve to reach 0.5, the median is genuinely
undefined and prints as `>N` rather than being invented.

**The convergence criterion has a hole, on purpose, and it is measured.** As specified,
`converged` = zero prereq + credit-cap + term-offering violations over *filled* slots, with no
coverage condition. **An empty plan therefore converges on attempt 1**, and the feedback variant
actively pushes toward it — deleting the course named in a violation always removes that
violation. `converged_strict` (full PLAN_VIABLE) and `degenerate` are recorded next to it on
every attempt so this is visible instead of silently inflating the headline. Read all three, or
add the remaining violation classes to `convergence.violation_classes`.

**A bad prereq edge compounds here.** In Mode B a wrong edge costs one scored plan and shows up
as a constant offset. In Mode C the same edge is re-hit on every attempt of every case, so it
costs a model the whole run and turns into a fabricated convergence difference.
`convergence.excluded_prereq_edges` (`"CS 38100->MA 26100"` form) exists for edges known to be
wrong while they are being fixed; exclusions are printed in the report header and stored in
`meta_convergence.json`, because silently excluding evidence is how you get a benchmark nobody
can trust.

**Throughput is fenced off.** `tok/s` is reported in its own column with a warning. A model
needing 3 attempts at 23 tok/s beats one needing 1 attempt at 6 tok/s on wall-clock while being
the worse planner. Rank on `attempts p50`; quote wall-clock only about this card.

### The database is the context

Mode B used to be handed a bullet list of courses that a prompt function formatted by hand.
Nothing in production produces that format — the app feeds the model **rows out of its
databases** — so the eval was measuring a call site the app does not have.

Mode B's system prompt is now a read-only export of a mock database that mirrors the real
schemas:

| Mock table | Mirrors | Carries |
|---|---|---|
| `courses`, `programs`, `requirement_groups`, `requirement_options`, `course_aliases`, `program_notes`, `catalog_years` | `catalog_ingestion` | what Purdue's Acalog catalog publishes |
| `course_prerequisites` | `purdueio.advisor` (003) | `prereq_groups` as AND-of-ORs, `coreq_codes`, the raw Banner sentence, a `confidence` |
| `course_planner_terms` | `purdueio.advisor` (002) | `offered_terms` plus `terms_observed` — offerings are **observed**, not declared |
| `course_workload` | advisor-assigned | a 1–5 estimate that no rule depends on |

The provenance split is deliberate and it is printed in the export: `courses.prerequisites_raw`
is empty in the mock because it is empty in the real crawled table too — Acalog publishes no
prerequisites, which is the entire reason `course_prerequisites` exists as a separately-crawled
table. A model reading `confidence: high` is reading the same caveat the report carries.

**Generated, never hand-maintained.** `run.py build-mock-db` projects the mock out of the plan
fixture, so there is exactly one place to edit a prerequisite, and `run.py check` **fails** if
the two have drifted — a model shown one catalog and scored against another is the one error
this harness cannot detect from the outside. Editing `mock_db/*.json` by hand is a mistake the
next build silently reverts.

**No retrieval yet, on purpose.** `render_context()` returns the *entire* database — ~6,700
tokens for one program, which fits. A real catalog is ~10,000 courses and will not, so this
function is precisely the seam a retrieval step replaces; the prompt around it does not change.

Cost, measured: Mode B's prompt went from ~2,900 tokens to ~10,700. `max_plan_tokens` came down
to 3072 to keep a real margin inside the 16,384-token slot.

#### There is no "level" any more

Every catalog line used to open with `[level N]`, the transitive prerequisite depth. It was
doing real work — prerequisite ordering is the single largest Mode B violation class — but
"level 0" reads as *freshman year* and "level 3" as *junior year*, a constraint the number
never expressed. A model could reasonably refuse to put a level-3 course in semester 1 even for
a student who had already completed everything behind it.

The ordering information now travels two ways, neither of which names a year:

- **`courses` rows are in prerequisite order.** A course never appears above one it depends on,
  so reading the table top to bottom is already close to a legal sequence.
- **`chain_depth`** is still available in `course_prerequisites`, and `PLAN_SYSTEM` says in as
  many words that it is not a year, a class standing, or a semester number — a chain_depth 4
  course belongs in semester 1 if its four prerequisites are already complete.

### PLAN_VIABLE

A plan is viable **only if** it has zero hard violations **and** covers every requirement
group:

| Hard violation | Meaning |
|---|---|
| `prereq_violation` | Scheduled before a prerequisite completes. Same-semester does **not** count as satisfied. |
| `term_offering_violation` | Scheduled in a term the course isn't offered |
| `credit_cap_violation` | Semester exceeds `hard_credit_cap` (18), the registrar's ceiling — *not* the student's stated cap; see the credit cap split below |
| `major_overload_violation` | More than `max_major_courses_per_semester` courses from the major's own subject in one term (default: more than **3 CS courses**) |
| `hallucinated_course` | Code not in the catalog |
| `duplicate_course` | Scheduled twice, or already completed |

One prerequisite mistake anywhere in eight semesters fails the whole plan. That is the
correct standard: a student following it gets turned away at registration. `violations/plan`
is reported alongside, so a near-miss is visibly different from a plan that invented six
courses.

**Deliberately not scored:** workload balance, spreading theory courses, summer usage. Those
are preferences, not correctness, and folding them in would let a model trade a real
prerequisite violation for a prettier credit distribution. They are recorded as diagnostics.

### The CS-load cap (added 2026-07-29 — this changed `fixture_hash`)

`major_overload_violation` is the one hard violation the registrar would *not* stop you from
committing, and it is scored anyway. Reading the transcripts, models routinely put four and
five CS courses in a single semester — and so did this repo's own deterministic planner
(`mi-fresh-start` and `mi-spring-start` each had a 5-CS term, and the since-removed
`mi-tight-horizon` had 5 out of 5 courses). No student follows that plan. Set from what students actually report: two at a
time is sustainable, three is the ceiling a 4+1 BS/MS student hits fitting BS and MS courses
into the same terms, four is where a plan stops describing a real semester.

Two numbers, in the fixture's `program:` block and overridable per scenario:

| Key | Default | Effect |
|---|---|---|
| `preferred_major_courses_per_semester` | 2 | **Soft.** Over it (and within the hard cap) increments `soft_major_overloads`, reported as *Heavy CS terms*. **Not** part of PLAN_VIABLE. |
| `max_major_courses_per_semester` | 3 | **Hard.** Over it is `major_overload_violation` and the plan is not viable. |

The split is what keeps the trade honest: three CS courses is a bad term and is recorded as
one; four means the plan is wrong. Both numbers are in the prompt — the rule in the static
block, the values on the student line next to the credit limit — so no model is scored on a
constraint it was not told. The deterministic planner enforces the hard cap and respects the
soft one (filling with non-major requirements first), in **both** `harness/planner.py` and
`backend/app/services/planner.py`; `run.py parity` is what keeps those two honest.

### The credit cap split (added 2026-08-02 — this changed `fixture_hash`)

Same shape as the CS-load cap above, for the same reason, one table over. `credit_cap_violation`
used to fire off `max_credits_per_semester` — the number the *student* names in their own words
("keep every semester at 16 credits or under"). So a 17-credit term made a plan NOT VIABLE: the
same verdict, and the same weight in every ranking, as scheduling a course before its own
prerequisite. Those are not the same failure. One is a registration wall; the other is a
semester the student would have to be talked into.

It was also billed twice, because the student's number is *already* scored by name as the
`max_credits_at_most` assertion under **Ask honoured** — which is where ignoring a stated
preference belongs.

| Key | Default | Effect |
|---|---|---|
| `max_credits_per_semester` | 15–16, per scenario | **Soft.** The student's ask. Over it (and within the hard cap) increments `soft_credit_overages`, reported as *Over-ask terms*, and fails `max_credits_at_most` where a scenario asserts it. Drives the planner and the prompt's "hard limit" line. **Not** part of PLAN_VIABLE. |
| `hard_credit_cap` | 18 | **Hard.** Purdue's full-time overload ceiling; past it needs a dean's signature. Over it is `credit_cap_violation` and the plan is not viable. Program-level, in the fixture's `program:` block. |

Unlike the CS-load pair, `hard_credit_cap` is **deliberately absent from every prompt**. Models
are told the student's number and a target to aim for — `ceil(credits outstanding / semesters
available)`, printed on the student line — and naming 18 would just hand them a bigger number to
fill to, which is the front-loading this harness spent 2026-07-29 removing. It is a scoring-side
constant only.

### The courses the sample plan names, and the scenario that refuses them

**2026-07-27 — this changed `fixture_hash`. Every run recorded before that date is no longer
comparable to one recorded after.** The reason was a scoring bug: Purdue's sample plan offers
alternatives ("MA 16100 … **or** MA 16500") and suggestions ("Elective — 1.00 cr, *CS 19300
suggested*"), and the fixture had none of those nine courses — so a model that scheduled one
was charged a `hallucinated_course` for a schedule a real student could register for. Same
class of error as the hand-written `offered_terms` that `refresh-offerings` fixed. All nine
are now in the catalog, in two groups:

**Five equivalents** (MA 16500, MA 16600, MA 27101, MA 35100, STAT 51100) carry
`equivalent_to:` naming the course they substitute for. That is a **scoring** concept, handled
in `plan_scorers.py`, and it applies to all three checks that care, together:

- requirement coverage counts either course for the primary's groups,
- a prereq naming the primary is satisfied by the substitute,
- scheduling **both** is a `duplicate_course` — you cannot take both for credit.

`harness/planner.py` deliberately ignores the field. Equivalents are in no requirement group,
so `select_remaining_courses` never offers one to `generate_plan`; the deterministic plan for
all 9 scenarios is byte-identical to before, and `run.py parity` still holds against the app's
planner (which has no such field). Equivalence is also stated **in the prompt** — the catalog
block prints `SUBSTITUTE FOR MA 16100 (take one, not both)` — because scoring a model on a
rule it was never told is not measurement.

**Four suggestions** (CS 19300, CS 29100, CS 39100, COM 21700) are ordinary catalog courses
tagged `suggested-elective`. They **satisfy no requirement group**, which is the honest state:
they fill free-elective credit this fixture does not model. Two consequences worth holding on
to — scheduling them can never raise `requirement_coverage`, and `idle_credits` becomes
partially gameable, since 1-credit seminars soak up capacity without progress. Read idle
credits next to coverage, which is what it was always for.

That in turn needed a scenario, because **nothing in the fixture tested a model's willingness
to leave something out**. `mi-no-filler` is a student who names all four and refuses them:

> *"Please don't put any of the optional seminar or suggested-elective courses in my plan — no
> CS 19300, CS 29100, CS 39100 or COM 21700. I'm paying per credit hour and I only want courses
> that actually count toward my degree requirements."*

It is scored by a new `none_scheduled` assertion, and it is deliberately **not** a PLAN_VIABLE
test: a plan padded with CS 19300 is still perfectly legal, still covers every requirement, and
still fails the student. That gap is why `Ask honoured` is now a column on all three plan modes
and not just Mode A.

Read it in **Mode B, not Mode A** — in Mode A the suggested electives are never in
`remaining_courses`, so the planner cannot schedule them and the assertion passes for everyone
regardless of what the model proposed. What Mode A can still show is whether the model reached
for the `avoid_tags: [suggested-elective]` handle.

It is scored on Mode B, where the model chooses the whole schedule and can therefore choose
to include courses the student refused.

### The unsatisfiable scenario was removed on 2026-08-03

`mi-tight-horizon` (one semester left, 21 credits of requirements) could not be solved by
design. It was excluded from the headline rate — 0% for everyone tells you nothing about a
model — and scored separately on the honesty question: does the model report what didn't fit,
or fabricate a semester and blow the credit cap to make the numbers work?

**It is gone, and so is that question.** Nothing in the fixture now tests whether a model says
"this does not fit" instead of inventing room, which on a public advising site is the worse
failure of the two because it looks like a complete plan. If that behaviour starts mattering,
re-add a scenario with `expect_unsatisfiable: true` — the machinery that reads the flag is
still in `fixtures.py`, `report.py` and `run.py check`, and only the scenario body was deleted.

Note what the old row was actually worth before you trust its successor. `_unsatisfiable_section`
filtered on `not r.get("error")` and never on `structure_ok`, so a model that produced **no
parseable plan at all** scored 100% "stayed in horizon", 100% "respected credit cap" and 0.0
violations/plan — gemma4-26b's three runs on 2026-08-03 were exactly that (a repetition loop
into the token ceiling) and the table read them as exemplary. Any replacement needs that guard,
and needs a `no output` column, or it will flatter the same failure again.

All 8 remaining scenarios are satisfiable. `python run.py check` verifies the deterministic
planner *can* produce a viable plan for every one of them. If it can't, every model scores 0
for reasons unrelated to the model — which is a fixture bug, and the check catches it before
you spend GPU hours.

### How long the student waits

Every record carries `ttft_s` (wall-clock from request sent to the first content token) and
`total_s`. The report prints them in **What a student waits for**, split by call site, because
one number for both would misdescribe the product:

| Call site | What the student experiences | Read |
|---|---|---|
| `qa`, `explain` | Prose streams into the browser token by token | **TTFT** — the answer starts moving at TTFT and the total is just how long it kept growing |
| `plan_mode_a/b/c` | A grammar-constrained JSON object; half a plan renders as nothing | **Total** — TTFT here only measures prompt processing |

p50, mean and p95 are all printed for TTFT: the mean is the "average wait" but it is dragged
around by the occasional slow request, so the median leads and p95 is what the unlucky
student gets. Mode A records the TTFT of its **first** request only — under `--mitigate` it
can make several round trips, but the student's wait starts once.

All of it excludes model load. The warmup generations run before anything is measured and are
discarded, so these are warm-server numbers; a student who hits a cold server also waits for
the weights to page in from disk, which for the larger ggufs is minutes, not seconds.

---

## Files

| File | Responsibility |
|---|---|
| `config.yaml` | Every knob that could pollute a comparison: context size, KV type, GPU layers, sampling, per-model gguf paths and `think`/`mlock` flags, decision thresholds. |
| `plan_fixtures/cs_machine_intelligence.yaml` | **The scoring authority.** Catalog, requirement groups, scenarios. Read its provenance header before quoting any number. The one place to edit a course, a prerequisite or a requirement. |
| `mock_db/*.json` | **What the model is shown.** A mock of the two real databases in their real table shapes, GENERATED from the fixture by `run.py build-mock-db` — never hand-edited. See *The database is the context* below. |
| `harness/mock_db.py` | Loads the mock database, checks the referential integrity a real FK would enforce, and renders the whole thing as Mode B's context. The seam RAG replaces. |
| `questions.yaml` | Grounded-QA items with their retrieved chunks **pinned in the file** — retrieval belongs to the embedding model, so letting it vary would smear a retrieval difference across every model's score. |
| `harness/server.py` | llama-server lifecycle as a native local process: builds argv from config, waits for `/health`, parses per-layer offload from stderr, stops between models. |
| `harness/llamacpp_client.py` | Streaming stdlib client for `/v1/chat/completions`. TTFT from the stream, token counts from `usage`, timings from llama.cpp's own `timings`. |
| `harness/planner.py` | **Vendored copy** of the app's deterministic planner + `_apply_proposal`. Mode A can only measure production if these match — see the drift warning below. |
| `harness/fixtures.py` | Loads the fixture; ports `planner_catalog.select_remaining_courses` so Mode A starts from production's baseline. |
| `harness/plan_scorers.py` | Viability, requirement coverage, scenario assertions, proposal groundedness — against the fixture. Scores Mode A (legal by construction, by design) and Mode C (see the known-gap note on `convergence.py`, below). Not used for Mode B any more. |
| `harness/real_scoring.py` | Mode B's scorer, and the only one it uses: every check `plan_scorers.py` runs, against `ctx.database` (the real, live catalog Mode B was actually shown and grammar-constrained to) instead of the fixture. Exists because scoring Mode B against the fixture produced real courses flagged `hallucinated_course` and invented gen-ed groups counted as missing requirements — see its module docstring. |
| `harness/prompts.py` | Static-first prompts for the app's four live call sites. sha256 of each static block **and its response schema** travels with every record. |
| `harness/scorers.py` | JSON extraction, faithfulness/recall heuristics, abstention detection. |
| `harness/runner.py` | Orchestration: server lifecycle, warmup + discard, VRAM delta, N replicates, one request at a time. Plan tasks run **first** so a cut-short run keeps the data that matters. |
| `harness/convergence.py` | **Mode C.** Locked slots, the retry loop, both variants, censoring. Computes convergence *from* `plan_scorers` output rather than reimplementing it, so D and B validate against one instrument. Own results file, own lock, and it refuses to start while any other run holds the GPU. **KNOWN GAP** (2026-08-04): Mode C shows the model `ctx.database` (real, grammar-constrained) exactly like Mode B does, but still scores it against the fixture, same class of bug `real_scoring.py` exists to fix for Mode B — not yet ported here because the retry/repair logic is coupled to `plan_scorers.score_plan`'s specific violation objects (the `repair` variant mutates them directly). Lower-risk in practice for the CS Machine Intelligence program specifically, because the convergence criterion is violations-only, not coverage (see `run.py run --tasks converge`'s report section) — the gen-ed-mismatch class of bug that hit Mode B's coverage number doesn't move this one. Still fixture-scored; treat accordingly. |
| `harness/convergence_report.py` | Mode C's tables and Kaplan-Meier estimator over right-censored data; progress traces per attempt. Own module because the record schema is a survival observation, not pass/fail — but `report.py` imports `mode_c_lines()` and splices it into the one `results/report.md`, not a separate file. |
| `plan_fixtures/*.locked_slots.yaml` | Mode C's locked slots. Prompt input, **not** scoring authority — recorded in `meta_convergence.json` and on each record, never folded into `fixture_hash`. |
| `harness/report.py` | Plan tables first, validity guards, manual review queue. No composite score exists. |
| `results/transcripts/` | One markdown file per (model, stage, item, replicate): system prompt, user prompt, raw output, verdict. The raw text is in the JSONL too, but a JSONL field is not something you can read — and reading what a model literally said is how you catch a metric measuring the wrong thing. Disable with `run.save_transcripts: false`. |
| `setup/allow_wsl_llamacpp.ps1` | Leftover from the Windows-interop era; not needed now that llama-server runs natively in WSL. |

Data flow: `plan_fixtures/*.yaml` + `questions.yaml` → `prompts.py` → `server.py` +
`llamacpp_client.py` → `plan_scorers.py` / `scorers.py` / `convergence.py` →
`results/runs_*.jsonl` → `report.py` (pulling in `convergence_report.mode_c_lines()`) →
`results/report.md` + `results/review_queue.jsonl`.

---

## What is automatic vs. your judgment (honestly)

| Metric | Status |
|---|---|
| **PLAN_VIABLE** | **Automatic and decidable.** Prereq order, term offerings, credit caps, catalog membership, requirement coverage — all checked against the fixture. The only judgment baked in is the fixture itself. |
| **Requirement coverage** | **Automatic.** Fraction of requirement groups satisfied. |
| **Proposal groundedness** | **Automatic.** Do the named codes/tags exist in the student's actual course list. |
| **Ask honoured** | **Automatic**, but only for what a scenario declares as a machine-checkable assertion. "Did it understand the student" in the broad sense is not measured. |
| **Structure OK** | **Automatic.** Under grammar-constrained decoding this should be ~100%; anything less is a real signal (truncation, a template mismatch, or a model that ignores the grammar). |
| **Behavior OK (QA)** | **Automatic-ish.** Abstention is a phrase heuristic. A polite correction and a polite refusal look the same to it — read the adversarial items. |
| **FAITHFULNESS** | **Manual, full stop.** The heuristic catches *entity* hallucination only (codes/numbers absent from the context). Relational hallucination ("X must come before Y") is not machine-checkable without a judge model, which is deliberately not here — judging small models with another model smuggles in a second unvalidated instrument. Quote your manual grade, not the flag rate. |
| **LATENCY** | Automatic. Tiebreaker only, and **box-specific**. As of 2026-07-27 this box is an **RTX 2060 SUPER (8 GB)** running the native WSL build; earlier numbers in `results_*/` were taken on a 5070 Ti (16 GB) over Windows interop. Latency does not transfer between them — different architecture, bandwidth and thermals — and on an 8 GB card neither does the *offload fraction*, so re-read layers-offloaded from the env row rather than assuming. Only quality scores transfer between boxes. |

---

## Validity threats already defended against

- One llama-server command line, built from config, applied identically to every model — and
  `/props` is read back so "every model ran at `num_ctx`" is **verified**, not assumed
- GPU offload counted per layer from llama-server's own stderr; `unverified` says so
- Warmup generations run and discarded; VRAM measured as an `nvidia-smi` delta after warmup
- Server stopped between models so the next VRAM baseline is clean
- Reasoning disabled at launch (`--reasoning off`) plus a per-request template kwarg, and any
  leaked `<think>` block is stripped before scoring
- Static prompt block hashed into every record; the report refuses to pool mixed hashes
- The fixture is hashed into every plan record — editing it invalidates prior results by design
- `run.py check` proves a viable plan is reachable before you spend GPU hours
- A model listed twice in `config.yaml` is flagged (it would silently pool two conditions)
- Retrieval is pinned in `questions.yaml`, so QA scores can't drift with the vector store

## Threats that remain

1. **Statistical power.** 9 scenarios is the real sample size; the 3 replicates are correlated
   repeats of the same item, not independent samples. A binomial 95% CI at n=9 is roughly
   ±33pp. If two models differ by less than that, the correct response is **write more
   scenarios**, not pick a winner. `decision.min_plan_scenarios` encodes this and the report
   enforces it. This applied with full force to the old Mode C − Mode B delta, which was a
   difference of two such estimates.
2. **Prerequisite edges are still hand-written.** Term offerings are now observed (above), but
   prereqs are not, and they are now the dominant failure mode — so this is the biggest
   remaining threat to the plan numbers. Purdue publishes prerequisites **only** in Banner
   (`selfservice.mypurdue.purdue.edu`, e.g. *"Undergraduate level CS 25000 Minimum Grade of C
   and Undergraduate level CS 25100 Minimum Grade of C"*), and that host's `robots.txt` is a
   blanket `User-agent: * / Disallow: /`. Automated collection is therefore off the table
   without Purdue's say-so. A wrong edge biases every model the same direction, so *rankings*
   survive it and *absolute* claims do not.
3. **Planner drift.** `harness/planner.py` is a copy. If the app's planner changes and this
   one doesn't, Mode A silently stops measuring production. `run.py parity` diffs them
   whenever the backend is importable — run it before trusting a Mode A column.
4. **Prompt-template fit.** One prompt fits some models' instruction tuning better than
   others; part of any measured gap is prompt compatibility, not capacity.
5. **Quantization confound.** Q4_K_M doesn't damage all architectures equally — you are
   comparing (model × quant) pairs. Acceptable, since you'd deploy Q4_K_M anyway, but say
   "quantized model" in anything you write down.

---

## Cost

One model × 9 scenarios × 3 replicates × 4 tasks ≈ 130 generations (
a whole extra schedule per scenario, on the longest prompt in the harness). On the 5070 Ti a 7B model
finished in ~5 minutes; on this box's 2060 SUPER expect longer, and expect anything above ~8 GB
to spill to CPU; a 120B MoE with CPU expert offload is far slower and its **load** time
alone can be minutes. Running all 15 models is an overnight job. Start with
`--brackets 8gb,coder` — those are the only real deployment candidates; `reference` models
exist to show what you're giving up, not what you could ship.


## Pre-registered decision rule

Thresholds live in `config.yaml` under `decision:`. The question is no longer "is a bigger GPU
worth it" in the abstract — it is **can an 8GB-class model be trusted with the plan-of-study
feature**:

1. `mode_a_grounded_floor` (0.90) — a model that can't emit a grounded `PlanEditProposal`
   isn't a candidate, regardless of how it scores elsewhere. This is the shipped path.
2. `mode_b_viable_floor` (0.90) — below this, keep the model out of sequencing entirely and
   stay on today's architecture (model proposes, planner disposes). Today's architecture is
   the *default*; Mode B has to earn its way in.
3. `plan_viable_gap_pp` (15) — reference-tier minus best deployable. Below it, the ceiling
   isn't worth chasing with hardware.

**Explicitly insufficient to justify hardware:** any latency number (a number from one card is
not evidence about another, in either direction), and any gap that closes under `--mitigate` (then the fix costs
$0). A faithfulness failure *can* justify action alone — a hallucinated answer gets cached by
the semantic cache and repeated.

**There is no Mode C threshold** — the arm was removed (see above). Mode A and Mode B carry
the decision on their own floors.