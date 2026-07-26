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
vanishing. Mode B's response grammar restricts the term enum too, so a model cannot emit a
summer semester at all. Add `summer` back to both settings when summer planning lands.

## The plan-of-study eval

The scoring authority is `plan_fixtures/cs_machine_intelligence.yaml`: a CS BS +
Machine Intelligence catalog (34 courses with prereq edges, term offerings and credits),
9 requirement groups, and 8 student scenarios. Nothing else decides a score.

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

#### Tried and dropped: a "Mode C" with the published sample plan

Mode B's failures are almost entirely prerequisite *ordering*, and Purdue publishes a sample
four-year plan per major — which is exactly a known-good topological order with term labels
on it. So an arm was built that added that plan to Mode B's prompt, changing nothing else,
and run on gemma4-26b and gemma4-e4b (3 replicates, 8 scenarios). **It did not pay for
itself, and it is gone.** Recorded here so nobody rebuilds it:

- Viability did not improve. gemma4-26b went 43% → 29%; gemma4-e4b stayed at 0%.
- Models anchored on the template. The share of courses placed in the sample plan's own
  semester slot rose on the scenarios the plan *doesn't* describe — `mi-light-load`, the
  12-credit-cap student, went 15% → 53% — and term-offering violations rose with it.
- The one clear win was the small model filling its schedule (coverage 75% → 92%, idle
  credits 31 → 6), which is not what the arm was built to buy.
- Only one of eight scenarios is the full-time fall-start student a published plan describes,
  so most of the fixture asks the model to *transform* the template, not reuse it.

The upkeep was the deciding factor: catalog.purdue.edu sits behind an AWS WAF JS challenge
(automated GET returns HTTP 202 + interstitial), so the plan cannot be fetched by the
ingester and has to be hand-saved from a browser per major. And a sample plan written by hand
instead of scraped agrees with the fixture's hand-written prereq edges by construction, which
makes the arm measure copying rather than planning. Real hoops, no benefit.

The run that produced these numbers is kept under `results_old5/`.

### PLAN_VIABLE

A plan is viable **only if** it has zero hard violations **and** covers every requirement
group:

| Hard violation | Meaning |
|---|---|
| `prereq_violation` | Scheduled before a prerequisite completes. Same-semester does **not** count as satisfied. |
| `term_offering_violation` | Scheduled in a term the course isn't offered |
| `credit_cap_violation` | Semester exceeds the student's cap |
| `hallucinated_course` | Code not in the catalog |
| `duplicate_course` | Scheduled twice, or already completed |

One prerequisite mistake anywhere in eight semesters fails the whole plan. That is the
correct standard: a student following it gets turned away at registration. `violations/plan`
is reported alongside, so a near-miss is visibly different from a plan that invented six
courses.

**Deliberately not scored:** workload balance, spreading theory courses, summer usage. Those
are preferences, not correctness, and folding them in would let a model trade a real
prerequisite violation for a prettier credit distribution. They are recorded as diagnostics.

### One scenario is unsatisfiable on purpose

`mi-tight-horizon` (one semester left, 21 credits of requirements) cannot be solved. It is
excluded from the headline rate — 0% for everyone tells you nothing about a model — and
scored separately on the honesty question: does the model report what didn't fit, or
fabricate a semester and blow the credit cap to make the numbers work? On a public advising
site the second failure is far worse, because it looks like a complete plan.

`python run.py check` verifies the deterministic planner *can* produce a viable plan for
every other scenario. If it can't, every model scores 0 for reasons unrelated to the model —
which is a fixture bug, and the check catches it before you spend GPU hours.

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
| `plan_fixtures/cs_machine_intelligence.yaml` | **The scoring authority.** Catalog, requirement groups, scenarios. Read its provenance header before quoting any number. |
| `questions.yaml` | Grounded-QA items with their retrieved chunks **pinned in the file** — retrieval belongs to the embedding model, so letting it vary would smear a retrieval difference across every model's score. |
| `harness/server.py` | llama-server lifecycle as a native local process: builds argv from config, waits for `/health`, parses per-layer offload from stderr, stops between models. |
| `harness/llamacpp_client.py` | Streaming stdlib client for `/v1/chat/completions`. TTFT from the stream, token counts from `usage`, timings from llama.cpp's own `timings`. |
| `harness/planner.py` | **Vendored copy** of the app's deterministic planner + `_apply_proposal`. Mode A can only measure production if these match — see the drift warning below. |
| `harness/fixtures.py` | Loads the fixture; ports `planner_catalog.select_remaining_courses` so Mode A starts from production's baseline. |
| `harness/plan_scorers.py` | Viability, requirement coverage, scenario assertions, proposal groundedness. All automatic and decidable. |
| `harness/prompts.py` | Static-first prompts mirroring the app's four live call sites. sha256 of each static block travels with every record. |
| `harness/scorers.py` | JSON extraction, faithfulness/recall heuristics, abstention detection. |
| `harness/runner.py` | Orchestration: server lifecycle, warmup + discard, VRAM delta, N replicates. Plan tasks run **first** so a cut-short run keeps the data that matters. |
| `harness/report.py` | Plan tables first, validity guards, manual review queue. No composite score exists. |
| `results/transcripts/` | One markdown file per (model, stage, item, replicate): system prompt, user prompt, raw output, verdict. The raw text is in the JSONL too, but a JSONL field is not something you can read — and reading what a model literally said is how you catch a metric measuring the wrong thing. Disable with `run.save_transcripts: false`. |
| `setup/allow_wsl_llamacpp.ps1` | Leftover from the Windows-interop era; not needed now that llama-server runs natively in WSL. |

Data flow: `plan_fixtures/*.yaml` + `questions.yaml` → `prompts.py` → `server.py` +
`llamacpp_client.py` → `plan_scorers.py` / `scorers.py` → `results/runs_*.jsonl` →
`report.py` → `results/report.md` + `results/review_queue.jsonl`.

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
| **LATENCY** | Automatic. Tiebreaker only. 5070 Ti latency says **nothing** about 2060 Super latency — different architecture, bandwidth and thermals. Only quality scores transfer between boxes. |

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

1. **Statistical power.** 8 scenarios is the real sample size; the 3 replicates are correlated
   repeats of the same item, not independent samples. A binomial 95% CI at n=8 is roughly
   ±35pp. If two models differ by less than that, the correct response is **write more
   scenarios**, not pick a winner. `decision.min_plan_scenarios` encodes this and the report
   enforces it.
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

One model × 8 scenarios × 3 replicates × 4 tasks ≈ 130 generations. On the 5070 Ti a 7B model
finishes in ~5 minutes; a 120B MoE with CPU expert offload is far slower and its **load** time
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

**Explicitly insufficient to justify hardware:** any latency number (5070 Ti latency is not
evidence about the 2060 Super), and any gap that closes under `--mitigate` (then the fix costs
$0). A faithfulness failure *can* justify action alone — a hallucinated answer gets cached by
the semantic cache and repeated.
