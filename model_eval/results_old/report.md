# Model evaluation — BoilerAdvisor

Fixture `cs_machine_intelligence.yaml` (hash `1ae6820982467b77`, 34 courses, 9 requirement groups)  
Run 2026-07-22T22:35:00 · tasks ['explain', 'plan_a', 'plan_b', 'qa'] · num_ctx 8192 · temp 0.15 · seed 67 · 3 replicates

## 1. Plan of study — the feature this is really about

A plan is VIABLE only if it has zero hard violations (prerequisite ordering, term offerings, credit cap, hallucinated course codes, duplicates) **and** covers every degree requirement. One prerequisite mistake anywhere in eight semesters fails the whole plan. That is the correct standard: a student following it gets turned away at registration.

### Mode B — the model builds the whole schedule

No production call site does this today. It is the discriminating measurement: the model gets the catalog and the requirements and must produce the schedule itself. Read it as *what would we gain, or risk, by trusting the model with sequencing?*

| Model | PLAN_VIABLE | Structure OK | Req. coverage | Violations/plan | Consistency | Median s |
|---|---|---|---|---|---|---|
| `qwen2.5-7b` | 0% (0%–0%) | 100% | 50% | 8.8 | 100% | 3.1 |
| `qwen2.5-coder-7b-instruct` | 0% (0%–0%) | 100% | 49% | 14.0 | 100% | 3.2 |
| `qwen3-8b` | 0% (0%–0%) | 100% | 90% | 8.6 | 100% | 3.7 |
| `qwen3.5-9b` | 0% (0%–0%) | 71% | 67% | 15.7 | 100% | 7.8 |
| `gpt-oss-20b` | 0% (0%–0%) | 0% | 0% | 0.0 | 100% | 55.2 |
| `qwen3-coder-30b` | 0% (0%–0%) | 100% | 85% | 11.0 | 100% | 10.9 |
| `qwen3.6-27b` | 0% (0%–0%) | 100% | 100% | 8.0 | 100% | 72.0 |
| `qwen3.6-35b-a3b` | 0% (0%–0%) | 86% | 84% | 7.4 | 100% | 11.1 |
| `glm-4.7-flash` | 0% (0%–0%) | 100% | 66% | 6.4 | 100% | 11.1 |
| `gemma4-26b` | 0% (0%–0%) | 100% | 88% | 6.1 | 100% | 11.5 |
| `nemotron-cascade-2` | 0% (0%–0%) | 71% | 37% | 11.1 | 100% | 8.7 |
| `qwen3-next` | 0% (0%–0%) | 100% | 95% | 5.7 | 100% | 9.5 |
| `gpt-oss-120b` | 0% | 0% | 0% | 0.0 | — | 122.0 |

**Where plans break** (violation counts across ALL runs, including the unsatisfiable scenario — a violation is a violation regardless of whether full coverage was reachable):

| Model | prereq | term offering | credit cap | hallucinated | duplicate |
|---|---|---|---|---|---|
| `qwen2.5-7b` | 169 | 220 | 34 | 24 | 105 |
| `qwen2.5-coder-7b-instruct` | 53 | 110 | 2 | 113 | 20 |
| `qwen3-8b` | 47 | 150 | 31 | 0 | 4 |
| `qwen3.5-9b` | 69 | 147 | 57 | 0 | 57 |
| `gpt-oss-20b` | 0 | 0 | 0 | 0 | 0 |
| `qwen3-coder-30b` | 46 | 165 | 21 | 0 | 6 |
| `qwen3.6-27b` | 12 | 135 | 21 | 0 | 0 |
| `qwen3.6-35b-a3b` | 18 | 111 | 27 | 0 | 0 |
| `glm-4.7-flash` | 34 | 93 | 0 | 0 | 7 |
| `gemma4-26b` | 6 | 123 | 0 | 0 | 0 |
| `nemotron-cascade-2` | 36 | 78 | 12 | 84 | 24 |
| `qwen3-next` | 21 | 102 | 0 | 0 | 0 |
| `gpt-oss-120b` | 0 | 0 | 0 | 0 | 0 |

**PLAN_VIABLE by scenario** — a model that only fails the hardest scenario is a different proposition from one that fails everywhere.

| Model | mi-ai-early | mi-fresh-start | mi-light-load | mi-sophomore-catchup | mi-spring-start | mi-summer-accelerate | mi-theory-averse | mi-tight-horizon |
|---|---|---|---|---|---|---|---|---|
| `gemma4-26b` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `glm-4.7-flash` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `gpt-oss-120b` | — | 0% | — | — | — | — | — | — |
| `gpt-oss-20b` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `nemotron-cascade-2` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `qwen2.5-7b` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `qwen2.5-coder-7b-instruct` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `qwen3-8b` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `qwen3-coder-30b` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `qwen3-next` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `qwen3.5-9b` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `qwen3.6-27b` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `qwen3.6-35b-a3b` | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

### Mode A — the app's real revise-plan path

This is `advisor_agent.revise_plan` as shipped: the model emits a `PlanEditProposal` and the deterministic planner rebuilds the schedule. Viability here should be ~100% for every model **by construction** — if it isn't, the harness's vendored planner has drifted from the app's, and that is itself the finding.

| Model | PLAN_VIABLE | Structure OK | Req. coverage | Violations/plan | Consistency | Median s |
|---|---|---|---|---|---|---|
| `gpt-oss-20b` | 100% (100%–100%) | 100% | 100% | 0.0 | 100% | 27.4 |
| `qwen3.6-27b` | 100% (100%–100%) | 100% | 100% | 0.0 | 100% | 109.1 |
| `qwen3.6-35b-a3b` | 100% (100%–100%) | 100% | 100% | 0.0 | 100% | 17.4 |
| `gpt-oss-120b` | 100% | 100% | 100% | 0.0 | — | 81.2 |
| `qwen3-8b` | 96% (91%–100%) | 100% | 99% | 0.0 | 86% | 1.5 |
| `qwen3-coder-30b` | 95% (86%–100%) | 100% | 99% | 0.0 | 86% | 5.1 |
| `qwen3.5-9b` | 86% (86%–86%) | 100% | 98% | 0.0 | 100% | 3.2 |
| `gemma4-26b` | 86% (86%–86%) | 100% | 98% | 0.0 | 100% | 5.4 |
| `qwen2.5-7b` | 76% (71%–86%) | 100% | 96% | 0.0 | 86% | 2.3 |
| `qwen2.5-coder-7b-instruct` | 76% (71%–86%) | 100% | 97% | 0.0 | 86% | 1.0 |
| `glm-4.7-flash` | 76% (71%–86%) | 100% | 97% | 0.0 | 86% | 7.1 |
| `nemotron-cascade-2` | 71% (71%–71%) | 100% | 95% | 0.0 | 100% | 6.9 |
| `qwen3-next` | 71% (71%–71%) | 100% | 95% | 0.0 | 100% | 10.2 |

**Where plans break** (violation counts across ALL runs, including the unsatisfiable scenario — a violation is a violation regardless of whether full coverage was reachable):

| Model | prereq | term offering | credit cap | hallucinated | duplicate |
|---|---|---|---|---|---|
| `qwen2.5-7b` | 0 | 0 | 0 | 0 | 0 |
| `qwen2.5-coder-7b-instruct` | 0 | 0 | 0 | 0 | 0 |
| `qwen3-8b` | 0 | 0 | 0 | 0 | 0 |
| `qwen3.5-9b` | 0 | 0 | 0 | 0 | 0 |
| `gpt-oss-20b` | 0 | 0 | 0 | 0 | 0 |
| `qwen3-coder-30b` | 0 | 0 | 0 | 0 | 0 |
| `qwen3.6-27b` | 0 | 0 | 0 | 0 | 0 |
| `qwen3.6-35b-a3b` | 0 | 0 | 0 | 0 | 0 |
| `glm-4.7-flash` | 0 | 0 | 0 | 0 | 0 |
| `gemma4-26b` | 0 | 0 | 0 | 0 | 0 |
| `nemotron-cascade-2` | 0 | 0 | 0 | 0 | 0 |
| `qwen3-next` | 0 | 0 | 0 | 0 | 0 |
| `gpt-oss-120b` | 0 | 0 | 0 | 0 | 0 |

**Did the model actually help?** Mode A's plan is legal no matter what the model says — the deterministic planner guarantees that, and a hallucinated proposal degrades to a no-op. These columns are what separates a model that understood the student from one that returned an empty proposal.

| Model | Proposal parsed | Touched anything | Grounded | Ask honoured | Plan not worse |
|---|---|---|---|---|---|
| `qwen2.5-7b` | 100% | 100% | 0% | 71% | 71% |
| `qwen2.5-coder-7b-instruct` | 100% | 88% | 75% | 93% | 67% |
| `qwen3-8b` | 100% | 89% | 79% | 78% | 89% |
| `qwen3.5-9b` | 62% | 62% | 38% | 71% | 88% |
| `gpt-oss-20b` | 4% | 4% | 4% | 74% | 100% |
| `qwen3-coder-30b` | 100% | 100% | 79% | 79% | 88% |
| `qwen3.6-27b` | 62% | 62% | 62% | 93% | 100% |
| `qwen3.6-35b-a3b` | 50% | 38% | 38% | 71% | 88% |
| `glm-4.7-flash` | 100% | 100% | 29% | 86% | 79% |
| `gemma4-26b` | 100% | 100% | 100% | 79% | 79% |
| `nemotron-cascade-2` | 100% | 88% | 88% | 79% | 75% |
| `qwen3-next` | 88% | 88% | 75% | 86% | 75% |
| `gpt-oss-120b` | 0% | 0% | 0% | 100% | 100% |

### When nothing fits — the honesty check

One scenario in the fixture is unsatisfiable by design (one semester left, more requirements than can fit). Nobody scores PLAN_VIABLE here. The question is whether the model says so.

| Model | Stayed in horizon | Respected credit cap | Declared what didn't fit | Violations/plan |
|---|---|---|---|---|
| `gemma4-26b` | 100% | 100% | 100% | 0.0 |
| `glm-4.7-flash` | 100% | 100% | 100% | 0.0 |
| `gpt-oss-20b` | 100% | 100% | 0% | 0.0 |
| `nemotron-cascade-2` | 100% | 100% | 0% | 0.0 |
| `qwen2.5-7b` | 100% | 100% | 100% | 0.0 |
| `qwen2.5-coder-7b-instruct` | 100% | 100% | 100% | 1.0 |
| `qwen3-8b` | 100% | 0% | 100% | 6.0 |
| `qwen3-coder-30b` | 100% | 0% | 100% | 2.0 |
| `qwen3-next` | 100% | 100% | 100% | 1.0 |
| `qwen3.5-9b` | 100% | 100% | 0% | 0.0 |
| `qwen3.6-27b` | 100% | 100% | 100% | 0.0 |
| `qwen3.6-35b-a3b` | 100% | 100% | 0% | 0.0 |

### Grounded QA (the RAG summarization step)

Retrieval is fixed and identical for every model — the chunks come from `questions.yaml`, not a live pgvector query — so this measures the chat model only, not the embedding model. `Faith-flagged` and `Recall-flagged` are HEURISTICS: entity-level triage, not verdicts. Grade the review queue before quoting a faithfulness number.

| Model | Behavior OK | Auto-pass | Faith-flagged | Recall-flagged | Median s |
|---|---|---|---|---|---|
| `qwen2.5-7b` | 89% (89%–89%) | 89% | 0% | 39% | 0.3 |
| `qwen2.5-coder-7b-instruct` | 94% (94%–94%) | 89% | 6% | 43% | 0.2 |
| `qwen3-8b` | 83% (83%–83%) | 83% | 0% | 37% | 0.3 |
| `qwen3.5-9b` | 89% (89%–89%) | 89% | 0% | 33% | 0.7 |
| `gpt-oss-20b` | 70% (67%–78%) | 22% | 83% | 83% | 4.8 |
| `qwen3-coder-30b` | 83% (83%–83%) | 78% | 6% | 24% | 1.7 |
| `qwen3.6-27b` | 83% (83%–83%) | 78% | 6% | 17% | 7.7 |
| `qwen3.6-35b-a3b` | 78% (78%–78%) | 78% | 0% | 50% | 0.8 |
| `glm-4.7-flash` | 80% (78%–83%) | 78% | 2% | 43% | 0.8 |
| `gemma4-26b` | 89% (89%–89%) | 89% | 0% | 50% | 1.0 |
| `nemotron-cascade-2` | 72% (72%–72%) | 33% | 72% | 78% | 1.3 |
| `qwen3-next` | 89% (89%–89%) | 89% | 0% | 28% | 1.4 |

### Explain-plan

Every model explains the SAME deterministic baseline plan, which isolates explanation quality from planning quality. Faith-flagged is a heuristic.

| Model | Faith-flagged | Truncated | Median output tokens | Median s |
|---|---|---|---|---|
| `qwen2.5-7b` | 4% | 17% | 857.5 | 5.3 |
| `qwen2.5-coder-7b-instruct` | 0% | 4% | 783.5 | 4.7 |
| `qwen3-8b` | 0% | 29% | 914.5 | 6.4 |
| `qwen3.5-9b` | 38% | 25% | 921.0 | 7.1 |
| `gpt-oss-20b` | 71% | 75% | 1024.0 | 27.4 |
| `qwen3-coder-30b` | 12% | 21% | 987.0 | 20.8 |
| `qwen3.6-27b` | 50% | 62% | 1024.0 | 119.5 |
| `qwen3.6-35b-a3b` | 50% | 88% | 1024.0 | 17.4 |
| `glm-4.7-flash` | 8% | 0% | 599.0 | 12.9 |
| `gemma4-26b` | 8% | 0% | 466.0 | 11.7 |
| `nemotron-cascade-2` | 100% | 88% | 1024.0 | 16.6 |
| `qwen3-next` | 38% | 50% | 957.0 | 20.1 |
| `gpt-oss-120b` | 100% | 100% | 1024.0 | 73.5 |

## 2. Environment and failures

### Environment (measured, not assumed)

| Model | VRAM delta MB | GPU offload | Server n_ctx | Matches config? |
|---|---|---|---|---|
| `qwen2.5-7b` | [4777] | 29/29 (llama_server_layer_assignment) | 8192 | yes |
| `qwen2.5-coder-7b-instruct` | [4791] | 29/29 (llama_server_layer_assignment) | 8192 | yes |
| `qwen2.5-7b` | [4773] | 29/29 (llama_server_layer_assignment) | 8192 | yes |
| `qwen3-8b` | [5412] | 37/37 (llama_server_layer_assignment) | 8192 | yes |
| `qwen3.5-9b` | [5852] | 34/34 (llama_server_layer_assignment) | 8192 | yes |
| `gpt-oss-20b` | [1868] | 25/25 (llama_server_layer_assignment) | 8192 | yes |
| `qwen3-coder-30b` | [6367] | 49/49 (llama_server_layer_assignment) | 8192 | yes |
| `qwen3.6-27b` | [14945] | 65/65 (llama_server_layer_assignment) | 8192 | yes |
| `qwen3.6-35b-a3b` | [5746] | 42/42 (llama_server_layer_assignment) | 8192 | yes |
| `glm-4.7-flash` | [6472] | 48/48 (llama_server_layer_assignment) | 8192 | yes |
| `gemma4-26b` | [2621] | 31/31 (llama_server_layer_assignment) | 8192 | yes |
| `nemotron-cascade-2` | [10837] | 53/53 (llama_server_layer_assignment) | 8192 | yes |
| `qwen3-next` | [15162] | 49/49 (llama_server_layer_assignment) | 8192 | yes |
| `gpt-oss-120b` | [4314] | 37/37 (llama_server_layer_assignment) | 8192 | yes |
| `qwen2.5-7b` | [4774] | 29/29 (llama_server_layer_assignment) | 8192 | yes |
| `qwen3-8b` | [5407] | 37/37 (llama_server_layer_assignment) | 8192 | yes |

### Failures

A model that would not load, OOMed, or timed out is a finding, not a gap. Listed here rather than silently omitted from the tables above.

- `laguna-xs-2.1` (error): llama-server exited during startup for laguna-xs-2.1 (code 1). Last lines:
0.00.253.335 I llama_model_loader: - kv  43:                      tokenizer.ggml.merges arr[str,100026]  = ["i n", "Ġ t", "Ġ Ġ", "e r", "Ġ a...
0.00.253.341 I llama_model_loader: - kv  44:                tokenizer.ggml.bos_to

## 3. Validity guards

- ⚠️ **mixed static-prompt hashes in `plan_mode_b`** (['19f5ad5d53ea593b', '298026a69c5ce46f']) — these records came from different prompts and MUST NOT be compared. Re-run, or split the file by hash.
- ⚠️ **mixed fixture hashes** (['1ae6820982467b77', 'e2e89b35e5f7fb40']) — the scoring authority (`plan_fixtures/*.yaml`) changed partway through.
- ⚠️ **plan fixture is `verified: false`** — its prerequisite edges and term offerings are hand-written, because Purdue's catalog publishes neither. Model *rankings* are usable (every model is scored against the same rules); any absolute claim about real degree progress is not, until `python run.py fixture-check` has been run against a populated catalog DB.

## 4. Your turn

`review_queue.jsonl` holds 1439 items awaiting manual grading: every free-text answer (faithfulness is not machine-checkable) and every non-viable plan (so you can confirm the violation the harness found is real and not a fixture bug — the fastest way to discover a wrong prereq edge is to see three good models all 'fail' the same course). The plan-viability numbers above stand on their own; the faithfulness numbers do not until this queue is graded.
