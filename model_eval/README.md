# Model Evaluation Harness

A standalone harness that answers one question with numbers instead of vibes:

> Is the quality gap between the best 8GB-class local model and Qwen3.6-27B large
> enough, **on this app's two narrow tasks**, to justify ~$1,400 of used 3090 + PSU?

Default answer: **no**. The harness exists to overturn that default only if the data
demands it. It talks directly to Ollama's HTTP API and depends on nothing in the app
(the app's backend/DB are not required or touched).

## Requirements

- Python 3.10+, `pip install pyyaml` (only dependency)
- A running Ollama server (`localhost:11434` or edit `config.yaml`)
- `nvidia-smi` on the measurement boxes (absent = VRAM recorded as null)
- Models pulled onto **SSD** storage (HDD cold-loads pollute latency)

## Quick start

```bash
cd model_eval
python run.py db        # build eval.sqlite from schema.sql
python run.py check     # verify golds execute, Ollama reachable, GPU visible
python run.py run --brackets 8gb,small          # on the 2060 Super box
python run.py run --models qwen3.6:27b          # on the 5070 Ti box
python run.py report                            # tables + decision-rule check
# then, for the champion the report names:
python run.py run --mitigate --models <best-8gb-model>
python run.py report                            # now includes before/after
```

Results are JSONL under `results/`; runs on two machines can be merged by
concatenating their `runs_baseline.jsonl` files (the report tolerates that — every
record is self-describing and carries the static-prompt hash).

## Files

| File | Responsibility |
|---|---|
| `config.yaml` | The single source of every knob that could pollute a comparison: `num_ctx`, temperature, seed, `num_predict`, runs-per-pair, warmup count, per-model `think` switch, decision thresholds. |
| `questions.yaml` | ~30 placeholder questions across 7 categories with `expected_behavior` and (where written) gold SQL. **Edit me** — the content is placeholder, the structure is final. |
| `schema.sql` | SQLite **stub** adapted from the real Postgres schema (`backend/app/data/db_schema.md`) + fake seed rows. Embedded verbatim in the static prompt block. |
| `harness/ollama_client.py` | Streaming stdlib HTTP client: TTFT, Ollama's own token/duration counters, version probe, explicit unload. |
| `harness/prompts.py` | Static-first prompt builder. Static block = role + schema + output contract + few-shot; question (and rows) last. sha256 of the static block travels with every record. |
| `harness/db.py` | Builds the DB; read-only execution (`mode=ro` + `PRAGMA query_only`); execution-accuracy row comparison. |
| `harness/scorers.py` | SQL_VALID / SQL_CORRECT / behavior (decline/clarify) / faithfulness heuristic. The "honesty ledger" at the top says which are truly automatic. |
| `harness/runner.py` | Orchestration: warmup + discard, VRAM delta via nvidia-smi, offload from server log, N runs per pair, stage A (text-to-SQL) + stage B (summarize **gold** rows), mitigation mode. |
| `harness/report.py` | Per-model table, head-to-head, variance display, mechanical decision-rule check, manual review queue. No composite score exists. |

Data flow: `questions.yaml` → `prompts.py` (static-first prompt) → `ollama_client.py` →
raw output → `scorers.py` (against `eval.sqlite`, read-only) → `results/runs_*.jsonl` →
`report.py` → `results/report.md` + `results/review_queue.jsonl` (your manual grading).

## What is automatic vs. your judgment (honestly)

| Metric | Status |
|---|---|
| SQL_VALID | **Automatic.** Parse + execute on a read-only connection. |
| SQL_CORRECT | **Automatic where you supply gold SQL.** Execution accuracy (row value-bags), so equivalent SQL formulations pass. Known false-positive mode: wrong logic returning coincidentally identical values on a small seed DB — spot-check passes. |
| DECLINE / CLARIFY | **Automatic-ish.** The output contract (`SQL:`/`CLARIFY:`/`DECLINE:`) makes it parseable; off-format outputs hit a phrase heuristic and are flagged into the review queue. Note the confound: sentinel-format compliance is itself a model skill being measured. |
| FAITHFULNESS | **Manual, full stop.** The heuristic (course codes/numbers in the answer that aren't in the rows) is a triage filter that catches *entity* hallucination only. Relational hallucination ("X requires Y") is not machine-checkable without a judge model — which I deliberately did not add, because judging small models with another model smuggles in a second unvalidated instrument. Every summary lands in `review_queue.jsonl`; the number you quote is your manual grade. |
| Ambiguity handling | Parsing is automatic; whether the clarifying question is *sensible* is your call (review queue). |
| LATENCY | Automatic. Tiebreaker only. 27B latency measured on the 5070 Ti says **nothing** about 3090 latency — different architecture, bandwidth, and thermals. Only its quality scores transfer. |

## Validity threats already defended against

- `num_ctx` set explicitly and identically for all models (config), never defaulted
- `ollama --version` recorded in `meta_*.json` (Gemma-4 fix landed in 0.22.0)
- text-only tags preferred; the tag used is the recorded model name
- thinking disabled per-model via `think: false` where supported, and recorded;
  `<think>` blocks are stripped before parsing so r1-style leakage doesn't crash scoring
- warmup generations per model are run and discarded (count in config + meta)
- VRAM measured as nvidia-smi **delta** around model load, after warmup
- GPU offload parsed from the server log (`offloaded X/Y layers`) — `ollama ps` reports a
  memory split, not a layer split, and is recorded only as a labeled fallback
- models unloaded between blocks so VRAM baselines don't stack
- static prompt block hashed into every record; the report refuses to compare mixed hashes

## The biggest threats you had NOT listed

1. **Statistical power.** ~30 questions is the real sample size — the 3 runs per pair are
   correlated replicates of the same item, not independent samples. A binomial 95% CI at
   n=30 is roughly ±15–18 percentage points. That is the same magnitude as any decision
   threshold you'd plausibly set, which is why the decision rule below demands both a
   large gap **and** a minimum gold count, and why the report refuses to call a gap real
   under `min_gold_questions`. If the observed gap is 10–20pp, the correct response is
   "write 30 more gold questions," not "buy the GPU."
2. **Prompt-template fit** (runner-up). One shared prompt + one set of few-shots will fit
   some models' instruction tuning better than others; part of any measured gap is
   prompt compatibility, not capacity. Mitigation mode improves the prompt only for the
   8GB champion, which biases the final comparison *against* buying — an acceptable
   direction given the default is NO, but be aware the 27B never gets the same favor.
3. **Quantization confound.** Q4_K_M does not damage all architectures equally; you're
   comparing (model × quant) pairs, not models. Acceptable — you'd deploy Q4_K_M anyway —
   but say "quantized model" in any conclusion you write down.

## Pre-registered decision rule (PROPOSAL — approve or edit before running)

Thresholds live in `config.yaml` under `decision:` and the report evaluates them
mechanically. Proposed:

**Buy the 3090 only if ALL of these hold** after mitigation:

1. `SQL_CORRECT(27B) − SQL_CORRECT(best-8GB, mitigated) ≥ 15pp`, with **≥ 30 gold-scored
   questions** (at exactly 30, insist on ≥ 20pp — see threat #1; at 50+ golds, 15pp stands).
2. `SQL_CORRECT(best-8GB, mitigated) < 80%` absolute. Rationale: at ≥80%, the semantic
   cache (repeat-heavy traffic) plus retry-on-error covers the residual; users see few
   failures even though the model has them.
3. Your **manual** faithfulness review shows the mitigated 8GB model inventing unsupported
   claims in >10% of summaries while the 27B stays ≤2%. This criterion can also fire the
   purchase **alone** — unfaithful summaries are the one failure the cache actively makes
   worse (a hallucinated answer gets cached and repeated).

**Explicitly insufficient to justify the purchase:**
- Any DECLINE_RATE gap (decline behavior is prompt- and middleware-fixable; your
  Turnstile/rate-limit layer is the real defense, not the model)
- Any latency number (the 2060 Super either serves the chosen 8GB model acceptably or
  you pick a smaller model; a 3090 bought for latency is a want, not a need — and the
  5070 Ti's 27B latency is not evidence about the 3090's)
- A gap that appears only without mitigation (then the fix costs $0, not $1,400)

## Interpreting `results/report.md`

- Rates show `overall (min–max across run replicates)` — if the range is wide, the
  model is unstable at temperature 0.15 and single-run comparisons are meaningless.
- "Consistency" = questions where all N runs agreed. Low consistency + small gaps = noise.
- Anything labeled UNVERIFIED (offload) or "heuristic" (faithfulness) means exactly that.
