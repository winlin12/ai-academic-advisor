# Model Evaluation Report

Ollama version: `0.31.2` · options: `{'num_ctx': 8192, 'temperature': 0.15, 'seed': 67, 'top_p': 0.9, 'num_predict': 1024}` · runs/pair: 3 · warmup discarded: 2

⚠️ **7 gold queries return ZERO rows against the seed DB**: lookup-08, lookup-12, lookup-13, typo-06, slang-05, slang-06, adv-06. Any candidate SQL that also returns nothing trivially matches these and counts as SQL_CORRECT — add the missing seed rows or drop the gold.

## Per-model results

Rates show overall (min–max across run replicates). SQL_CORRECT (attempted) covers only calls where the model emitted SQL against a gold question — a decline/clarify/unparseable silently drops out of that denominator. SQL_CORRECT (honest) counts every gold-eligible call, scoring abstentions as misses. E2E feeds the model's OWN retrieved rows (not gold rows) into the summarizer — closer to what a student actually sees — and is a heuristic pass/fail triage, not a verdict. Faithfulness numbers are heuristic triage counts — the real grade is your manual review of `results/review_queue.jsonl`.

| Model | Bracket | SQL_VALID | SQL_CORRECT (attempted) | SQL_CORRECT (honest) | E2E (heuristic) | DECLINE | CLARIFY | Faithfulness | p50/p95 | Consistency | Offload |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.6:35b-a3b | new | 97% (97%–97%) | 68% (68%–68%) | 62% (62%–63%) | 108/448 heuristic-pass | 100% (100%–100%) | 57% (57%–57%) | 232/448 flagged (heuristic; grade manually) | 2.1s / 16.0s | 54/54 questions run-stable | UNVERIFIED |
| qwen2.5-coder:7b | coder | 87% (86%–89%) | 70% (68%–71%) | 59% (57%–59%) | 58/222 heuristic-pass | 90% (90%–90%) | 57% (57%–57%) | 106/222 flagged (heuristic; grade manually) | 0.3s / 0.6s | 53/54 questions run-stable | full |
| qwen2.5-coder:7b-instruct | coder | 87% (86%–89%) | 70% (68%–71%) | 59% (57%–59%) | 29/111 heuristic-pass | 90% (90%–90%) | 57% (57%–57%) | 53/111 flagged (heuristic; grade manually) | 0.3s / 0.6s | 53/54 questions run-stable | full |
| qwen3.6:27b | new | 100% (100%–100%) | 58% (58%–58%) | 57% (57%–57%) | 90/333 heuristic-pass | 90% (90%–90%) | 57% (57%–57%) | 180/333 flagged (heuristic; grade manually) | 5.8s / 29.7s | 54/54 questions run-stable | UNVERIFIED |
| gemma4:26b | new | 97% (97%–97%) | 60% (60%–60%) | 57% (57%–57%) | 27/111 heuristic-pass | 70% (70%–70%) | 57% (57%–57%) | 57/111 flagged (heuristic; grade manually) | 1.2s / 3.9s | 54/54 questions run-stable | UNVERIFIED |
| qwen3-coder:30b | new | 90% (89%–92%) | 63% (60%–65%) | 57% (55%–58%) | 67/271 heuristic-pass | 10% (10%–10%) | 13% (13%–13%) | 132/272 flagged (heuristic; grade manually) | 0.9s / 5.5s | 54/54 questions run-stable | UNVERIFIED |
| qwen2.5:7b | 8gb | 84% (82%–88%) | 72% (69%–74%) | 54% (54%–54%) | 42/222 heuristic-pass | 70% (70%–70%) | 57% (57%–57%) | 120/222 flagged (heuristic; grade manually) | 0.3s / 0.6s | 54/54 questions run-stable | full |
| glm-4.7-flash | new | 100% (100%–100%) | 51% (49%–57%) | 51% (49%–57%) | 27/111 heuristic-pass | 0% (0%–0%) | 0% (0%–0%) | 53/111 flagged (heuristic; grade manually) | 0.9s / 2.4s | 54/54 questions run-stable | UNVERIFIED |
| laguna-xs-2.1 | new | 95% (95%–95%) | 54% (54%–54%) | 51% (51%–51%) | 21/111 heuristic-pass | 20% (20%–20%) | 43% (43%–43%) | 60/111 flagged (heuristic; grade manually) | 1.4s / 3.5s | 54/54 questions run-stable | UNVERIFIED |
| qwen3.5:9b | 8gb | 93% (92%–93%) | 48% (47%–49%) | 44% (44%–45%) | 36/241 heuristic-pass | 70% (70%–70%) | 0% (0%–0%) | 136/241 flagged (heuristic; grade manually) | 0.7s / 1.1s | 54/54 questions run-stable | full |
| qwen3:8b | 8gb | 93% (93%–94%) | 53% (52%–54%) | 41% (41%–41%) | 18/111 heuristic-pass | 70% (70%–70%) | 43% (43%–43%) | 56/111 flagged (heuristic; grade manually) | 0.4s / 0.8s | 53/54 questions run-stable | full |
| nemotron-cascade-2 | new | 86% (86%–86%) | 34% (34%–34%) | 30% (30%–30%) | 15/111 heuristic-pass | 0% (0%–0%) | 0% (0%–0%) | 72/111 flagged (heuristic; grade manually) | 2.2s / 4.7s | 54/54 questions run-stable | UNVERIFIED |
| hf.co/prism-ml/Bonsai-27B-gguf:Q1_0 | ? | 88% (88%–88%) | 30% (30%–30%) | 26% (26%–26%) | 21/126 heuristic-pass | 0% (0%–0%) | 0% (0%–0%) | 60/126 flagged (heuristic; grade manually) | 1.5s / 2.0s | 54/54 questions run-stable | full |
| gpt-oss:20b | new | 100% (100%–100%) | 68% (67%–71%) | 12% (11%–14%) | 6/222 heuristic-pass | 0% (0%–0%) | 86% (86%–86%) | 118/222 flagged (heuristic; grade manually) | 4.4s / 7.1s | 53/54 questions run-stable | full |
| qwen3-next | reference | — | — | 0% (0%–0%) | 0/28 heuristic-pass | — | — | 25/28 flagged (heuristic; grade manually) | 59.3s / 560.1s | 8/8 questions run-stable | UNVERIFIED |
| qwen3.5:122b-a10b | reference | — | — | — | — | — | — | — | — | — | UNVERIFIED |

## Head-to-head: best 8GB vs reference

Best 8GB model by SQL_CORRECT: **qwen2.5:7b**

| Metric | qwen2.5:7b (baseline) | qwen2.5:7b (mitigated) | qwen3.5:122b-a10b (reference) |
|---|---|---|---|
| SQL_VALID | 84% (82%–88%) | 97% (97%–97%) | — |
| SQL_CORRECT (attempted) | 72% (69%–74%) | 65% (65%–65%) | — |
| DECLINE | 70% (70%–70%) | 90% (90%–90%) | — |
| CLARIFY | 57% (57%–57%) | 29% (29%–29%) | — |
| SQL_CORRECT (honest, abstentions=miss) | 54% (54%–54%) | 59% (59%–59%) | — |
| E2E (heuristic pass) | 42/222 heuristic-pass | 26/111 heuristic-pass | — |
| Faithfulness (heuristic triage) | 120/222 flagged (heuristic; grade manually) | 60/111 flagged (heuristic; grade manually) | — |
| Latency p50/p95 | 0.3s / 0.6s | 0.5s / 1.6s | — (5070 Ti — says NOTHING about 3090 latency) |

### Decision rule check

⚠️ Only **29** gold-scored questions (< 30 pre-registered minimum). Any gap below ~18pp is within binomial noise at this sample size — **no purchase decision can be read from this run.**

Manual review queue written to `/home/wylin/ai-academic-advisor/model_eval/results/review_queue.jsonl` — faithfulness, e2e, and off-format decline calls are graded by you, not by this script.
