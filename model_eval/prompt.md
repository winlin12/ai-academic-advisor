# Mode C: Retry-to-Convergence Benchmark

## Context

This is a new benchmark mode for the model evaluation bracket used in the Purdue
academic advising chatbot project. Prior modes (Mode A: one-shot quality columns,
Mode B: viability under load, existing Mode C: slot/pacing anchoring tests) measure
single-attempt or structural performance. This new test measures something
different: **how long it takes a model to converge on a valid plan of study (POS)
when given repeated attempts**, which better reflects real usage — a student
iterating with the assistant rather than accepting or rejecting a single output.

If "Mode C" is already in use for the slot/pacing anchoring tests, rename that one
or slot this in as Mode D — just flag the naming collision before wiring it into
the harness so results don't get merged under the same label.

## Goal

For each candidate model, measure how many attempts and how much wall-clock time
it takes to reach a 100%-valid plan of study, given a fixed timeout window.

## Definitions

- **Valid plan of study (100%)**: zero prerequisite violations, zero credit-hour
  cap violations, and zero term-offering violations across all **filled** slots.
    Empty/unfilled slots are not violations — per the locked/ai_generated/empty slot
      model, an empty slot is a legitimate terminal state, not a failure. Convergence
        means "no errors among what's filled," not "every slot filled."
        - **Locked slots**: user-specified courses (planned/preferred). These are fixed
          inputs to every attempt and must never be altered by the model across retries.
            Only `ai_generated` and `empty` slots are eligible for revision between attempts.
            - **Attempt**: one full generation of a plan (or one revision pass), each checked
              against the validator (prereq DAG, credit-hour bin-packing, term-offering
                constraints, boolean AND/OR prereq paths).
                - **Timeout**: a wall-clock cap per test case, configurable in the 5–15 minute
                  range. A model that fails to converge before timeout is **not** simply recorded
                    as a failure — record it as right-censored (i.e., "did not converge within N
                      minutes"), not equivalent to a model that converged at, say, minute 14.

                      ## Two retry variants to implement (track separately, do not conflate)

                      1. **Blind resampling**: same prompt, new sample (temperature/seed variation), no
                         feedback about why the prior attempt failed. Tests raw variance / luck of the
                            draw.
                            2. **Feedback-guided retry**: the validator's specific violations (e.g. "prereq
                               edge X→Y violated in term 3," "credit hours exceed cap in term 5") are fed back
                                  into the next attempt's context. Tests self-correction capability, which is
                                     closer to a real advising session where structured error output could be
                                        surfaced to the model.

                                        These are different capabilities. Report them as separate columns, not averaged
                                        together.

                                        ## Metrics to report per model, per variant

                                        - **Attempts to converge** (median, p90, and count of timeouts)
                                        - **Wall-clock time to converge** (median, p90)
                                        - **Tokens/sec throughput** (already known from Mode A/B — report alongside, but
                                          do NOT let throughput differences masquerade as quality differences). A model
                                            needing 3 attempts at 23 tok/s may converge faster in wall-clock time than a
                                              model needing 1 attempt at 6 tok/s — that's a hardware/throughput artifact, not
                                                a reasoning-quality result. Surface both numbers so they can't be conflated.
                                                - **Timeout rate**: fraction of test cases that hit the cap without converging,
                                                  treated as censored data in any statistical summary (not simply dropped, not
                                                    simply counted as "0% quality").

                                                    ## Implementation notes

                                                    - Reuse the existing benchmark fixture and validator from Mode A/B, but confirm
                                                      the three flagged prereq edges (suspected bugs inflating violation counts) are
                                                        resolved or excluded before this test runs — they'll compound across every
                                                          retry attempt and skew convergence rates.
                                                          - Locked slots should be stripped out of whatever bin-packing/DAG pass runs per
                                                            attempt and re-inserted as fixed constraints, consistent with how locked slots
                                                              are handled in the production POS schema.
                                                              - Log every intermediate attempt (not just the final converged plan or the
                                                                timeout state) so attempt-by-attempt error patterns can be inspected later —
                                                                  this is useful for diagnosing whether a model is making *progress* toward
                                                                    convergence or just resampling randomly without improvement.
                                                                    - Keep this mode's harness separate from Mode A/B's one-shot runner; it needs its
                                                                      own timeout/retry loop and its own result schema (attempts, wall-clock, censored
                                                                        flag) rather than reusing the pass/fail schema from prior modes.

                                                                        ## Open questions to resolve before running

                                                                        - Exact timeout value within the 5–15 min range (per test case or shared across
                                                                          the batch?)
                                                                          - Whether feedback-guided retries get the full violation list or a truncated
                                                                            "top N" list (full list may make convergence trivial and uninformative;
                                                                              truncated list is closer to what a real system would realistically surface)
                                                                              - Whether attempts are capped (e.g. max 10 tries) independent of the wall-clock
                                                                                timeout, in case a model converges on attempt count before time runs out or
                                                                                  vice versa
                                                                                  