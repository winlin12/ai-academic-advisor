"use client";

import { useEffect, useRef, useState } from "react";

import {
  askAdvisor,
  explainPlan,
  revisePlan,
  type AdvisorSource,
  type PlanEditProposal,
  type PlanResponse,
  type StudentProfile,
} from "@/lib/api";

// THE THREE THINGS THE CHAT CAN DO, and they are three different backend paths rather than
// three prompts — which is why the chat routes instead of just sending text:
//
//   ask     -> POST /v1/advisor/ask          retrieval-grounded QA over the catalog (pgvector)
//   explain -> POST /v1/advisor/explain-plan free text about the plan already on screen
//   revise  -> POST /v1/advisor/revise-plan  MODE A: a schema-constrained edit proposal, then
//                                            the plan is rebuilt and re-validated
//
// Only `revise` changes the plan. Keeping that a separate, explicitly-labelled path is the
// point: a student asking "why is CS 38100 so late?" must not have their schedule rewritten
// as a side effect of asking a question.
type Intent = "ask" | "explain" | "revise";
type Mode = Intent | "auto";

type ChatMessage = {
  id: number;
  role: "user" | "assistant";
  content: string;
  intent?: Intent;
  sources?: AdvisorSource[];
  proposal?: PlanEditProposal;
  // What produced this answer, kept so "regenerate" can re-run the identical request with a
  // different seed rather than making the student retype it.
  request?: string;
  // The plan as it stood BEFORE a revision. Regenerating a revision has to start from the same
  // place the first attempt did, or the second answer revises the first answer's output and
  // the edits compound — press regenerate three times and you have applied "lighter" thrice.
  planBefore?: PlanResponse | null;
  planner?: "ai" | "deterministic";
  failed?: boolean;
};

type AdvisorChatProps = {
  profile?: StudentProfile;
  plan?: PlanResponse | null;
  onPlanRevised?: (plan: PlanResponse) => void;
};

const GREETING: ChatMessage = {
  id: 0,
  role: "assistant",
  content:
    "Ask me about your degree (“what counts for the science core?”), about this plan " +
    "(“why is CS 38100 so late?”), or tell me how to change it " +
    "(“keep me at 12 credits”).",
};

// Intent routing, in priority order. Deliberately keyword-based rather than a model call: an
// extra generation to decide which generation to run would double the latency of every message
// on a local GPU, and getting it wrong is cheap here because the student can see which path ran
// and switch modes to override it.
const REVISE_PATTERNS = [
  /\b(move|swap|shift|push|pull|drop|remove|add|take|put|reorder|rearrange|rebuild|redo)\b/i,
  /\b(earlier|later|sooner|first|last|lighter|heavier|fewer|more|less)\b/i,
  /\b(cap|limit|keep me|hold me|max|at most|no more than|under)\b.{0,30}\b(credit|course|class)/i,
  /\b(instead of|rather than|change|adjust|tweak|balance)\b/i,
  /\bgraduate (early|sooner|faster)\b/i,
];

const EXPLAIN_PATTERNS = [
  /\b(why|how come|explain|walk me through|reason|rationale)\b/i,
  /\b(is this|does this|will this|am i|can i)\b.{0,40}\b(plan|schedule|semester|graduate|on track)\b/i,
  /\b(risk|risky|safe|realistic|feasible|workload|heavy|balanced)\b/i,
  /\b(what order|sequenc)/i,
  /\bthis (plan|schedule|semester)\b/i,
];

export function routeIntent(text: string, canUsePlan: boolean): Intent {
  if (!canUsePlan) return "ask";
  // Revise is checked first and needs TWO signals, because a false positive rewrites the
  // student's schedule while a false negative merely answers a question. "Should I take
  // CS 38100 earlier?" is a question about the plan; "take CS 38100 earlier" is an instruction.
  const reviseHits = REVISE_PATTERNS.filter((pattern) => pattern.test(text)).length;
  const asksAQuestion = /\?\s*$/.test(text.trim()) || /^(should|could|would|can|is|are|do|does|what|when|which|who|why|how)\b/i.test(text.trim());
  if (reviseHits >= 2 && !asksAQuestion) return "revise";
  if (reviseHits >= 1 && /^(please\s+)?(make|move|swap|drop|add|put|keep|cap|limit|reorder|rearrange|balance|change)\b/i.test(text.trim())) {
    return "revise";
  }
  if (EXPLAIN_PATTERNS.some((pattern) => pattern.test(text))) return "explain";
  return "ask";
}

function sourceLabel(source: AdvisorSource): string {
  const meta = source.metadata as {
    type?: string;
    code?: string;
    program?: string;
    block?: string;
  };
  if (meta.type === "course" && meta.code) return meta.code;
  if (meta.type === "requirement" && meta.program) {
    return meta.block ? `${meta.program} — ${meta.block}` : meta.program;
  }
  return `Source ${source.id}`;
}

// Compact one-liner describing the knobs the agent turned, so the edit is inspectable.
function proposalSummary(proposal: PlanEditProposal): string | null {
  const parts: string[] = [];
  if (proposal.reorder.length) parts.push(`earlier: ${proposal.reorder.join(", ")}`);
  if (proposal.defer.length) parts.push(`deferred: ${proposal.defer.join(", ")}`);
  if (proposal.avoid_tags.length) parts.push(`avoid: ${proposal.avoid_tags.join(", ")}`);
  if (proposal.max_credits_per_semester != null) {
    parts.push(`cap: ${proposal.max_credits_per_semester} cr`);
  }
  return parts.length ? parts.join(" · ") : null;
}

const INTENT_LABEL: Record<Intent, string> = {
  ask: "Catalog",
  explain: "About your plan",
  revise: "Plan updated",
};

export default function AdvisorChat({ profile, plan, onPlanRevised }: AdvisorChatProps) {
  const canUsePlan = Boolean(profile && plan);
  const [mode, setMode] = useState<Mode>("auto");
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([GREETING]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nextId = useRef(1);
  const scroller = useRef<HTMLDivElement>(null);

  // Follow the conversation as it grows, the way every chat client does.
  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  async function run(text: string, intent: Intent, planBefore: PlanResponse | null, seed?: number) {
    if (intent === "revise" && profile) {
      const response = await revisePlan(profile, text, planBefore ?? undefined, { seed });
      onPlanRevised?.(response.plan);
      // Wording follows `layout`, because the two cases are different facts. With the model's
      // own layout, a dropped course really was unschedulable where it was put. When the
      // deterministic planner laid the semesters out instead, the differences are just two
      // plans picking different options from the same selective menus — calling those
      // "unschedulable" tells a student a course they can take is unavailable.
      const note = response.removed.length
        ? response.layout === "model"
          ? ` Dropped as unschedulable: ${response.removed.join(", ")}.`
          : ` Swapped for other options covering the same requirements: ${response.removed.join(", ")}.`
        : "";
      return {
        content: (response.rationale || "Updated the plan.") + note,
        proposal: response.proposal,
        planner: response.planner,
      };
    }
    if (intent === "explain" && plan) {
      const response = await explainPlan(text, planBefore ?? plan, seed);
      return { content: response.answer || "No response received." };
    }
    const response = await askAdvisor(text, seed);
    return {
      content: response.answer || "No response received.",
      sources: response.sources,
    };
  }

  async function send(text: string, intent: Intent, planBefore: PlanResponse | null, seed?: number) {
    setError(null);
    setLoading(true);
    try {
      const result = await run(text, intent, planBefore, seed);
      setMessages((prev) => [
        ...prev,
        {
          id: nextId.current++,
          role: "assistant",
          intent,
          request: text,
          planBefore,
          ...result,
        },
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      setMessages((prev) => [
        ...prev,
        {
          id: nextId.current++,
          role: "assistant",
          intent,
          request: text,
          planBefore,
          failed: true,
          content: `That didn't go through — ${message}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function handleSend() {
    const text = prompt.trim();
    if (!text || loading) return;
    const intent = mode === "auto" ? routeIntent(text, canUsePlan) : mode;
    // Modes the current state cannot serve fall back to plain catalog QA rather than erroring:
    // there is no plan to explain or revise before one exists.
    const effective: Intent = canUsePlan ? intent : "ask";

    setMessages((prev) => [
      ...prev,
      { id: nextId.current++, role: "user", content: text },
    ]);
    setPrompt("");
    await send(text, effective, plan ?? null);
  }

  // REGENERATE, the way a chatbot does it: drop the answer being replaced, re-run the exact
  // same request with a different seed. A revision rewinds to the plan it started from first,
  // so pressing this twice does not apply the same edit twice.
  async function handleRegenerate(message: ChatMessage) {
    if (loading || !message.request || !message.intent) return;
    if (message.intent === "revise" && message.planBefore) {
      onPlanRevised?.(message.planBefore);
    }
    setMessages((prev) => prev.filter((item) => item.id !== message.id));
    await send(
      message.request,
      message.intent,
      message.planBefore ?? plan ?? null,
      Math.floor(Math.random() * 2_000_000_000),
    );
  }

  function handleReset() {
    setMessages([GREETING]);
    setError(null);
    setPrompt("");
  }

  const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant" && m.request);
  const modes: Mode[] = canUsePlan ? ["auto", "ask", "explain", "revise"] : ["ask"];
  const modeLabel: Record<Mode, string> = {
    auto: "Auto",
    ask: "Catalog",
    explain: "Explain",
    revise: "Revise",
  };

  return (
    <div className="card card-accent flex h-[42rem] flex-col p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="kicker">Copilot</p>
          <h2 className="font-display mb-4 mt-1 text-2xl font-semibold uppercase tracking-wide text-[var(--ink)]">
            Advisor Chat
          </h2>
        </div>
        <button
          onClick={handleReset}
          disabled={loading || messages.length <= 1}
          className="btn-ghost mt-1 whitespace-nowrap px-3 py-1.5 text-xs disabled:opacity-40"
          title="Clear the conversation. Your plan is not affected."
        >
          New chat
        </button>
      </div>

      {modes.length > 1 && (
        <div className="mb-4 flex rounded-full border border-[var(--stroke)] bg-black/30 p-1 text-xs font-semibold">
          {modes.map((value) => (
            <button
              key={value}
              onClick={() => setMode(value)}
              className={
                mode === value
                  ? "btn-gold flex-1 !rounded-full px-2 py-1.5"
                  : "flex-1 rounded-full px-2 py-1.5 text-[var(--muted)] transition hover:text-[var(--ink)]"
              }
            >
              {modeLabel[value]}
            </button>
          ))}
        </div>
      )}

      <div
        ref={scroller}
        className="scroll-dark mb-4 flex-1 space-y-3 overflow-y-auto rounded-xl border border-[var(--stroke)] bg-black/25 p-3"
      >
        {messages.map((message) => (
          <div
            key={message.id}
            className={
              message.role === "assistant"
                ? `rounded-xl border p-4 text-sm text-[var(--ink)] ${
                    message.failed
                      ? "border-red-500/30 bg-red-500/10"
                      : "border-[var(--stroke)] bg-[rgba(207,185,145,0.07)]"
                  }`
                : "ml-4 rounded-xl border border-[var(--stroke-strong)] bg-[rgba(221,185,69,0.12)] p-4 text-sm text-[var(--ink)]"
            }
          >
            <div className="mb-1 flex items-center gap-2">
              <p
                className={
                  message.role === "assistant"
                    ? "text-[0.65rem] font-semibold uppercase tracking-widest text-[var(--gold)]"
                    : "text-[0.65rem] font-semibold uppercase tracking-widest text-[var(--gold-soft)]"
                }
              >
                {message.role === "assistant" ? "Advisor" : "You"}
              </p>
              {message.role === "assistant" && message.intent && !message.failed && (
                <span className="rounded-full border border-[var(--stroke)] px-2 py-0.5 text-[0.6rem] uppercase tracking-wider text-[var(--muted)]">
                  {INTENT_LABEL[message.intent]}
                </span>
              )}
            </div>

            <div className="whitespace-pre-wrap">{message.content}</div>

            {message.proposal && proposalSummary(message.proposal) && (
              <p className="mt-2 border-t border-[var(--stroke)] pt-2 text-xs font-medium text-[var(--gold)]">
                {proposalSummary(message.proposal)}
              </p>
            )}

            {message.sources && message.sources.length > 0 && (
              <div className="mt-3 space-y-2 border-t border-[var(--stroke)] pt-3">
                <p className="text-[0.65rem] font-semibold uppercase tracking-widest text-[var(--muted)]">
                  Sources
                </p>
                {message.sources.map((source) => (
                  <div
                    key={source.id}
                    className="rounded-lg border border-[var(--stroke)] bg-black/30 p-2 text-xs text-[var(--muted)]"
                  >
                    <p className="font-semibold text-[var(--gold)]">
                      {sourceLabel(source)}{" "}
                      <span className="font-normal text-[var(--muted)]">
                        {Math.round(source.similarity * 100)}% match
                      </span>
                    </p>
                    <p className="mt-1">
                      {source.content.length > 180
                        ? `${source.content.slice(0, 180)}…`
                        : source.content}
                    </p>
                  </div>
                ))}
              </div>
            )}

            {message.role === "assistant" && message.request && (
              <button
                onClick={() => handleRegenerate(message)}
                disabled={loading}
                className="mt-3 text-xs font-semibold text-[var(--muted)] underline-offset-2 transition hover:text-[var(--gold)] hover:underline disabled:opacity-40"
              >
                {message.failed ? "Try again" : "Regenerate"}
                {message.intent === "revise" && !message.failed && " (undoes this edit first)"}
              </button>
            )}
          </div>
        ))}

        {loading && (
          <div className="rounded-xl border border-[var(--stroke)] bg-[rgba(207,185,145,0.07)] p-4 text-sm text-[var(--muted)]">
            Thinking…
          </div>
        )}
      </div>

      {error && !loading && (
        <p className="mb-3 text-xs text-red-300">{error}</p>
      )}

      <div>
        <textarea
          rows={3}
          placeholder={
            mode === "revise"
              ? "Tell me how to adjust the plan, e.g. “keep me at 12 credits”…"
              : mode === "explain"
                ? "Ask about this plan, e.g. “why is CS 38100 so late?”…"
                : "Ask your advisor…"
          }
          value={prompt}
          disabled={loading}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            // Enter sends, shift+enter newlines — the convention every chat client uses.
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void handleSend();
            }
          }}
          className="field resize-none text-sm"
        />

        <div className="mt-3 flex gap-2">
          <button
            onClick={handleSend}
            disabled={loading || !prompt.trim()}
            className="btn-gold flex-1 px-4 py-3"
          >
            {loading ? "Working…" : "Send"}
          </button>
          {lastAssistant && !loading && (
            <button
              onClick={() => handleRegenerate(lastAssistant)}
              className="btn-ghost whitespace-nowrap px-4 py-3 text-sm"
              title="Re-run the last request for a different answer"
            >
              ↻
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
