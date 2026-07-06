"use client";

import { useState } from "react";

import {
  askAdvisor,
  revisePlan,
  type AdvisorSource,
  type PlanEditProposal,
  type PlanResponse,
  type StudentProfile,
} from "@/lib/api";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: AdvisorSource[];
  proposal?: PlanEditProposal;
};

type Mode = "ask" | "revise";

type AdvisorChatProps = {
  // When a profile + plan are supplied, the chat can also revise the plan in place.
  profile?: StudentProfile;
  plan?: PlanResponse | null;
  onPlanRevised?: (plan: PlanResponse) => void;
};

// A short, human-readable label for a retrieved chunk, from whatever tags it carries.
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

export default function AdvisorChat({ profile, plan, onPlanRevised }: AdvisorChatProps) {
  const canRevise = Boolean(profile && plan);
  const [mode, setMode] = useState<Mode>("ask");
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "Ask me anything about your degree, e.g. \"I want to major in CS, what should I take my second semester freshman year?\"",
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reviseMode = canRevise && mode === "revise";

  async function handleSend() {
    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt || loading) {
      return;
    }

    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: trimmedPrompt }]);
    setPrompt("");
    setLoading(true);

    try {
      if (reviseMode && profile) {
        const response = await revisePlan(profile, trimmedPrompt, plan ?? undefined);
        onPlanRevised?.(response.plan);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: response.rationale || "Updated the plan.",
            proposal: response.proposal,
          },
        ]);
      } else {
        const response = await askAdvisor(trimmedPrompt);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: response.answer || "No response received.",
            sources: response.sources,
          },
        ]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card card-accent flex h-[42rem] flex-col p-5">
      <p className="kicker">Copilot</p>
      <h2 className="font-display mb-4 mt-1 text-2xl font-semibold uppercase tracking-wide text-[var(--ink)]">
        Advisor Chat
      </h2>

      {canRevise && (
        <div className="mb-4 flex rounded-full border border-[var(--stroke)] bg-black/30 p-1 text-xs font-semibold">
          {(["ask", "revise"] as Mode[]).map((value) => (
            <button
              key={value}
              onClick={() => setMode(value)}
              className={
                mode === value
                  ? "btn-gold flex-1 !rounded-full px-3 py-1.5"
                  : "flex-1 rounded-full px-3 py-1.5 text-[var(--muted)] transition hover:text-[var(--ink)]"
              }
            >
              {value === "ask" ? "Ask a question" : "Revise my plan"}
            </button>
          ))}
        </div>
      )}

      <div className="scroll-dark mb-4 flex-1 space-y-3 overflow-y-auto rounded-xl border border-[var(--stroke)] bg-black/25 p-3">
        {messages.map((message, index) => (
          <div
            key={`${message.role}-${index}`}
            className={
              message.role === "assistant"
                ? "rounded-xl border border-[var(--stroke)] bg-[rgba(207,185,145,0.07)] p-4 text-sm text-[var(--ink)]"
                : "ml-4 rounded-xl border border-[var(--stroke-strong)] bg-[rgba(221,185,69,0.12)] p-4 text-sm text-[var(--ink)]"
            }
          >
            <p
              className={
                message.role === "assistant"
                  ? "mb-1 text-[0.65rem] font-semibold uppercase tracking-widest text-[var(--gold)]"
                  : "mb-1 text-[0.65rem] font-semibold uppercase tracking-widest text-[var(--gold-soft)]"
              }
            >
              {message.role === "assistant" ? "Advisor" : "You"}
            </p>
            {message.content}
            {message.proposal && proposalSummary(message.proposal) && (
              <p className="mt-2 border-t border-[var(--stroke)] pt-2 text-xs font-medium text-[var(--gold)]">
                Plan updated — {proposalSummary(message.proposal)}
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
          </div>
        ))}
        {loading && (
          <div className="rounded-xl border border-[var(--stroke)] bg-[rgba(207,185,145,0.07)] p-4 text-sm text-[var(--muted)]">
            Thinking…
          </div>
        )}
        {error && (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
            {error}
          </div>
        )}
      </div>

      <div>
        <textarea
          rows={3}
          placeholder={
            reviseMode
              ? "Tell me how to adjust the plan, e.g. \"less theory-heavy\" or \"cap me at 6 credits\"…"
              : "Ask your advisor…"
          }
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="field resize-none text-sm"
        />

        <button
          onClick={handleSend}
          disabled={loading || !prompt.trim()}
          className="btn-gold mt-3 w-full px-4 py-3"
        >
          {reviseMode ? "Revise plan" : "Send"}
        </button>
      </div>
    </div>
  );
}
