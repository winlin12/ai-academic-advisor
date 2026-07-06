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
    <div className="card flex h-[42rem] flex-col p-6 text-center">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-orange-700">
        Copilot
      </p>
      <h2 className="mb-4 mt-2 text-2xl font-bold text-stone-900">Advisor Chat</h2>

      {canRevise && (
        <div className="mb-4 flex rounded-full border border-orange-900/10 bg-white/60 p-1 text-xs font-semibold">
          {(["ask", "revise"] as Mode[]).map((value) => (
            <button
              key={value}
              onClick={() => setMode(value)}
              className={
                mode === value
                  ? "flex-1 rounded-full bg-gradient-to-r from-orange-600 to-orange-500 px-3 py-1.5 text-white shadow"
                  : "flex-1 rounded-full px-3 py-1.5 text-stone-600 transition hover:text-stone-900"
              }
            >
              {value === "ask" ? "Ask a question" : "Revise my plan"}
            </button>
          ))}
        </div>
      )}

      <div className="mb-4 flex-1 space-y-3 overflow-y-auto rounded-2xl border border-orange-900/10 bg-white/60 p-3 text-center">
        {messages.map((message, index) => (
          <div
            key={`${message.role}-${index}`}
            className={
              message.role === "assistant"
                ? "rounded-2xl border border-orange-900/10 bg-gradient-to-br from-amber-100 to-orange-100 p-4 text-sm text-stone-800 shadow-[0_8px_20px_rgba(120,53,15,0.1)]"
                : "rounded-2xl bg-stone-900 p-4 text-sm text-stone-100 shadow-[0_8px_20px_rgba(28,25,23,0.22)]"
            }
          >
            <p className="mb-1 text-xs font-semibold uppercase tracking-wide opacity-70">
              {message.role === "assistant" ? "Advisor" : "You"}
            </p>
            {message.content}
            {message.proposal && proposalSummary(message.proposal) && (
              <p className="mt-2 border-t border-orange-900/10 pt-2 text-left text-xs font-medium text-orange-800">
                Plan updated — {proposalSummary(message.proposal)}
              </p>
            )}
            {message.sources && message.sources.length > 0 && (
              <div className="mt-3 space-y-2 border-t border-orange-900/10 pt-3 text-left">
                <p className="text-[0.65rem] font-semibold uppercase tracking-wide opacity-60">
                  Sources
                </p>
                {message.sources.map((source) => (
                  <div
                    key={source.id}
                    className="rounded-lg border border-orange-900/10 bg-white/70 p-2 text-xs text-stone-700"
                  >
                    <p className="font-semibold text-stone-800">
                      {sourceLabel(source)}{" "}
                      <span className="font-normal opacity-60">
                        {Math.round(source.similarity * 100)}% match
                      </span>
                    </p>
                    <p className="mt-1 opacity-80">
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
          <div className="rounded-2xl border border-orange-900/10 bg-gradient-to-br from-amber-100 to-orange-100 p-4 text-sm text-stone-700">
            Thinking...
          </div>
        )}
        {error && (
          <div className="rounded-xl border border-red-400/40 bg-red-100 p-4 text-sm text-red-900">
            {error}
          </div>
        )}
      </div>

      <div className="rounded-2xl border border-orange-900/15 bg-white/75 p-3">
        <textarea
          rows={3}
          placeholder={
            reviseMode
              ? "Tell me how to adjust the plan, e.g. \"less theory-heavy\" or \"cap me at 6 credits\"..."
              : "Ask your advisor..."
          }
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="w-full resize-none rounded-xl border border-orange-900/15 bg-white/70 p-3 text-stone-900 outline-none transition placeholder:text-stone-400 focus:border-orange-500 focus:ring-2 focus:ring-orange-200"
        />

        <button
          onClick={handleSend}
          disabled={loading || !prompt.trim()}
          className="mt-3 w-full rounded-xl bg-gradient-to-r from-orange-600 to-orange-500 px-4 py-3 font-semibold text-white shadow-[0_10px_20px_rgba(234,88,12,0.28)] transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {reviseMode ? "Revise plan" : "Send"}
        </button>
      </div>
    </div>
  );
}
