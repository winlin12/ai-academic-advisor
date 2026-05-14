import { useState } from "react";

import { explainPlan, PlanResponse } from "@/lib/api";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type Props = {
  plan: PlanResponse;
};

export default function AdvisorChat({ plan }: Props) {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content: "Ask about your current plan and I will explain it.",
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSend() {
    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt || loading) {
      return;
    }

    setError(null);
    setMessages((prev) => [
      ...prev,
      { role: "user", content: trimmedPrompt },
    ]);
    setPrompt("");
    setLoading(true);

    try {
      const response = await explainPlan(trimmedPrompt, plan);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: response.answer || "No response received." },
      ]);
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
          placeholder="Ask your advisor..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="w-full resize-none rounded-xl border border-orange-900/15 bg-white/70 p-3 text-stone-900 outline-none transition placeholder:text-stone-400 focus:border-orange-500 focus:ring-2 focus:ring-orange-200"
        />

        <button
          onClick={handleSend}
          disabled={loading || !prompt.trim()}
          className="mt-3 w-full rounded-xl bg-gradient-to-r from-orange-600 to-orange-500 px-4 py-3 font-semibold text-white shadow-[0_10px_20px_rgba(234,88,12,0.28)] transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
        >
          Send
        </button>
      </div>
    </div>
  );
}
