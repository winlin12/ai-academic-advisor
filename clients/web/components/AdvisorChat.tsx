import { useState } from "react";

import { askAdvisor, type AdvisorSource } from "@/lib/api";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: AdvisorSource[];
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

export default function AdvisorChat() {
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
      const response = await askAdvisor(trimmedPrompt);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: response.answer || "No response received.",
          sources: response.sources,
        },
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
