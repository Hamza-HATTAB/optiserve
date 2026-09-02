"use client";

import React, { useState } from "react";
import { Play, Zap, Terminal, Sparkles } from "lucide-react";
import { MOCK_TRAJECTORIES } from "@/lib/mockData";

interface LiveInferenceConsoleProps {
  onRunInference: (prompt: string, speculative: boolean) => void;
  isGenerating: boolean;
  outputStream: string;
  telemetry: {
    ttft_ms: number;
    itl_ms: number;
    tps: number;
    speedup: number;
    tokens_count: number;
  };
}

export const LiveInferenceConsole: React.FC<LiveInferenceConsoleProps> = ({
  onRunInference,
  isGenerating,
  outputStream,
  telemetry,
}) => {
  const [prompt, setPrompt] = useState<string>(MOCK_TRAJECTORIES[0].prompt);
  const [speculative, setSpeculative] = useState<boolean>(true);

  const handleSelectTrajectory = (idx: number) => {
    setPrompt(MOCK_TRAJECTORIES[idx].prompt);
  };

  return (
    <div className="glass-panel rounded-2xl p-6 border border-white/10 shadow-xl flex flex-col justify-between">
      <div>
        {/* Console Header */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-500/10 border border-purple-500/30 text-purple-400">
              <Terminal className="h-5 w-5" />
            </div>
            <div>
              <h2 className="font-sans font-bold text-lg text-white">Live Inference Studio</h2>
              <p className="text-xs text-slate-300 font-medium">
                Token generation telemetry with continuous batching &amp; speculative verification
              </p>
            </div>
          </div>

          {/* Speculative Toggle */}
          <button
            type="button"
            onClick={() => setSpeculative(!speculative)}
            className={`flex items-center gap-2 rounded-xl border px-3.5 py-1.5 text-xs font-semibold transition-all shadow-sm ${
              speculative
                ? "bg-emerald-500/20 border-emerald-500/40 text-emerald-300"
                : "bg-dark-900/80 border-white/10 text-slate-300 hover:text-white"
            }`}
          >
            <Zap className={`h-4 w-4 ${speculative ? "text-emerald-400" : "text-slate-400"}`} />
            <span>Speculative Proposer: <strong className="font-mono">{speculative ? "ENABLED (1.92x)" : "DISABLED"}</strong></span>
          </button>
        </div>

        {/* Preset Problem Selectors */}
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold text-slate-300">CoT Reasoning Presets:</span>
          {MOCK_TRAJECTORIES.map((t, idx) => (
            <button
              key={t.id}
              onClick={() => handleSelectTrajectory(idx)}
              className="rounded-lg bg-dark-900/90 border border-white/10 px-3 py-1 text-xs font-medium text-slate-200 hover:border-emerald-500/50 hover:text-emerald-300 transition-all shadow-sm"
            >
              Problem #{idx + 1}
            </button>
          ))}
        </div>

        {/* Input Prompt Box */}
        <div className="mt-3">
          <textarea
            rows={3}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter a mathematical or multi-step reasoning problem..."
            className="w-full rounded-xl bg-dark-950/80 border border-white/15 p-3.5 text-sm text-slate-100 placeholder-slate-400 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500 font-sans shadow-inner leading-relaxed"
          />
        </div>

        {/* Action Row */}
        <div className="mt-3.5 flex flex-wrap items-center justify-between gap-3">
          <button
            onClick={() => onRunInference(prompt, speculative)}
            disabled={isGenerating || !prompt.trim()}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-500 px-5 py-2.5 text-xs font-bold text-dark-950 shadow-lg shadow-emerald-500/25 hover:opacity-95 disabled:opacity-50 transition-all cursor-pointer"
          >
            <Play className="h-3.5 w-3.5 fill-current" />
            <span>{isGenerating ? "Streaming Reasoning Tokens..." : "Run Live Inference"}</span>
          </button>

          {/* Telemetry Chips */}
          <div className="flex items-center gap-3 text-xs font-mono">
            <div className="rounded-lg bg-dark-900/90 border border-white/10 px-2.5 py-1 text-slate-300 shadow-inner">
              TTFT: <strong className="text-cyan-400 font-bold">{telemetry.ttft_ms} ms</strong>
            </div>
            <div className="rounded-lg bg-dark-900/90 border border-white/10 px-2.5 py-1 text-slate-300 shadow-inner">
              Speed: <strong className="text-emerald-400 font-bold">{telemetry.tps} tok/s</strong>
            </div>
            <div className="rounded-lg bg-dark-900/90 border border-white/10 px-2.5 py-1 text-slate-300 shadow-inner">
              Speedup: <strong className="text-purple-300 font-bold">{telemetry.speedup}x</strong>
            </div>
          </div>
        </div>
      </div>

      {/* Streaming Terminal Output */}
      <div className="mt-4 rounded-xl bg-dark-950 border border-white/15 p-4.5 font-mono text-xs text-slate-100 min-h-[170px] max-h-[290px] overflow-y-auto shadow-inner">
        {outputStream ? (
          <div className="whitespace-pre-wrap leading-relaxed">
            {outputStream}
            {isGenerating && <span className="inline-block h-3.5 w-2 bg-emerald-400 animate-pulse ml-1 align-middle" />}
          </div>
        ) : (
          <div className="flex h-36 flex-col items-center justify-center text-slate-400 text-center">
            <Sparkles className="h-6 w-6 text-emerald-500/70 mb-2 animate-pulse" />
            <p className="font-sans font-medium text-xs text-slate-300">
              Ready to stream. Click "Run Live Inference" above to evaluate reasoning trajectories.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
