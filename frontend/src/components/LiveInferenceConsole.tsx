"use client";

import React, { useState } from "react";
import { Play, RotateCcw, Zap, Terminal, Sparkles } from "lucide-react";
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
    <div className="glass-panel rounded-2xl p-5 border border-white/10 shadow-xl">
      {/* Console Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-purple-500/10 border border-purple-500/20 text-purple-400">
            <Terminal className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-white">Live Inference Studio</h2>
            <p className="text-xs text-slate-400">
              Interactive token generation stream with continuous batching & speculative verification
            </p>
          </div>
        </div>

        {/* Speculative Toggle */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setSpeculative(!speculative)}
            className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-semibold transition-all ${
              speculative
                ? "bg-emerald-500/15 border-emerald-500/30 text-emerald-300"
                : "bg-dark-900 border-white/10 text-slate-400"
            }`}
          >
            <Zap className={`h-3.5 w-3.5 ${speculative ? "text-emerald-400" : "text-slate-500"}`} />
            <span>Speculative Engine: {speculative ? "ENABLED (1.92x)" : "DISABLED"}</span>
          </button>
        </div>
      </div>

      {/* Preset Problem Selectors */}
      <div className="mt-4 flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold text-slate-400">Preset Trajectories:</span>
        {MOCK_TRAJECTORIES.map((t, idx) => (
          <button
            key={t.id}
            onClick={() => handleSelectTrajectory(idx)}
            className="rounded-lg bg-dark-900 border border-white/5 px-2.5 py-1 text-xs text-slate-300 hover:border-white/20 hover:text-white transition-all"
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
          className="w-full rounded-xl bg-dark-900/90 border border-white/10 p-3 text-xs text-slate-100 placeholder-slate-500 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500 font-sans"
        />
      </div>

      {/* Action Row */}
      <div className="mt-3 flex items-center justify-between">
        <button
          onClick={() => onRunInference(prompt, speculative)}
          disabled={isGenerating || !prompt.trim()}
          className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-500 px-5 py-2 text-xs font-bold text-dark-950 shadow-lg shadow-emerald-500/20 hover:opacity-90 disabled:opacity-50 transition-all cursor-pointer"
        >
          <Play className="h-3.5 w-3.5 fill-current" />
          <span>{isGenerating ? "Streaming Tokens..." : "Run Inference"}</span>
        </button>

        {/* Telemetry Chips */}
        <div className="flex items-center gap-3 text-xs font-mono">
          <div className="hidden sm:block text-slate-400">
            TTFT: <span className="text-cyan-400 font-bold">{telemetry.ttft_ms} ms</span>
          </div>
          <div className="hidden sm:block text-slate-400">
            Speed: <span className="text-emerald-400 font-bold">{telemetry.tps} tok/s</span>
          </div>
          <div className="text-slate-400">
            Speedup: <span className="text-purple-400 font-bold">{telemetry.speedup}x</span>
          </div>
        </div>
      </div>

      {/* Streaming Terminal Output */}
      <div className="mt-4 rounded-xl bg-dark-950 border border-white/10 p-4 font-mono text-xs text-slate-200 min-h-[160px] max-h-[300px] overflow-y-auto">
        {outputStream ? (
          <div className="whitespace-pre-wrap leading-relaxed">
            {outputStream}
            {isGenerating && <span className="inline-block h-3 w-1.5 bg-emerald-400 animate-pulse ml-1" />}
          </div>
        ) : (
          <div className="flex h-32 flex-col items-center justify-center text-slate-500 text-center">
            <Sparkles className="h-6 w-6 text-slate-600 mb-2" />
            <p>Ready to stream. Click "Run Inference" above to evaluate reasoning trajectories.</p>
          </div>
        )}
      </div>
    </div>
  );
};
