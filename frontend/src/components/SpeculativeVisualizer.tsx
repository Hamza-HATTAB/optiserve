"use client";

import React, { useState } from "react";
import { SpeculativeStep } from "@/types";
import { CheckCircle2, XCircle, ArrowRight, Play, RotateCcw, Zap, Layers } from "lucide-react";

interface SpeculativeVisualizerProps {
  steps: SpeculativeStep[];
}

export const SpeculativeVisualizer: React.FC<SpeculativeVisualizerProps> = ({ steps }) => {
  const [selectedStepIdx, setSelectedStepIdx] = useState<number>(0);
  const currentStep = steps[selectedStepIdx] || steps[0];

  return (
    <div className="glass-panel rounded-2xl p-5 border border-white/10 shadow-xl">
      {/* Visualizer Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
            <Layers className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-white">
              Speculative Decoding Visualizer (K = 3)
            </h2>
            <p className="text-xs text-slate-400">
              Draft Proposer (Qwen-1.5B Distilled) vs Target Verifier (Qwen-7B Quantized)
            </p>
          </div>
        </div>

        {/* Step Selector Tabs */}
        <div className="flex items-center gap-1.5 rounded-lg bg-dark-900 p-1 border border-white/5">
          {steps.map((s, idx) => (
            <button
              key={idx}
              onClick={() => setSelectedStepIdx(idx)}
              className={`rounded px-3 py-1 text-xs font-mono transition-all ${
                selectedStepIdx === idx
                  ? "bg-emerald-500 text-dark-950 font-bold shadow"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Cycle #{idx + 1}
            </button>
          ))}
        </div>
      </div>

      {/* Cycle High-Level Diagnostics Banner */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 my-4">
        <div className="rounded-xl bg-dark-900/60 p-3 border border-white/5">
          <span className="text-[11px] text-slate-400 font-medium">Proposed Tokens</span>
          <p className="text-lg font-bold font-mono text-cyan-400">
            {currentStep?.draft_tokens.length || 3}
          </p>
        </div>
        <div className="rounded-xl bg-dark-900/60 p-3 border border-white/5">
          <span className="text-[11px] text-slate-400 font-medium">Accepted Tokens</span>
          <p className="text-lg font-bold font-mono text-emerald-400">
            {currentStep?.decisions.filter((d) => d.accepted).length || 0}
          </p>
        </div>
        <div className="rounded-xl bg-dark-900/60 p-3 border border-white/5">
          <span className="text-[11px] text-slate-400 font-medium">Cycle Acceptance Rate</span>
          <p className="text-lg font-bold font-mono text-emerald-300">
            {Math.round((currentStep?.acceptance_rate || 0) * 100)}%
          </p>
        </div>
        <div className="rounded-xl bg-dark-900/60 p-3 border border-white/5">
          <span className="text-[11px] text-slate-400 font-medium">Cycle Latency</span>
          <p className="text-lg font-bold font-mono text-purple-400">
            {currentStep?.cycle_latency_ms} ms
          </p>
        </div>
      </div>

      {/* Step Rejection Sampling Flow Cards */}
      <div className="space-y-3 mt-4">
        <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          Rejection Sampling Verification Tree (Cycle #{selectedStepIdx + 1})
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {currentStep?.decisions.map((dec, i) => (
            <div
              key={i}
              className={`rounded-xl p-4 border transition-all ${
                dec.accepted
                  ? "bg-emerald-950/20 border-emerald-500/30 shadow-lg shadow-emerald-500/5"
                  : "bg-amber-950/20 border-amber-500/30 shadow-lg shadow-amber-500/5"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-dark-900 border border-white/10 text-slate-300">
                  Draft Pos #{i + 1}
                </span>
                {dec.accepted ? (
                  <span className="flex items-center gap-1 text-xs font-bold text-emerald-400">
                    <CheckCircle2 className="h-4 w-4" /> ACCEPTED
                  </span>
                ) : (
                  <span className="flex items-center gap-1 text-xs font-bold text-amber-400">
                    <XCircle className="h-4 w-4" /> REJECTED
                  </span>
                )}
              </div>

              <div className="space-y-1.5 text-xs font-mono">
                <div className="flex justify-between text-slate-400">
                  <span>Token ID:</span>
                  <span className="text-slate-100 font-bold">{dec.token_id}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Draft Prob q(x):</span>
                  <span className="text-cyan-400">{(dec.draft_prob * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Target Prob p(x):</span>
                  <span className="text-emerald-400">{(dec.target_prob * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Ratio min(1, p/q):</span>
                  <span className="text-purple-400">
                    {Math.min(1.0, dec.target_prob / Math.max(dec.draft_prob, 1e-4)).toFixed(3)}
                  </span>
                </div>

                {!dec.accepted && dec.resampled_token_id !== undefined && (
                  <div className="mt-2 pt-2 border-t border-amber-500/20 flex items-center justify-between text-amber-300 font-sans">
                    <span>Resampled from (p - q)⁺:</span>
                    <span className="font-mono font-bold">#{dec.resampled_token_id}</span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Bonus Token Notification */}
        {currentStep?.bonus_token && (
          <div className="flex items-center gap-3 rounded-xl bg-cyan-950/30 border border-cyan-500/30 p-3 text-xs text-cyan-300">
            <Zap className="h-4 w-4 text-cyan-400 flex-shrink-0" />
            <div>
              <span className="font-bold">Bonus Target Token #{currentStep.bonus_token} sampled!</span> All K=3 draft tokens were accepted, allowing an extra token forward pass for free.
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
