"use client";

import React, { useState } from "react";
import { SpeculativeStep } from "@/types";
import { CheckCircle2, XCircle, Zap, Layers } from "lucide-react";

interface SpeculativeVisualizerProps {
  steps: SpeculativeStep[];
}

export const SpeculativeVisualizer: React.FC<SpeculativeVisualizerProps> = ({ steps }) => {
  const [selectedStepIdx, setSelectedStepIdx] = useState<number>(0);
  const currentStep = steps[selectedStepIdx] || steps[0];

  return (
    <div className="glass-panel rounded-2xl p-6 border border-white/10 shadow-xl flex flex-col justify-between">
      {/* Visualizer Header */}
      <div>
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
              <Layers className="h-5 w-5" />
            </div>
            <div>
              <h2 className="font-sans font-bold text-lg text-white">
                Speculative Decoding Visualizer (K = 3)
              </h2>
              <p className="text-xs text-slate-300 font-medium">
                Draft Proposer (Qwen-1.5B Distilled) vs Target Verifier (Qwen-7B Quantized)
              </p>
            </div>
          </div>

          {/* Step Selector Tabs */}
          <div className="flex items-center gap-1.5 rounded-xl bg-dark-900/90 p-1.5 border border-white/10 shadow-inner">
            {steps.map((s, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedStepIdx(idx)}
                className={`rounded-lg px-3.5 py-1.5 text-xs font-mono transition-all ${
                  selectedStepIdx === idx
                    ? "bg-emerald-500 text-slate-950 font-bold shadow-md"
                    : "text-slate-300 hover:text-white hover:bg-dark-800/80"
                }`}
              >
                Cycle #{idx + 1}
              </button>
            ))}
          </div>
        </div>

        {/* Cycle High-Level Diagnostics Banner */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 my-4">
          <div className="rounded-xl bg-dark-900/90 p-3.5 border border-white/10 shadow-inner">
            <span className="text-xs text-slate-300 font-medium">Proposed Tokens</span>
            <p className="text-xl font-bold font-mono text-cyan-400 mt-1">
              {currentStep?.draft_tokens.length || 3}
            </p>
          </div>
          <div className="rounded-xl bg-dark-900/90 p-3.5 border border-white/10 shadow-inner">
            <span className="text-xs text-slate-300 font-medium">Accepted Tokens</span>
            <p className="text-xl font-bold font-mono text-emerald-400 mt-1">
              {currentStep?.decisions.filter((d) => d.accepted).length || 0}
            </p>
          </div>
          <div className="rounded-xl bg-dark-900/90 p-3.5 border border-white/10 shadow-inner">
            <span className="text-xs text-slate-300 font-medium">Cycle Acceptance</span>
            <p className="text-xl font-bold font-mono text-emerald-300 mt-1">
              {Math.round((currentStep?.acceptance_rate || 0) * 100)}%
            </p>
          </div>
          <div className="rounded-xl bg-dark-900/90 p-3.5 border border-white/10 shadow-inner">
            <span className="text-xs text-slate-300 font-medium">Cycle Latency</span>
            <p className="text-xl font-bold font-mono text-purple-400 mt-1">
              {currentStep?.cycle_latency_ms} ms
            </p>
          </div>
        </div>

        {/* Step Rejection Sampling Flow Cards */}
        <div className="space-y-3.5 mt-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-200 uppercase tracking-wider font-mono">
              Rejection Sampling Verification Tree (Cycle #{selectedStepIdx + 1})
            </span>
            <span className="text-[11px] text-slate-400 font-mono">
              Criterion: u &le; min(1, p/q)
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5">
            {currentStep?.decisions.map((dec, i) => (
              <div
                key={i}
                className={`rounded-xl p-4 border transition-all ${
                  dec.accepted
                    ? "bg-emerald-950/40 border-emerald-500/40 shadow-lg shadow-emerald-500/10"
                    : "bg-amber-950/40 border-amber-500/40 shadow-lg shadow-amber-500/10"
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-mono font-bold px-2.5 py-1 rounded-md bg-dark-950 border border-white/10 text-slate-200">
                    Draft Pos #{i + 1}
                  </span>
                  {dec.accepted ? (
                    <span className="flex items-center gap-1.5 text-xs font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/30">
                      <CheckCircle2 className="h-3.5 w-3.5" /> ACCEPTED
                    </span>
                  ) : (
                    <span className="flex items-center gap-1.5 text-xs font-bold text-amber-300 bg-amber-500/10 px-2 py-0.5 rounded-full border border-amber-500/30">
                      <XCircle className="h-3.5 w-3.5" /> REJECTED
                    </span>
                  )}
                </div>

                <div className="space-y-2 text-xs font-mono">
                  <div className="flex justify-between text-slate-300 border-b border-white/5 pb-1">
                    <span className="font-sans text-slate-400">Token ID:</span>
                    <strong className="text-white font-bold text-sm">#{dec.token_id}</strong>
                  </div>
                  <div className="flex justify-between text-slate-300">
                    <span className="font-sans text-slate-400">Draft Prob q(x):</span>
                    <strong className="text-cyan-400 font-bold">{(dec.draft_prob * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="flex justify-between text-slate-300">
                    <span className="font-sans text-slate-400">Target Prob p(x):</span>
                    <strong className="text-emerald-400 font-bold">{(dec.target_prob * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="flex justify-between text-slate-300 border-t border-white/5 pt-1">
                    <span className="font-sans text-slate-400">Ratio min(1, p/q):</span>
                    <strong className="text-purple-300 font-bold">
                      {Math.min(1.0, dec.target_prob / Math.max(dec.draft_prob, 1e-4)).toFixed(3)}
                    </strong>
                  </div>

                  {!dec.accepted && dec.resampled_token_id !== undefined && (
                    <div className="mt-2.5 pt-2 border-t border-amber-500/30 flex items-center justify-between text-amber-300 font-sans">
                      <span className="text-xs font-semibold">Resampled from (p - q)&plus;:</span>
                      <strong className="font-mono font-bold text-sm text-amber-200">
                        #{dec.resampled_token_id}
                      </strong>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Bonus Token Notification */}
          {currentStep?.bonus_token && (
            <div className="flex items-center gap-3 rounded-xl bg-cyan-950/40 border border-cyan-500/40 p-3.5 text-xs text-cyan-200 shadow-md">
              <Zap className="h-4 w-4 text-cyan-400 flex-shrink-0" />
              <div>
                <strong className="text-white font-semibold">Bonus Target Token #{currentStep.bonus_token} Sampled!</strong> All K=3 draft tokens were accepted, triggering an extra target verification token for free.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
