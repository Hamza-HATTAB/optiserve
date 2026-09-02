"use client";

import React, { useState } from "react";
import { QuantBenchmark } from "@/types";
import { BarChart3, TrendingUp, Layers, Table } from "lucide-react";

interface ParetoFrontierChartProps {
  benchmarks: QuantBenchmark[];
}

export const ParetoFrontierChart: React.FC<ParetoFrontierChartProps> = ({ benchmarks }) => {
  const [activeMetric, setActiveMetric] = useState<"speed_vram" | "accuracy_latency">("speed_vram");

  return (
    <div className="glass-panel rounded-2xl p-6 border border-white/10 shadow-xl space-y-6">
      {/* Chart Header & Metric Toggle */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <BarChart3 className="h-5 w-5" />
          </div>
          <div>
            <h2 className="font-sans font-bold text-lg text-white">
              The 5-Way Quantization Bake-Off &amp; Pareto Frontier
            </h2>
            <p className="text-xs text-slate-300 font-medium">
              Empirical RTX 4060 hardware benchmarks across AWQ, GPTQ, GGUF, FP8, FP16 &amp; AirLLM 70B
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 rounded-xl bg-dark-900/90 p-1.5 border border-white/10 text-xs shadow-inner">
          <button
            onClick={() => setActiveMetric("speed_vram")}
            className={`rounded-lg px-3.5 py-1.5 font-sans font-semibold transition-all ${
              activeMetric === "speed_vram"
                ? "bg-cyan-500 text-slate-950 shadow-md font-bold"
                : "text-slate-300 hover:text-white"
            }`}
          >
            Throughput vs VRAM
          </button>
          <button
            onClick={() => setActiveMetric("accuracy_latency")}
            className={`rounded-lg px-3.5 py-1.5 font-sans font-semibold transition-all ${
              activeMetric === "accuracy_latency"
                ? "bg-cyan-500 text-slate-950 shadow-md font-bold"
                : "text-slate-300 hover:text-white"
            }`}
          >
            GSM8K % vs Latency
          </button>
        </div>
      </div>

      {/* Visual Relative Bar Comparison */}
      <div className="space-y-4">
        {benchmarks.map((b) => {
          const isAirLLM = b.format_name.includes("AirLLM");
          const isFP8 = b.format_name.includes("FP8");

          // normalize metrics for bar display
          const throughputPct = Math.min(100, (b.throughput_tokens_per_sec / 75.0) * 100);
          const vramPct = Math.min(100, (b.vram_peak_mb / 7200.0) * 100);

          return (
            <div
              key={b.format_name}
              className={`rounded-xl p-4.5 border transition-all ${
                isFP8
                  ? "bg-emerald-950/30 border-emerald-500/40 shadow-lg shadow-emerald-500/5"
                  : isAirLLM
                  ? "bg-purple-950/30 border-purple-500/40 shadow-lg shadow-purple-500/5"
                  : "bg-dark-900/80 border-white/10 hover:border-white/20"
              }`}
            >
              <div className="flex flex-wrap items-center justify-between gap-3 mb-2.5">
                <div className="flex items-center gap-2.5">
                  <span className="font-sans font-bold text-sm text-white">{b.format_name}</span>
                  {isFP8 && (
                    <span className="rounded-md bg-emerald-500/20 px-2 py-0.5 text-xs font-bold text-emerald-300 border border-emerald-500/40">
                      PARETO WINNER (SPEED)
                    </span>
                  )}
                  {isAirLLM && (
                    <span className="rounded-md bg-purple-500/20 px-2 py-0.5 text-xs font-bold text-purple-300 border border-purple-500/40">
                      PARETO WINNER (ACCURACY)
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-3.5 text-xs font-mono">
                  <span className="text-cyan-400 font-bold">{b.throughput_tokens_per_sec} tok/s</span>
                  <span className="text-slate-600">|</span>
                  <span className="text-emerald-400 font-bold">{b.vram_peak_mb} MB</span>
                  <span className="text-slate-600">|</span>
                  <span className="text-purple-300 font-semibold">PPL {b.perplexity.toFixed(2)}</span>
                  <span className="text-slate-600">|</span>
                  <span className="text-amber-300 font-bold">{b.gsm8k_accuracy.toFixed(1)}% GSM8K</span>
                </div>
              </div>

              {/* Progress Bars */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-2 text-xs">
                <div>
                  <div className="flex justify-between text-slate-300 font-medium mb-1.5">
                    <span>Generation Throughput</span>
                    <strong className="font-mono text-cyan-300 font-bold">{b.throughput_tokens_per_sec} tps</strong>
                  </div>
                  <div className="h-2.5 w-full rounded-full bg-dark-950 overflow-hidden border border-white/5">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-emerald-400 transition-all duration-500"
                      style={{ width: `${throughputPct}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-slate-300 font-medium mb-1.5">
                    <span>VRAM Memory Footprint</span>
                    <strong className="font-mono text-emerald-300 font-bold">{b.vram_peak_mb} MB / 8GB</strong>
                  </div>
                  <div className="h-2.5 w-full rounded-full bg-dark-950 overflow-hidden border border-white/5">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-purple-500 to-rose-400 transition-all duration-500"
                      style={{ width: `${vramPct}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Structured Comparative Matrix Table */}
      <div className="rounded-xl border border-white/10 bg-dark-950/80 overflow-hidden shadow-inner">
        <div className="flex items-center gap-2 p-3.5 border-b border-white/10 text-xs font-semibold text-slate-200">
          <Table className="h-4 w-4 text-cyan-400" />
          <span>Full Precision &amp; Compression Telemetry Matrix</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-dark-900/90 text-slate-400 font-mono border-b border-white/10">
              <tr>
                <th className="py-2.5 px-4 font-semibold">Format</th>
                <th className="py-2.5 px-4 font-semibold">Peak VRAM</th>
                <th className="py-2.5 px-4 font-semibold">TTFT (Prompt 256)</th>
                <th className="py-2.5 px-4 font-semibold">ITL (ms/tok)</th>
                <th className="py-2.5 px-4 font-semibold">Throughput</th>
                <th className="py-2.5 px-4 font-semibold">Perplexity</th>
                <th className="py-2.5 px-4 font-semibold">GSM8K Accuracy</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 font-mono">
              {benchmarks.map((b) => (
                <tr key={b.format_name} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-2.5 px-4 font-sans font-bold text-white">{b.format_name}</td>
                  <td className="py-2.5 px-4 text-emerald-400 font-semibold">{b.vram_peak_mb} MB</td>
                  <td className="py-2.5 px-4 text-slate-300">{b.ttft_ms_prompt_256} ms</td>
                  <td className="py-2.5 px-4 text-purple-300 font-semibold">{b.itl_ms_per_token} ms</td>
                  <td className="py-2.5 px-4 text-cyan-400 font-bold">{b.throughput_tokens_per_sec} tok/s</td>
                  <td className="py-2.5 px-4 text-slate-300">{b.perplexity.toFixed(2)}</td>
                  <td className="py-2.5 px-4 text-amber-300 font-bold">{b.gsm8k_accuracy.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Engineering Takeaway Box */}
      <div className="rounded-xl bg-dark-900/90 border border-white/10 p-4.5 text-xs text-slate-200 leading-relaxed shadow-md">
        <div className="flex items-center gap-2 font-bold text-emerald-400 mb-1.5 text-sm">
          <TrendingUp className="h-4 w-4" /> Architectural Inference Takeaway
        </div>
        <p>
          On the local 8GB RTX 4060, <strong className="text-white">FP8 (E4M3)</strong> represents the optimal latency-accuracy frontier for interactive serving (<span className="text-cyan-300 font-mono font-bold">73.0 tok/s</span> with only <span className="text-emerald-300 font-mono">0.05 PPL delta</span>). For zero-cloud reasoning distillation, <strong className="text-white">AirLLM 70B</strong> achieves frontier reasoning (<span className="text-amber-300 font-mono font-bold">84.2% GSM8K</span>) inside <span className="text-emerald-300 font-mono">2,150 MB peak VRAM</span> by offloading sequential layers to local NVMe SSD storage.
        </p>
      </div>
    </div>
  );
};
