"use client";

import React, { useState } from "react";
import { QuantBenchmark } from "@/types";
import { BarChart3, HardDrive, Cpu, Gauge, TrendingUp, HelpCircle } from "lucide-react";

interface ParetoFrontierChartProps {
  benchmarks: QuantBenchmark[];
}

export const ParetoFrontierChart: React.FC<ParetoFrontierChartProps> = ({ benchmarks }) => {
  const [activeMetric, setActiveMetric] = useState<"speed_vram" | "accuracy_latency">("speed_vram");

  return (
    <div className="glass-panel rounded-2xl p-5 border border-white/10 shadow-xl">
      {/* Chart Header & Metric Toggle */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
            <BarChart3 className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-white">
              The 5-Way Quantization Bake-Off & Pareto Frontier
            </h2>
            <p className="text-xs text-slate-400">
              Empirical RTX 4060 hardware benchmarks across AWQ, GPTQ, GGUF, FP8, FP16 & AirLLM 70B
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 rounded-lg bg-dark-900 p-1 border border-white/5 text-xs">
          <button
            onClick={() => setActiveMetric("speed_vram")}
            className={`rounded px-3 py-1 font-medium transition-all ${
              activeMetric === "speed_vram"
                ? "bg-cyan-500 text-dark-950 font-bold shadow"
                : "text-slate-400 hover:text-white"
            }`}
          >
            Throughput vs VRAM
          </button>
          <button
            onClick={() => setActiveMetric("accuracy_latency")}
            className={`rounded px-3 py-1 font-medium transition-all ${
              activeMetric === "accuracy_latency"
                ? "bg-cyan-500 text-dark-950 font-bold shadow"
                : "text-slate-400 hover:text-white"
            }`}
          >
            GSM8K % vs Latency
          </button>
        </div>
      </div>

      {/* Visual Relative Bar Comparison */}
      <div className="my-6 space-y-4">
        {benchmarks.map((b) => {
          const isAirLLM = b.format_name.includes("AirLLM");
          const isFP8 = b.format_name.includes("FP8");
          const isAWQ = b.format_name.includes("AWQ");

          // normalize metrics for bar display
          const throughputPct = Math.min(100, (b.throughput_tokens_per_sec / 75.0) * 100);
          const vramPct = Math.min(100, (b.vram_peak_mb / 7200.0) * 100);

          return (
            <div
              key={b.format_name}
              className={`rounded-xl p-4 border transition-all ${
                isFP8
                  ? "bg-emerald-950/20 border-emerald-500/30"
                  : isAirLLM
                  ? "bg-purple-950/20 border-purple-500/30"
                  : "bg-dark-900/40 border-white/5 hover:border-white/10"
              }`}
            >
              <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-sm text-slate-100">{b.format_name}</span>
                  {isFP8 && (
                    <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 text-[10px] font-bold text-emerald-300 border border-emerald-500/30">
                      PARETO WINNER (SPEED)
                    </span>
                  )}
                  {isAirLLM && (
                    <span className="rounded bg-purple-500/20 px-1.5 py-0.5 text-[10px] font-bold text-purple-300 border border-purple-500/30">
                      PARETO WINNER (ACCURACY)
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-4 text-xs font-mono">
                  <span className="text-cyan-400 font-bold">{b.throughput_tokens_per_sec} tok/s</span>
                  <span className="text-slate-500">|</span>
                  <span className="text-emerald-400">{b.vram_peak_mb} MB</span>
                  <span className="text-slate-500">|</span>
                  <span className="text-purple-300">PPL {b.perplexity.toFixed(2)}</span>
                  <span className="text-slate-500">|</span>
                  <span className="text-amber-400 font-bold">{b.gsm8k_accuracy.toFixed(1)}% GSM8K</span>
                </div>
              </div>

              {/* Progress Bars */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-2 text-[11px]">
                <div>
                  <div className="flex justify-between text-slate-400 mb-1">
                    <span>Generation Throughput</span>
                    <span className="font-mono text-cyan-300">{b.throughput_tokens_per_sec} tps</span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-dark-950 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-emerald-400 transition-all duration-500"
                      style={{ width: `${throughputPct}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-slate-400 mb-1">
                    <span>VRAM Memory Footprint</span>
                    <span className="font-mono text-emerald-300">{b.vram_peak_mb} MB / 8GB</span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-dark-950 overflow-hidden">
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

      {/* Engineering Takeaway Box */}
      <div className="rounded-xl bg-dark-900/80 border border-white/10 p-4 text-xs text-slate-300">
        <div className="flex items-center gap-2 font-semibold text-emerald-400 mb-1">
          <TrendingUp className="h-4 w-4" /> Architectural Inference Takeaway
        </div>
        <p>
          On local 8GB RTX 4060, <strong className="text-white">FP8 (E4M3)</strong> delivers the optimal latency-accuracy trade-off for interactive serving (73.0 tok/s with only 0.05 PPL delta). For zero-cloud distillation generation, <strong className="text-white">AirLLM 70B</strong> achieves frontier reasoning (84.2% GSM8K) inside 2,150 MB peak VRAM by offloading sequential layers to local NVMe SSD.
        </p>
      </div>
    </div>
  );
};
