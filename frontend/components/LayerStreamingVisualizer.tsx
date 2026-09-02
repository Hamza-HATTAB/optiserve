"use client";

import React, { useState, useEffect } from "react";
import {
  HardDrive,
  Cpu,
  Zap,
  ArrowRight,
  ShieldCheck,
  Activity,
  Layers,
  Play,
  Pause,
  RefreshCw,
  Gauge,
  Sliders,
} from "lucide-react";

interface LayerStreamingVisualizerProps {
  currentLayer?: number;
}

export const LayerStreamingVisualizer: React.FC<LayerStreamingVisualizerProps> = () => {
  const [activeLayer, setActiveLayer] = useState<number>(24);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [streamSpeedMs, setStreamSpeedMs] = useState<number>(180);
  const totalLayers = 80; // 70B parameter model architecture

  // Layer cycling simulation
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setActiveLayer((prev) => (prev >= totalLayers - 1 ? 0 : prev + 1));
    }, streamSpeedMs);
    return () => clearInterval(interval);
  }, [isPlaying, streamSpeedMs]);

  // Calculations for memory visualizer
  const activeBlockMb = 1280;
  const staticBuffersMb = 870;
  const totalActiveMb = activeBlockMb + staticBuffersMb; // 2150 MB
  const totalVramMb = 8188;
  const safetyCeilingMb = 6800; // 6.8 GB strict limit
  const kvCacheReservedMb = 1600; // 1.6 GB reserved for continuous batching

  const usedPct = (totalActiveMb / totalVramMb) * 100;
  const safetyLimitPct = (safetyCeilingMb / totalVramMb) * 100;
  const kvCacheStartPct = ((safetyCeilingMb - kvCacheReservedMb) / totalVramMb) * 100;
  const kvCacheWidthPct = (kvCacheReservedMb / totalVramMb) * 100;

  // Window of visible layers for the pipeline visualization
  const startWindow = Math.max(0, activeLayer - 3);
  const windowLayers = Array.from({ length: 7 }, (_, i) => startWindow + i).filter(
    (l) => l < totalLayers
  );

  return (
    <div className="glass-panel rounded-2xl p-6 border border-white/10 shadow-2xl relative overflow-hidden">
      {/* Background ambient glow */}
      <div className="absolute -right-20 -bottom-20 w-80 h-80 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -left-20 -top-20 w-80 h-80 bg-violet-500/5 rounded-full blur-3xl pointer-events-none" />

      {/* Header & Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-6 border-b border-white/10 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-violet-600 to-cyan-500 text-white shadow-lg shadow-violet-500/20">
            <Layers className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <h3 className="font-sans text-base font-bold text-white tracking-tight">
                AirLLM Sequential Layer-Streaming Pipeline
              </h3>
              <span className="font-mono text-xs font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/30 px-2.5 py-0.5 rounded-full">
                70B on 8GB VRAM
              </span>
            </div>
            <p className="text-xs text-slate-300 font-medium mt-0.5">
              NVMe SSD &rarr; PCIe Gen4 x16 &rarr; Ada Lovelace Execution Window
            </p>
          </div>
        </div>

        {/* Playback Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-mono font-bold transition-all shadow-md ${
              isPlaying
                ? "border-cyan-500/40 bg-cyan-500/15 text-cyan-300 hover:bg-cyan-500/25"
                : "border-white/10 bg-dark-900 text-slate-300 hover:text-white"
            }`}
          >
            {isPlaying ? (
              <>
                <Pause className="h-3.5 w-3.5" /> STREAMING
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5" /> PAUSED
              </>
            )}
          </button>

          <button
            onClick={() => setActiveLayer(0)}
            title="Rewind to Layer 0"
            className="rounded-xl border border-white/10 bg-dark-900 p-2 text-xs text-slate-300 hover:text-white hover:bg-white/5 transition-all"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Main PCIe Pipeline Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 mb-6">
        {/* Source: NVMe Storage Node */}
        <div className="lg:col-span-3 rounded-xl border border-white/10 bg-dark-950/80 p-4 shadow-inner flex flex-col justify-between">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2 text-slate-200 font-semibold text-xs">
              <HardDrive className="h-4 w-4 text-cyan-400" />
              <span>Samsung 990 Pro NVMe</span>
            </div>
            <span className="h-2 w-2 rounded-full bg-cyan-400 animate-ping" />
          </div>

          <div className="space-y-2 font-mono text-xs">
            <div className="flex justify-between text-slate-400">
              <span>Throughput:</span>
              <strong className="text-cyan-400 font-bold">1,542 MB/s</strong>
            </div>
            <div className="flex justify-between text-slate-400">
              <span>Storage Form:</span>
              <strong className="text-slate-200">Safetensors (FP8)</strong>
            </div>
            <div className="flex justify-between text-slate-400">
              <span>Total Layers:</span>
              <strong className="text-purple-300 font-bold">80 Transformer Blocks</strong>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-white/5 text-[11px] text-slate-400 font-sans">
            Pre-buffered asynchronous layer prefetch queue active.
          </div>
        </div>

        {/* The Animated PCIe Gen4 Bus */}
        <div className="lg:col-span-6 rounded-xl border border-white/10 bg-dark-950/90 p-4 shadow-inner flex flex-col justify-between relative overflow-hidden">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2 text-slate-200 font-semibold text-xs">
              <Zap className="h-4 w-4 text-emerald-400" />
              <span>PCIe Gen4 x16 Direct Bus (10.21 GB/s)</span>
            </div>
            <span className="font-mono text-xs text-emerald-400 font-bold">
              Layer #{activeLayer} / {totalLayers - 1}
            </span>
          </div>

          {/* Sequential Layer Train */}
          <div className="relative py-3 flex items-center justify-center gap-2 overflow-x-auto">
            {windowLayers.map((layerNum) => {
              const isCurrent = layerNum === activeLayer;
              const isPast = layerNum < activeLayer;
              return (
                <div
                  key={layerNum}
                  className={`flex flex-col items-center justify-center rounded-xl p-2.5 min-w-[70px] border transition-all ${
                    isCurrent
                      ? "border-cyan-400 bg-cyan-950/60 shadow-lg shadow-cyan-500/25 scale-105"
                      : isPast
                      ? "border-white/5 bg-dark-900/50 text-slate-500 opacity-60"
                      : "border-white/10 bg-dark-900 text-slate-300"
                  }`}
                >
                  <span className="text-[10px] font-mono text-slate-400">Block</span>
                  <span
                    className={`font-mono text-sm font-bold ${
                      isCurrent ? "text-cyan-300" : "text-slate-200"
                    }`}
                  >
                    #{layerNum}
                  </span>
                  <span className="text-[9px] font-sans mt-0.5 text-slate-400">
                    {isCurrent ? "Computing" : isPast ? "Evicted" : "Queued"}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Bus Progress Bar */}
          <div className="space-y-1.5 mt-2">
            <div className="flex justify-between text-[11px] font-mono text-slate-400">
              <span>Layer Pipeline Progress</span>
              <span className="text-cyan-300 font-bold">
                {(((activeLayer + 1) / totalLayers) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-2 w-full rounded-full bg-dark-950 overflow-hidden border border-white/10">
              <div
                className="h-full rounded-full bg-gradient-to-r from-violet-500 via-cyan-400 to-emerald-400 transition-all duration-200 shadow-md"
                style={{ width: `${((activeLayer + 1) / totalLayers) * 100}%` }}
              />
            </div>
          </div>
        </div>

        {/* Destination: GPU Execution Window Node */}
        <div className="lg:col-span-3 rounded-xl border border-white/10 bg-dark-950/80 p-4 shadow-inner flex flex-col justify-between">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2 text-slate-200 font-semibold text-xs">
              <Cpu className="h-4 w-4 text-emerald-400" />
              <span>RTX 4060 8GB VRAM</span>
            </div>
            <span className="rounded-md bg-emerald-500/15 border border-emerald-500/30 px-2 py-0.5 text-[10px] font-mono text-emerald-300 font-semibold">
              Compute 8.9
            </span>
          </div>

          <div className="space-y-2 font-mono text-xs">
            <div className="flex justify-between text-slate-400">
              <span>Active Layer Footprint:</span>
              <strong className="text-emerald-400 font-bold">{activeBlockMb} MB</strong>
            </div>
            <div className="flex justify-between text-slate-400">
              <span>Working Buffers:</span>
              <strong className="text-slate-200">{staticBuffersMb} MB</strong>
            </div>
            <div className="flex justify-between text-slate-400">
              <span>CUDA Alloc Latency:</span>
              <strong className="text-purple-300 font-bold">11.3 &micro;s</strong>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-white/5 text-[11px] text-emerald-300 font-sans flex items-center gap-1.5">
            <ShieldCheck className="h-3.5 w-3.5 flex-shrink-0 text-emerald-400" />
            <span>0.00% Out-of-Memory probability</span>
          </div>
        </div>
      </div>

      {/* Dynamic VRAM Memory Gauge with Strict 6.8 GB Safety Ceiling */}
      <div className="rounded-xl border border-white/10 bg-dark-950/70 p-4.5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-xs font-semibold text-white">
            <Activity className="h-4 w-4 text-cyan-400" />
            <span>Dynamic VRAM Partitioning &amp; Safety Envelope</span>
          </div>

          <div className="flex items-center gap-3 text-xs font-mono">
            <div className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-cyan-400" />
              <span className="text-slate-300">AirLLM Footprint ({totalActiveMb} MB)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              <span className="text-slate-300">Reserved KV-Cache Buffer (1.6 GB)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-rose-500" />
              <span className="text-amber-300 font-semibold">Strict 6.8 GB Guard</span>
            </div>
          </div>
        </div>

        {/* The Layered Progress Bar */}
        <div className="relative h-5 w-full rounded-xl bg-dark-950 overflow-hidden border border-white/10 shadow-inner">
          {/* Active AirLLM Footprint (2150 MB) */}
          <div
            className="h-full bg-gradient-to-r from-violet-600 to-cyan-500 transition-all duration-300"
            style={{ width: `${usedPct}%` }}
            title={`Active AirLLM Peak: ${totalActiveMb} MB`}
          />

          {/* Reserved Continuous-Batching KV-Cache Buffer (1.6 GB) */}
          <div
            className="absolute top-0 bottom-0 bg-emerald-500/25 border-x border-emerald-500/40"
            style={{
              left: `${kvCacheStartPct}%`,
              width: `${kvCacheWidthPct}%`,
            }}
            title="Dedicated 1.6 GB Continuous Batching KV-Cache Buffer"
          />

          {/* Strict 6.8 GB Safety Ceiling Marker */}
          <div
            className="absolute top-0 bottom-0 w-1 bg-rose-500 z-20 shadow-[0_0_12px_rgba(244,63,94,1)]"
            style={{ left: `${safetyLimitPct}%` }}
            title="Strict 6.8 GB Safety Ceiling (Guarantees System Stability)"
          />
        </div>

        {/* Numeric Labels Under Bar */}
        <div className="flex justify-between text-xs font-mono text-slate-300 pt-1">
          <span>0.0 GB</span>
          <span className="text-cyan-300 font-bold">
            AirLLM Peak: 2.15 GB ({((totalActiveMb / totalVramMb) * 100).toFixed(0)}%)
          </span>
          <span className="text-emerald-300 font-semibold">
            +1.6 GB Continuous Batching Buffer
          </span>
          <span className="text-rose-400 font-bold">
            6.80 GB Safety Limit
          </span>
          <span className="text-slate-400 font-medium">8.19 GB Total</span>
        </div>
      </div>
    </div>
  );
};
