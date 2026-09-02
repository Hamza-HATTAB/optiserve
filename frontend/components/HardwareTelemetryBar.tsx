"use client";

import React from "react";
import { Cpu, HardDrive, Zap, ShieldAlert, CheckCircle2 } from "lucide-react";
import { GpuTelemetry } from "@/types";

interface HardwareTelemetryBarProps {
  telemetry: GpuTelemetry;
}

export const HardwareTelemetryBar: React.FC<HardwareTelemetryBarProps> = ({ telemetry }) => {
  const vramUsedPct = Math.min(100, (telemetry.vram_used_mb / telemetry.vram_total_mb) * 100);
  const safetyLimitPct = (6963 / telemetry.vram_total_mb) * 100; // 6.8 GB limit

  return (
    <div className="glass-panel rounded-2xl p-6 border border-white/10 shadow-xl">
      <div className="flex flex-wrap items-center justify-between gap-4 mb-5">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
            <Cpu className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <span className="font-sans font-bold text-base text-white">{telemetry.gpu_name}</span>
              <span className="rounded-md bg-emerald-500/15 border border-emerald-500/30 px-2.5 py-0.5 text-xs font-mono font-semibold text-emerald-300">
                Ada Lovelace Compute 8.9
              </span>
            </div>
            <p className="text-xs text-slate-300 font-medium mt-0.5">
              Dedicated 8,188 MiB VRAM Host with Fast Layer Streaming Architecture
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="rounded-xl bg-dark-900/90 border border-white/10 px-3 py-1.5 shadow-inner">
            <span className="text-slate-400">NVMe Read: </span>
            <strong className="text-cyan-400 font-bold">1,542 MB/s</strong>
          </div>
          <div className="rounded-xl bg-dark-900/90 border border-white/10 px-3 py-1.5 shadow-inner">
            <span className="text-slate-400">PCIe Gen4: </span>
            <strong className="text-emerald-400 font-bold">10.21 GB/s</strong>
          </div>
          <div className="rounded-xl bg-dark-900/90 border border-white/10 px-3 py-1.5 shadow-inner">
            <span className="text-slate-400">CUDA Alloc: </span>
            <strong className="text-purple-300 font-bold">11.3 µs</strong>
          </div>
        </div>
      </div>

      {/* VRAM Meter with Safety Ceiling */}
      <div className="space-y-2.5">
        <div className="flex justify-between items-center text-xs">
          <span className="text-slate-200 font-medium">
            Active VRAM Allocation:{" "}
            <strong className="text-emerald-400 font-mono text-sm">{telemetry.vram_used_mb} MB</strong>
            <span className="text-slate-400 font-mono"> / {telemetry.vram_total_mb} MB</span>
          </span>
          <span className="text-amber-300 font-mono font-semibold flex items-center gap-1.5 bg-amber-500/10 border border-amber-500/30 px-2.5 py-0.5 rounded-md text-xs">
            <ShieldAlert className="h-3.5 w-3.5 text-amber-400" /> Strict 6.8 GB Safety Ceiling Guard
          </span>
        </div>

        <div className="relative h-3.5 w-full rounded-full bg-dark-950 overflow-hidden border border-white/10 shadow-inner">
          {/* Active Usage */}
          <div
            className="h-full rounded-full bg-gradient-to-r from-cyan-500 via-emerald-500 to-amber-400 transition-all duration-500 shadow-md"
            style={{ width: `${vramUsedPct}%` }}
          />
          {/* Safety Marker Line */}
          <div
            className="absolute top-0 bottom-0 w-1 bg-rose-500 z-10 shadow-[0_0_8px_rgba(244,63,94,0.8)]"
            style={{ left: `${safetyLimitPct}%` }}
            title="6.8 GB Safety Ceiling"
          />
        </div>

        <div className="flex justify-between text-xs text-slate-300 font-mono font-medium pt-0.5">
          <span>0.0 GB</span>
          <span className="text-amber-300 font-semibold">6.8 GB Safe Headroom (Preserves 1.6GB for KV-Cache)</span>
          <span>8.19 GB Total</span>
        </div>
      </div>
    </div>
  );
};
