"use client";

import React from "react";
import { Cpu, HardDrive, Zap, ShieldAlert, Gauge } from "lucide-react";
import { GpuTelemetry } from "@/types";

interface HardwareTelemetryBarProps {
  telemetry: GpuTelemetry;
}

export const HardwareTelemetryBar: React.FC<HardwareTelemetryBarProps> = ({ telemetry }) => {
  const vramUsedPct = Math.min(100, (telemetry.vram_used_mb / telemetry.vram_total_mb) * 100);
  const safetyLimitPct = (6963 / telemetry.vram_total_mb) * 100; // 6.8 GB limit

  return (
    <div className="glass-panel rounded-2xl p-5 border border-white/10 shadow-xl">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          <Cpu className="h-5 w-5 text-emerald-400" />
          <span className="font-semibold text-sm text-white">{telemetry.gpu_name}</span>
          <span className="rounded bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 text-[11px] font-mono text-emerald-300">
            Ada Lovelace Compute 8.9
          </span>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono text-slate-400">
          <div>
            NVMe Read: <span className="text-cyan-400 font-bold">1,542 MB/s</span>
          </div>
          <div>
            PCIe Gen4: <span className="text-emerald-400 font-bold">10.21 GB/s</span>
          </div>
          <div>
            CUDA Alloc: <span className="text-purple-400 font-bold">11.3 µs</span>
          </div>
        </div>
      </div>

      {/* VRAM Meter with Safety Ceiling */}
      <div className="space-y-2">
        <div className="flex justify-between text-xs">
          <span className="text-slate-400">
            Interactive VRAM Allocation:{" "}
            <strong className="text-white font-mono">{telemetry.vram_used_mb} MB</strong> /{" "}
            <span className="font-mono">{telemetry.vram_total_mb} MB</span>
          </span>
          <span className="text-amber-400 font-mono flex items-center gap-1 text-[11px]">
            <ShieldAlert className="h-3.5 w-3.5" /> Strict 6.8 GB Safety Ceiling
          </span>
        </div>

        <div className="relative h-3 w-full rounded-full bg-dark-950 overflow-hidden border border-white/5">
          {/* Active Usage */}
          <div
            className="h-full rounded-full bg-gradient-to-r from-cyan-500 via-emerald-500 to-amber-500 transition-all duration-500"
            style={{ width: `${vramUsedPct}%` }}
          />
          {/* Safety Marker Line */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-rose-500 z-10"
            style={{ left: `${safetyLimitPct}%` }}
            title="6.8 GB Safety Ceiling"
          />
        </div>

        <div className="flex justify-between text-[11px] text-slate-500">
          <span>0 GB</span>
          <span className="text-amber-400/80">6.8 GB Limit</span>
          <span>8.18 GB Total</span>
        </div>
      </div>
    </div>
  );
};
