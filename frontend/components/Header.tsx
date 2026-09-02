"use client";

import React from "react";
import { Cpu, Zap, Radio, Info } from "lucide-react";
import { ConnectionMode } from "@/types";

interface HeaderProps {
  connectionMode: ConnectionMode;
  onOpenConnectModal: () => void;
  isBackendConnected: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  connectionMode,
  onOpenConnectModal,
  isBackendConnected,
}) => {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-white/10 bg-[#07090e]/90 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
        {/* Brand & Identity */}
        <div className="flex items-center gap-3.5">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500 to-cyan-600 shadow-lg shadow-emerald-500/25 border border-emerald-400/30">
            <Zap className="h-6 w-6 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-xl font-extrabold tracking-tight text-white sm:text-2xl font-sans">
                OPTISERVE
              </h1>
              <span className="font-mono text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/30 px-2.5 py-0.5 rounded-full font-semibold">
                DISTILLFLOW ENGINE v0.1.0
              </span>
            </div>
            <p className="text-xs text-slate-300 font-medium hidden sm:block">
              PyTorch Reasoning Distillation, Quantization Benchmarks & Speculative Decoding Engine
            </p>
          </div>
        </div>

        {/* System & Connection Telemetry Badge */}
        <div className="flex items-center gap-3">
          <div className="hidden lg:flex items-center gap-2 rounded-xl bg-dark-900/90 border border-white/10 px-3.5 py-1.5 text-xs shadow-inner">
            <Cpu className="h-4 w-4 text-cyan-400" />
            <span className="text-slate-200 font-semibold">NVIDIA RTX 4060 (8GB)</span>
            <span className="text-slate-600">|</span>
            <span className="text-emerald-400 font-mono font-bold">NVMe 1.54 GB/s</span>
          </div>

          <div className="flex flex-col items-end">
            <button
              onClick={onOpenConnectModal}
              className={`flex items-center gap-2 rounded-xl border px-3.5 py-1.5 text-xs font-semibold transition-all shadow-md ${
                connectionMode === "live" && isBackendConnected
                  ? "border-emerald-500/50 bg-emerald-500/15 text-emerald-300 hover:bg-emerald-500/25"
                  : "border-cyan-500/40 bg-cyan-500/15 text-cyan-300 hover:bg-cyan-500/25"
              }`}
            >
              <span className="relative flex h-2 w-2">
                <span
                  className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                    connectionMode === "live" && isBackendConnected
                      ? "bg-emerald-400"
                      : "bg-cyan-400"
                  }`}
                ></span>
                <span
                  className={`relative inline-flex rounded-full h-2 w-2 ${
                    connectionMode === "live" && isBackendConnected
                      ? "bg-emerald-500"
                      : "bg-cyan-500"
                  }`}
                ></span>
              </span>
              <span className="font-mono tracking-wide">
                {connectionMode === "live" && isBackendConnected
                  ? "Live GPU Tunnel (RTX 4060)"
                  : "Simulator Mode (Offline Replay)"}
              </span>
            </button>
            <span className="text-[10px] text-slate-400 font-medium mt-0.5 flex items-center gap-1">
              <Info className="h-2.5 w-2.5 text-cyan-400" /> Interactive replay of empirical RTX 4060 GPU runs
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
