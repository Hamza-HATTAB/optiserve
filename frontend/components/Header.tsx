"use client";

import React from "react";
import { Cpu, Zap, Activity, Radio, ShieldCheck, Terminal } from "lucide-react";
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
    <header className="sticky top-0 z-40 w-full border-b border-white/10 bg-dark-950/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
        {/* Brand & Identity */}
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500 to-cyan-600 shadow-lg shadow-emerald-500/20">
            <Zap className="h-5 w-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold tracking-tight text-white sm:text-xl">
                OPTISERVE
              </h1>
              <span className="rounded-md bg-emerald-500/10 px-2 py-0.5 text-xs font-semibold text-emerald-400 border border-emerald-500/20">
                DISTILLFLOW
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              PyTorch Reasoning Distillation & Speculative Decoding Engine
            </p>
          </div>
        </div>

        {/* System & Connection Telemetry Badge */}
        <div className="flex items-center gap-3">
          <div className="hidden lg:flex items-center gap-2 rounded-lg bg-dark-900 border border-white/5 px-3 py-1.5 text-xs">
            <Cpu className="h-4 w-4 text-cyan-400" />
            <span className="text-slate-300">NVIDIA RTX 4060 (8GB)</span>
            <span className="text-slate-500">|</span>
            <span className="text-emerald-400 font-mono">NVMe 1.54 GB/s</span>
          </div>

          <button
            onClick={onOpenConnectModal}
            className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium transition-all ${
              connectionMode === "live" && isBackendConnected
                ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20"
                : "border-cyan-500/30 bg-cyan-500/10 text-cyan-300 hover:bg-cyan-500/20"
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
            <span className="uppercase font-mono tracking-wider">
              {connectionMode === "live" && isBackendConnected
                ? "GPU Live (Tunnel)"
                : "Simulator Mode"}
            </span>
          </button>
        </div>
      </div>
    </header>
  );
};
