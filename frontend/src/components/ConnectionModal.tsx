"use client";

import React, { useState } from "react";
import { X, Radio, Server, CheckCircle2, AlertTriangle, ShieldCheck } from "lucide-react";
import { ConnectionMode } from "@/types";

interface ConnectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  connectionMode: ConnectionMode;
  onModeChange: (mode: ConnectionMode, url: string) => void;
  backendUrl: string;
}

export const ConnectionModal: React.FC<ConnectionModalProps> = ({
  isOpen,
  onClose,
  connectionMode,
  onModeChange,
  backendUrl,
}) => {
  const [mode, setMode] = useState<ConnectionMode>(connectionMode);
  const [url, setUrl] = useState<string>(backendUrl);
  const [isTesting, setIsTesting] = useState<boolean>(false);
  const [testResult, setTestResult] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleTestConnection = async () => {
    setIsTesting(true);
    setTestResult(null);
    try {
      const res = await fetch(`${url}/health`, { method: "GET" });
      if (res.ok) {
        setTestResult("success");
      } else {
        setTestResult("failed");
      }
    } catch (err) {
      setTestResult("failed");
    } finally {
      setIsTesting(false);
    }
  };

  const handleSave = () => {
    onModeChange(mode, url);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="glass-panel w-full max-w-md rounded-2xl p-6 border border-white/10 shadow-2xl relative">
        <button
          onClick={onClose}
          className="absolute right-4 top-4 text-slate-400 hover:text-white"
        >
          <X className="h-5 w-5" />
        </button>

        <div className="flex items-center gap-3 mb-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
            <Radio className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white">Engine Connection</h3>
            <p className="text-xs text-slate-400">
              Select between standalone simulation or live GPU tunnel
            </p>
          </div>
        </div>

        {/* Mode Selector */}
        <div className="space-y-3 my-4">
          <div
            onClick={() => setMode("simulator")}
            className={`cursor-pointer rounded-xl p-3.5 border transition-all ${
              mode === "simulator"
                ? "bg-cyan-950/30 border-cyan-500 text-white"
                : "bg-dark-900/60 border-white/5 text-slate-400 hover:border-white/20"
            }`}
          >
            <div className="flex items-center justify-between font-semibold text-xs mb-1">
              <span>Interactive Simulator Mode (Default)</span>
              {mode === "simulator" && <CheckCircle2 className="h-4 w-4 text-cyan-400" />}
            </div>
            <p className="text-[11px] text-slate-400">
              Zero-latency standalone mode running real pre-recorded inference trajectories and Pareto benchmark datasets.
            </p>
          </div>

          <div
            onClick={() => setMode("live")}
            className={`cursor-pointer rounded-xl p-3.5 border transition-all ${
              mode === "live"
                ? "bg-emerald-950/30 border-emerald-500 text-white"
                : "bg-dark-900/60 border-white/5 text-slate-400 hover:border-white/20"
            }`}
          >
            <div className="flex items-center justify-between font-semibold text-xs mb-1">
              <span>Live GPU Mode (Cloudflare Tunnel)</span>
              {mode === "live" && <CheckCircle2 className="h-4 w-4 text-emerald-400" />}
            </div>
            <p className="text-[11px] text-slate-400">
              Direct connection to local RTX 4060 GPU via secure Cloudflare Tunnel (<code className="text-emerald-400">make tunnel</code>).
            </p>
          </div>
        </div>

        {/* Live URL Input */}
        {mode === "live" && (
          <div className="space-y-2 mt-3 pt-3 border-t border-white/10">
            <label className="text-xs font-semibold text-slate-300">FastAPI Daemon URL</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="http://localhost:8000 or https://tunnel.optiserve.ai"
                className="flex-1 rounded-lg bg-dark-900 border border-white/10 px-3 py-1.5 text-xs text-white focus:border-emerald-500 focus:outline-none"
              />
              <button
                onClick={handleTestConnection}
                disabled={isTesting}
                className="rounded-lg bg-white/10 px-3 py-1.5 text-xs font-semibold text-white hover:bg-white/20 disabled:opacity-50"
              >
                {isTesting ? "Ping..." : "Test"}
              </button>
            </div>
            {testResult === "success" && (
              <p className="text-xs text-emerald-400 flex items-center gap-1 mt-1">
                <CheckCircle2 className="h-3.5 w-3.5" /> RTX 4060 daemon connected successfully!
              </p>
            )}
            {testResult === "failed" && (
              <p className="text-xs text-rose-400 flex items-center gap-1 mt-1">
                <AlertTriangle className="h-3.5 w-3.5" /> Backend unreachable. Falling back to simulator mode.
              </p>
            )}
          </div>
        )}

        {/* Action Buttons */}
        <div className="mt-6 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="rounded-lg px-4 py-2 text-xs font-semibold text-slate-400 hover:text-white"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="rounded-lg bg-emerald-500 px-4 py-2 text-xs font-bold text-dark-950 hover:bg-emerald-400"
          >
            Apply Settings
          </button>
        </div>
      </div>
    </div>
  );
};
