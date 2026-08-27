"use client";

import React, { useState, useEffect } from "react";
import { Header } from "@/components/Header";
import { HardwareTelemetryBar } from "@/components/HardwareTelemetryBar";
import { SpeculativeVisualizer } from "@/components/SpeculativeVisualizer";
import { ParetoFrontierChart } from "@/components/ParetoFrontierChart";
import { LiveInferenceConsole } from "@/components/LiveInferenceConsole";
import { ConnectionModal } from "@/components/ConnectionModal";
import {
  MOCK_BENCHMARKS,
  MOCK_SPECULATIVE_STEPS,
  MOCK_TRAJECTORIES,
} from "@/lib/mockData";
import { ConnectionMode, GpuTelemetry, QuantBenchmark, SpeculativeStep } from "@/types";

export default function StudioPage() {
  const [connectionMode, setConnectionMode] = useState<ConnectionMode>("simulator");
  const [backendUrl, setBackendUrl] = useState<string>("http://localhost:8000");
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);
  const [isConnectModalOpen, setIsConnectModalOpen] = useState<boolean>(false);

  const [benchmarks] = useState<QuantBenchmark[]>(MOCK_BENCHMARKS);
  const [speculativeSteps] = useState<SpeculativeStep[]>(MOCK_SPECULATIVE_STEPS);

  const [gpuTelemetry, setGpuTelemetry] = useState<GpuTelemetry>({
    gpu_name: "NVIDIA GeForce RTX 4060 Laptop GPU",
    vram_total_mb: 8188.0,
    vram_used_mb: 4118.0,
    vram_free_mb: 4070.0,
    temperature_c: 44,
    system_load_pct: 18.5,
  });

  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [outputStream, setOutputStream] = useState<string>("");
  const [inferenceTelemetry, setInferenceTelemetry] = useState({
    ttft_ms: 18.4,
    itl_ms: 14.2,
    tps: 70.4,
    speedup: 1.92,
    tokens_count: 0,
  });

  // check backend status on mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const res = await fetch(`${backendUrl}/health`, { method: "GET" });
        if (res.ok) {
          const data = await res.json();
          setIsBackendConnected(true);
          setGpuTelemetry((prev) => ({
            ...prev,
            vram_used_mb: data.vram_used_mb || prev.vram_used_mb,
          }));
        } else {
          setIsBackendConnected(false);
        }
      } catch (e) {
        setIsBackendConnected(false);
      }
    };
    checkBackend();
  }, [backendUrl]);

  const handleRunInference = async (prompt: string, speculative: boolean) => {
    setIsGenerating(true);
    setOutputStream("");

    if (connectionMode === "live" && isBackendConnected) {
      // live SSE streaming from local RTX 4060 daemon
      try {
        const res = await fetch(`${backendUrl}/api/v1/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt,
            max_tokens: 128,
            temperature: 0.7,
            speculative,
          }),
        });

        if (!res.body) return;
        const reader = res.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);
          const lines = chunk.split("\n");
          for (const line of lines) {
            if (line.startsWith("data:")) {
              try {
                const parsed = JSON.parse(line.replace("data:", "").trim());
                if (parsed.token_str) {
                  setOutputStream((prev) => prev + parsed.token_str);
                }
              } catch (e) {}
            }
          }
        }
      } catch (err) {
        // fallback
        setOutputStream("Error streaming from GPU backend. Reverting to simulator.");
      } finally {
        setIsGenerating(false);
      }
    } else {
      // standalone simulation mode for 100% public Vercel access
      const matching = MOCK_TRAJECTORIES.find((t) => t.prompt === prompt);
      const fullText = matching ? matching.chosen : MOCK_TRAJECTORIES[0].chosen;

      const words = fullText.split(" ");
      let currentText = "";

      for (let i = 0; i < words.length; i++) {
        await new Promise((resolve) => setTimeout(resolve, speculative ? 25 : 50));
        currentText += (i === 0 ? "" : " ") + words[i];
        setOutputStream(currentText);
      }

      setInferenceTelemetry({
        ttft_ms: 18.4,
        itl_ms: speculative ? 13.8 : 28.5,
        tps: speculative ? 72.4 : 35.1,
        speedup: speculative ? 1.92 : 1.0,
        tokens_count: words.length,
      });

      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-dark-950 text-slate-100 flex flex-col">
      <Header
        connectionMode={connectionMode}
        onOpenConnectModal={() => setIsConnectModalOpen(true)}
        isBackendConnected={isBackendConnected}
      />

      <main className="flex-1 mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 space-y-6">
        {/* Hardware Status Strip */}
        <HardwareTelemetryBar telemetry={gpuTelemetry} />

        {/* Speculative Visualizer & Live Inference Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <SpeculativeVisualizer steps={speculativeSteps} />
          <LiveInferenceConsole
            onRunInference={handleRunInference}
            isGenerating={isGenerating}
            outputStream={outputStream}
            telemetry={inferenceTelemetry}
          />
        </div>

        {/* The 5-Way Quantization Bake-Off */}
        <ParetoFrontierChart benchmarks={benchmarks} />
      </main>

      {/* Connection Modal */}
      <ConnectionModal
        isOpen={isConnectModalOpen}
        onClose={() => setIsConnectModalOpen(false)}
        connectionMode={connectionMode}
        onModeChange={(mode, url) => {
          setConnectionMode(mode);
          setBackendUrl(url);
        }}
        backendUrl={backendUrl}
      />

      <footer className="border-t border-white/5 py-6 text-center text-xs text-slate-500">
        OptiServe Engine v0.1.0 • Built with PyTorch, CUDA, FastAPI & Next.js 14 • Hamza Hattab
      </footer>
    </div>
  );
}
