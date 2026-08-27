export interface QuantBenchmark {
  format_name: string;
  vram_peak_mb: number;
  ttft_ms_prompt_64: number;
  ttft_ms_prompt_256: number;
  ttft_ms_prompt_1024: number;
  itl_ms_per_token: number;
  throughput_tokens_per_sec: number;
  perplexity: number;
  gsm8k_accuracy: number;
  timestamp: string;
}

export interface SpeculativeTokenDecision {
  token_id: number;
  draft_prob: number;
  target_prob: number;
  accepted: boolean;
  resampled_token_id?: number | null;
}

export interface SpeculativeStep {
  step_index: number;
  draft_tokens: number[];
  decisions: SpeculativeTokenDecision[];
  emitted_tokens: number[];
  bonus_token?: number | null;
  cycle_latency_ms: number;
  acceptance_rate: number;
}

export interface ReasoningTrajectory {
  id: string;
  prompt: string;
  chosen: string;
  rejected: string;
}

export interface GpuTelemetry {
  gpu_name: string;
  vram_total_mb: number;
  vram_used_mb: number;
  vram_free_mb: number;
  temperature_c: number;
  system_load_pct: number;
}

export type ConnectionMode = "simulator" | "live";
