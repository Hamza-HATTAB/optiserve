from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input user prompt or reasoning problem")
    max_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    speculative: bool = Field(default=True, description="Enable speculative decoding")
    draft_k: int = Field(default=3, ge=1, le=8)


class TokenDecisionSchema(BaseModel):
    token_id: int
    draft_prob: float
    target_prob: float
    accepted: bool
    resampled_token_id: Optional[int] = None


class SpeculativeStepSchema(BaseModel):
    step_index: int
    draft_tokens: List[int]
    decisions: List[TokenDecisionSchema]
    emitted_tokens: List[int]
    bonus_token: Optional[int]
    cycle_latency_ms: float
    acceptance_rate: float


class GenerateResponse(BaseModel):
    text: str
    tokens: List[int]
    total_tokens: int
    ttft_ms: float
    itl_ms: float
    tokens_per_second: float
    acceptance_rate: Optional[float] = None
    speedup_ratio: Optional[float] = None
    speculative_steps: Optional[List[SpeculativeStepSchema]] = None


class HealthResponse(BaseModel):
    status: str
    gpu_name: str
    vram_total_mb: float
    vram_used_mb: float
    vram_free_mb: float
    active_batch_size: int
    system_load_pct: float
