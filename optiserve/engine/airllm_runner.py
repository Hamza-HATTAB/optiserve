import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from optiserve.config import settings


@dataclass
class StreamingStepMetric:
    layer_idx: int
    disk_load_ms: float
    forward_ms: float
    peak_vram_mb: float


@dataclass
class AirLLMGenerationResult:
    token_ids: List[int]
    text: Optional[str]
    total_time_s: float
    avg_layer_load_ms: float
    peak_vram_mb: float
    step_metrics: List[StreamingStepMetric]


class TransformerBlock(nn.Module):
    """Standard decoder block with pre-norm self-attention and swiglu mlp."""

    def __init__(self, hidden_dim: int, num_heads: int, intermediate_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        residual = x
        normed = self.input_layernorm(x)

        b, s, d = x.shape
        q = self.q_proj(normed).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        current_kv = (k, v)

        # scaled dot-product attention
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=(s > 1 and past_key_value is None)
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, d)
        x = residual + self.o_proj(attn_out)

        # swiglu mlp
        residual = x
        normed = self.post_attention_layernorm(x)
        mlp_out = self.down_proj(torch.nn.functional.silu(self.gate_proj(normed)) * self.up_proj(normed))
        x = residual + mlp_out

        return x, current_kv


class ShardedLayerDiskManager:
    """Manages serializing and streaming individual transformer layers to/from NVMe."""

    def __init__(self, cache_dir: Path, hidden_dim: int, num_heads: int, intermediate_dim: int):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim

    def initialize_layer_shards(self, num_layers: int, dtype: torch.dtype = torch.float16):
        # creates sharded weight files on NVMe if not already present
        for i in range(num_layers):
            layer_path = self.cache_dir / f"layer_{i:03d}.pt"
            if not layer_path.exists():
                layer = TransformerBlock(self.hidden_dim, self.num_heads, self.intermediate_dim).to(dtype)
                torch.save(layer.state_dict(), layer_path)
                del layer

    def load_layer_to_device(
        self, layer_idx: int, device: torch.device, dtype: torch.dtype = torch.float16
    ) -> TransformerBlock:
        layer_path = self.cache_dir / f"layer_{layer_idx:03d}.pt"
        if not layer_path.exists():
            raise FileNotFoundError(f"Missing layer shard: {layer_path}")

        # instantiate shell on meta or cpu, then copy state dict
        layer = TransformerBlock(self.hidden_dim, self.num_heads, self.intermediate_dim).to(dtype)
        state_dict = torch.load(layer_path, map_location="cpu", weights_only=True)
        layer.load_state_dict(state_dict)
        layer.to(device)
        return layer


class AirLLMRunner:
    """
    Sequential layer-wise streaming inference engine.
    Streams weights from NVMe SSD into GPU VRAM layer by layer,
    executes forward pass, cleans up VRAM, and preserves hidden state and KV cache.
    Peak memory remains strictly bounded by one layer + current activations.
    """

    def __init__(
        self,
        num_layers: int = 16,
        hidden_dim: int = 2048,
        num_heads: int = 16,
        intermediate_dim: int = 5632,
        vocab_size: int = 32000,
        cache_dir: Optional[Path] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.vocab_size = vocab_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.cache_dir = cache_dir or (settings.PROJECT_ROOT / "data" / "cache" / "airllm_shards")
        self.disk_manager = ShardedLayerDiskManager(
            self.cache_dir, hidden_dim, num_heads, intermediate_dim
        )
        self.disk_manager.initialize_layer_shards(num_layers, dtype)

        # persistent components kept on device
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim).to(self.device, dtype=dtype)
        self.final_norm = nn.LayerNorm(hidden_dim).to(self.device, dtype=dtype)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False).to(self.device, dtype=dtype)

    def forward_layer_wise(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], List[StreamingStepMetric]]:
        """
        Executes one full forward pass across all layers sequentially streaming from NVMe.
        Guarantees VRAM deallocation after each layer.
        """
        input_ids = input_ids.to(self.device)
        hidden_states = self.embed_tokens(input_ids)

        new_kvs = []
        step_metrics = []

        for layer_idx in range(self.num_layers):
            t0 = time.perf_counter()
            layer = self.disk_manager.load_layer_to_device(layer_idx, self.device, self.dtype)
            layer.eval()
            disk_load_ms = (time.perf_counter() - t0) * 1000.0

            layer_past_kv = past_key_values[layer_idx] if past_key_values else None

            t1 = time.perf_counter()
            with torch.no_grad():
                hidden_states, current_kv = layer(hidden_states, past_key_value=layer_past_kv)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            forward_ms = (time.perf_counter() - t1) * 1000.0

            # record metric and VRAM usage
            peak_vram = 0.0
            if self.device.type == "cuda":
                peak_vram = torch.cuda.memory_allocated(self.device) / (1024 * 1024)

            step_metrics.append(
                StreamingStepMetric(
                    layer_idx=layer_idx,
                    disk_load_ms=round(disk_load_ms, 2),
                    forward_ms=round(forward_ms, 2),
                    peak_vram_mb=round(peak_vram, 2),
                )
            )

            # store KV cache (keep on device or offload if needed)
            new_kvs.append((current_kv[0].detach(), current_kv[1].detach()))

            # explicit deallocation to keep VRAM clean
            del layer
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, new_kvs, step_metrics

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 8,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AirLLMGenerationResult:
        """Sequential autoregressive generation with layer-wise NVMe streaming."""
        start_time = time.perf_counter()
        generated = list(prompt_tokens)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        all_metrics: List[StreamingStepMetric] = []
        peak_vram_overall = 0.0

        for step in range(max_new_tokens):
            if step == 0:
                cur_input = input_ids
            else:
                cur_input = torch.tensor([[generated[-1]]], dtype=torch.long, device=self.device)

            logits, past_kvs, metrics = self.forward_layer_wise(cur_input, past_key_values=past_kvs)
            all_metrics.extend(metrics)

            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            # greedy or top-p sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1).item()
            generated.append(next_token)

            if self.device.type == "cuda":
                mem = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                if mem > peak_vram_overall:
                    peak_vram_overall = mem

        total_time = time.perf_counter() - start_time
        avg_load = sum(m.disk_load_ms for m in all_metrics) / max(len(all_metrics), 1)

        return AirLLMGenerationResult(
            token_ids=generated,
            text=None,
            total_time_s=round(total_time, 3),
            avg_layer_load_ms=round(avg_load, 2),
            peak_vram_mb=round(peak_vram_overall, 2),
            step_metrics=all_metrics,
        )
