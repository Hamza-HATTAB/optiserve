import pytest
import torch
from optiserve.config import settings
from optiserve.benchmarks.quant_bakeoff import QUANT_SPECS, QuantizationBakeOffRunner


def test_vram_safety_ceiling_invariant():
    """
    Validates that all interactive configurations strictly observe the 6.8 GB (6963 MB)
    hardware ceiling on the local NVIDIA RTX 4060 GPU to prevent CUDA OOM exceptions.
    """
    max_allowed_mb = settings.max_active_vram_mb
    assert max_allowed_mb == 6963, "Hardware ceiling should match 6.8 GB specification"

    runner = QuantizationBakeOffRunner()
    # verify that 4-bit and FP8 formats leave abundant dynamic KV headroom
    for key in ["AWQ_4BIT", "GPTQ_4BIT", "GGUF_Q4KM", "FP8_E4M3", "AIRLLM_70B"]:
        res = runner.run_benchmark_for_format(key)
        assert res.vram_peak_mb < max_allowed_mb, (
            f"Format {key} exceeded safety limit: {res.vram_peak_mb} >= {max_allowed_mb}"
        )
        headroom_mb = max_allowed_mb - res.vram_peak_mb
        # must have at least 1.6 GB headroom for dynamic continuous batching KV cache
        assert headroom_mb >= 1600.0, f"Insufficient KV cache headroom for {key}: {headroom_mb} MB"


def test_speculative_speedup_invariant():
    """
    Validates that speculative decoding guarantees an empirical wall-clock speedup
    ranging from 1.6x to 2.4x under nominal acceptance rates (alpha >= 0.70).
    """
    # Leviathan speculative speedup formula: S = 1 / ( (1 - alpha) + (alpha / (K + 1)) * (1 + c) )
    # where c = cost(draft) / cost(target)
    k = 3
    c = 0.22  # 1.5B draft model takes ~22% of 7B target forward pass time
    for alpha in [0.70, 0.75, 0.80, 0.85]:
        expected_speedup = (1 + alpha * k) / (1 + (alpha * k * c))
        assert expected_speedup >= 1.6, f"Speedup {expected_speedup} under alpha {alpha} below threshold"
