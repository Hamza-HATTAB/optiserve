import json
import pytest
from pathlib import Path
from optiserve.benchmarks.quant_bakeoff import QuantizationBakeOffRunner, QUANT_SPECS


def test_quantization_bakeoff_runner(tmp_path):
    output_file = tmp_path / "test_quant_results.jsonl"
    runner = QuantizationBakeOffRunner(output_path=output_file)
    results = runner.run_all()

    assert len(results) == len(QUANT_SPECS)
    assert output_file.exists()

    # check each result against hardware safety constraints
    for r in results:
        # peak interactive VRAM must stay under 6.8 GB (6963 MB) or offload
        if r.format_name != "FP16":
            assert r.vram_peak_mb <= 5324.0, f"{r.format_name} VRAM {r.vram_peak_mb} exceeded limit"
        assert r.itl_ms_per_token > 0
        assert r.throughput_tokens_per_sec > 0
        assert r.perplexity > 0
        assert 0.0 <= r.gsm8k_accuracy <= 100.0

    # verify jsonl parsing
    with open(output_file, "r") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == len(QUANT_SPECS)
    assert lines[0]["format_name"] == "FP16"
