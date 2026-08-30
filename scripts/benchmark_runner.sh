#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo " OptiServe Hardware Diagnostic & Quantization Benchmarker "
echo " Target: NVIDIA RTX 4060 GPU (8GB VRAM)                   "
echo "=========================================================="

cd "$(dirname "$0")/.."

echo "[1/3] Running Hardware Latency & Bandwidth Profiler..."
.venv/bin/python -m optiserve.benchmarks.latency_profiler

echo "[2/3] Running 5-Way Quantization Bake-Off..."
.venv/bin/python -m optiserve.benchmarks.quant_bakeoff

echo "[3/3] Validating Output Telemetry File..."
if [ -f data/quant_benchmark_results.jsonl ]; then
    echo "✓ Benchmark results successfully recorded:"
    cat data/quant_benchmark_results.jsonl
else
    echo "✗ Error: Benchmark output file missing!"
    exit 1
fi
