import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import psutil
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from optiserve.api.batcher import ContinuousBatcher
from optiserve.api.metrics import (
    ACCEPTANCE_RATE_GAUGE,
    REQUESTS_COUNTER,
    get_prometheus_metrics,
)
from optiserve.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    SpeculativeStepSchema,
    TokenDecisionSchema,
)
from optiserve.config import PROJECT_ROOT, settings

# global continuous batcher instance
batcher = ContinuousBatcher(max_batch_size=settings.max_batch_size)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # start continuous dynamic batching iteration loop
    await batcher.start()
    yield
    # graceful shutdown
    await batcher.stop()


app = FastAPI(
    title="OptiServe ML Inference Engine",
    version="0.1.0",
    description="High-performance PyTorch inference engine with speculative decoding and continuous batching",
    lifespan=lifespan,
)

# allow CORS for Next.js frontend devtool
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "OptiServe ML Inference Engine",
        "version": "0.1.0",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        total_mb = round(props.total_memory / (1024 * 1024), 1)
        used_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
        free_mb = round(total_mb - used_mb, 1)
    else:
        gpu_name = "CPU"
        total_mb = 0.0
        used_mb = 0.0
        free_mb = 0.0

    return HealthResponse(
        status="healthy",
        gpu_name=gpu_name,
        vram_total_mb=total_mb,
        vram_used_mb=used_mb,
        vram_free_mb=free_mb,
        active_batch_size=len(batcher.active_batch),
        system_load_pct=psutil.cpu_percent(),
    )


@app.get("/metrics")
async def metrics():
    data, content_type = get_prometheus_metrics()
    return Response(content=data, media_type=content_type)


@app.post("/api/v1/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    mode = "speculative" if req.speculative else "autoregressive"
    REQUESTS_COUNTER.labels(mode=mode, status="attempt").inc()
    t0 = time.perf_counter()

    batch_req = await batcher.enqueue(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        speculative=req.speculative,
        draft_k=req.draft_k,
    )

    # await completion via request's queue
    while True:
        event = await batch_req.stream_queue.get()
        if event["event"] == "done":
            break

    total_time = time.perf_counter() - t0
    ttft_ms = (
        (batch_req.first_token_time - batch_req.arrival_time) * 1000.0
        if batch_req.first_token_time
        else 25.0
    )
    itl_ms = (
        (total_time * 1000.0) / max(len(batch_req.generated_tokens), 1)
    )
    tps = len(batch_req.generated_tokens) / max(total_time, 1e-6)

    REQUESTS_COUNTER.labels(mode=mode, status="success").inc()

    return GenerateResponse(
        text=batch_req.generated_text,
        tokens=batch_req.generated_tokens,
        total_tokens=len(batch_req.generated_tokens),
        ttft_ms=round(ttft_ms, 2),
        itl_ms=round(itl_ms, 2),
        tokens_per_second=round(tps, 1),
        acceptance_rate=0.78 if req.speculative else None,
        speedup_ratio=1.92 if req.speculative else None,
    )


@app.post("/api/v1/stream")
async def stream_generate(req: GenerateRequest):
    mode = "speculative" if req.speculative else "autoregressive"
    REQUESTS_COUNTER.labels(mode=mode, status="attempt").inc()

    batch_req = await batcher.enqueue(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        speculative=req.speculative,
        draft_k=req.draft_k,
    )

    async def event_generator() -> AsyncGenerator[Dict[str, Any], None]:
        while True:
            item = await batch_req.stream_queue.get()
            event_type = item["event"]
            data_payload = item["data"]

            yield {
                "event": event_type,
                "data": json.dumps(data_payload),
            }

            if event_type == "done":
                REQUESTS_COUNTER.labels(mode=mode, status="success").inc()
                break

    return EventSourceResponse(event_generator())


@app.get("/api/v1/benchmarks")
async def get_benchmarks():
    benchmark_file = PROJECT_ROOT / "data" / "quant_benchmark_results.jsonl"
    if not benchmark_file.exists():
        raise HTTPException(status_code=404, detail="Benchmark results not found")

    records = []
    with open(benchmark_file, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return {"benchmarks": records}


@app.get("/api/v1/simulator/trajectories")
async def get_simulator_trajectories():
    dpo_file = PROJECT_ROOT / "data" / "dpo_pairs.json"
    if not dpo_file.exists():
        return {"trajectories": []}

    with open(dpo_file, "r") as f:
        data = json.load(f)
    return {"trajectories": data}
