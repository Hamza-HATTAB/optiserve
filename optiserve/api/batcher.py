import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

from optiserve.api.metrics import (
    ACTIVE_BATCH_GAUGE,
    ITL_HISTOGRAM,
    TOKENS_COUNTER,
    TTFT_HISTOGRAM,
    update_gpu_memory_metric,
)
from optiserve.config import settings


@dataclass
class BatchRequest:
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    speculative: bool
    draft_k: int
    arrival_time: float = field(default_factory=time.perf_counter)
    first_token_time: Optional[float] = None
    generated_tokens: List[int] = field(default_factory=list)
    generated_text: str = ""
    status: str = "queued"  # queued, decoding, finished
    stream_queue: asyncio.Queue = field(default_factory=asyncio.Queue)


class ContinuousBatcher:
    """
    Iteration-level continuous dynamic batching scheduler.
    Dynamically inserts newly arrived requests into running decode cycles
    and immediately evicts completed sequences to prevent pipeline bubbles.
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        engine: Optional[Any] = None,
    ):
        self.max_batch_size = max_batch_size
        self.engine = engine
        self.waiting_queue: asyncio.Queue[BatchRequest] = asyncio.Queue()
        self.active_batch: List[BatchRequest] = []
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None

    async def enqueue(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.7,
        speculative: bool = True,
        draft_k: int = 3,
    ) -> BatchRequest:
        req = BatchRequest(
            request_id=str(uuid.uuid4())[:8],
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            speculative=speculative,
            draft_k=draft_k,
        )
        await self.waiting_queue.put(req)
        return req

    async def start(self):
        if not self._running:
            self._running = True
            self._loop_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self):
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

    async def _scheduler_loop(self):
        while self._running:
            # 1. Admit waiting requests if capacity available
            while len(self.active_batch) < self.max_batch_size and not self.waiting_queue.empty():
                try:
                    req = self.waiting_queue.get_nowait()
                    req.status = "decoding"
                    self.active_batch.append(req)
                except asyncio.QueueEmpty:
                    break

            ACTIVE_BATCH_GAUGE.set(len(self.active_batch))

            if not self.active_batch:
                await asyncio.sleep(0.005)
                continue

            # 2. Execute single iteration decode step across all active requests
            t_step_start = time.perf_counter()
            await self._step_iteration()
            t_step_elapsed = time.perf_counter() - t_step_start

            # record ITL for the active batch
            ITL_HISTOGRAM.observe(t_step_elapsed)
            update_gpu_memory_metric()

            # small yield to event loop
            await asyncio.sleep(0.001)

    async def _step_iteration(self):
        completed_indices = []

        for idx, req in enumerate(self.active_batch):
            now = time.perf_counter()
            is_first_token = len(req.generated_tokens) == 0

            # record TTFT on first token generation
            if is_first_token:
                req.first_token_time = now
                ttft_s = now - req.arrival_time
                TTFT_HISTOGRAM.observe(ttft_s)

            # simulate realistic token emission with vocabulary projection
            token_val = (len(req.prompt) * 17 + len(req.generated_tokens) * 31) % 5000 + 100
            req.generated_tokens.append(token_val)

            # synthesize token piece
            token_str = f" tok_{token_val}" if not is_first_token else "Rationale:"
            req.generated_text += token_str

            mode_tag = "speculative" if req.speculative else "autoregressive"
            TOKENS_COUNTER.labels(mode=mode_tag).inc()

            # push token to SSE stream queue
            await req.stream_queue.put(
                {
                    "event": "token",
                    "data": {
                        "request_id": req.request_id,
                        "token_id": token_val,
                        "token_str": token_str,
                        "token_index": len(req.generated_tokens),
                        "is_first": is_first_token,
                    },
                }
            )

            # check termination condition
            if len(req.generated_tokens) >= req.max_tokens:
                req.status = "finished"
                await req.stream_queue.put(
                    {
                        "event": "done",
                        "data": {
                            "request_id": req.request_id,
                            "total_tokens": len(req.generated_tokens),
                            "text": req.generated_text,
                            "elapsed_s": round(now - req.arrival_time, 3),
                        },
                    }
                )
                completed_indices.append(idx)

        # evict finished requests in reverse order to preserve indexing
        for idx in reversed(completed_indices):
            self.active_batch.pop(idx)
