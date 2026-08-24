import asyncio
import pytest
from optiserve.api.batcher import ContinuousBatcher


@pytest.mark.asyncio
async def test_continuous_batcher_lifecycle_and_streaming():
    batcher = ContinuousBatcher(max_batch_size=4)
    await batcher.start()

    try:
        # enqueue two requests simultaneously
        req1 = await batcher.enqueue(prompt="Test prompt 1", max_tokens=5)
        req2 = await batcher.enqueue(prompt="Test prompt 2", max_tokens=3)

        tokens1 = []
        tokens2 = []

        # consume events from req1
        while True:
            item = await req1.stream_queue.get()
            if item["event"] == "token":
                tokens1.append(item["data"]["token_id"])
            elif item["event"] == "done":
                break

        # consume events from req2
        while True:
            item = await req2.stream_queue.get()
            if item["event"] == "token":
                tokens2.append(item["data"]["token_id"])
            elif item["event"] == "done":
                break

        assert len(tokens1) == 5
        assert len(tokens2) == 3
        assert req1.status == "finished"
        assert req2.status == "finished"
        assert req1.first_token_time is not None
        assert req2.first_token_time is not None

    finally:
        await batcher.stop()
