import pytest
import torch
import shutil
from pathlib import Path
from optiserve.engine.airllm_runner import AirLLMRunner, TransformerBlock


@pytest.fixture
def temp_cache_dir(tmp_path):
    cache = tmp_path / "airllm_test_shards"
    yield cache
    if cache.exists():
        shutil.rmtree(cache)


def test_transformer_block_forward_and_kv():
    # test single transformer block forward pass and kv cache update
    hidden_dim = 256
    num_heads = 4
    intermediate_dim = 512
    block = TransformerBlock(hidden_dim, num_heads, intermediate_dim)

    x = torch.randn(1, 4, hidden_dim)
    out, (k, v) = block(x)

    assert out.shape == (1, 4, hidden_dim)
    assert k.shape == (1, num_heads, 4, hidden_dim // num_heads)
    assert v.shape == (1, num_heads, 4, hidden_dim // num_heads)

    # test subsequent single-token forward pass with cached kv
    x_next = torch.randn(1, 1, hidden_dim)
    out_next, (k_next, v_next) = block(x_next, past_key_value=(k, v))
    assert out_next.shape == (1, 1, hidden_dim)
    assert k_next.shape == (1, num_heads, 5, hidden_dim // num_heads)
    assert v_next.shape == (1, num_heads, 5, hidden_dim // num_heads)


def test_airllm_runner_streaming_and_memory_ceiling(temp_cache_dir):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    runner = AirLLMRunner(
        num_layers=4,
        hidden_dim=256,
        num_heads=4,
        intermediate_dim=512,
        vocab_size=1000,
        cache_dir=temp_cache_dir,
        device=device,
        dtype=torch.float32,
    )

    prompt = [10, 20, 30]
    input_ids = torch.tensor([prompt], dtype=torch.long)
    logits, past_kvs, metrics = runner.forward_layer_wise(input_ids)

    assert logits.shape == (1, 3, 1000)
    assert len(past_kvs) == 4
    assert len(metrics) == 4

    # assert memory stays well under the 6.8 GB (6963 MB) safety limit
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        assert peak_mb < 5000.0, f"Peak memory exceeded limit: {peak_mb} MB"


def test_airllm_autoregressive_generation(temp_cache_dir):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    runner = AirLLMRunner(
        num_layers=3,
        hidden_dim=128,
        num_heads=2,
        intermediate_dim=256,
        vocab_size=500,
        cache_dir=temp_cache_dir,
        device=device,
        dtype=torch.float32,
    )

    prompt = [1, 2]
    res = runner.generate(prompt, max_new_tokens=4)

    assert len(res.token_ids) == 6
    assert res.token_ids[:2] == prompt
    assert res.total_time_s > 0
    assert len(res.step_metrics) == 12  # 3 layers * 4 steps
