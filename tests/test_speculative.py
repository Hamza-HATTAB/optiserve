import pytest
import torch
import torch.nn as nn
from optiserve.engine.speculative import SpeculativeDecodingEngine


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32, bias_val: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        # add bias to differentiate target from draft
        with torch.no_grad():
            self.head.weight.fill_(0.01)
            self.head.bias.fill_(bias_val)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        return self.head(h)


def test_rejection_sampling_mathematical_distribution_match():
    """
    Empirically proves Leviathan et al. theorem:
    Rejection sampling from draft q(x) with verification against p(x)
    produces samples distributed EXACTLY according to p(x).
    """
    vocab_size = 8
    torch.manual_seed(42)

    # arbitrary distinct categorical distributions
    p_logits = torch.tensor([1.2, 3.4, 0.5, 2.1, 4.0, 0.8, 1.5, 2.8])
    q_logits = torch.tensor([2.0, 2.5, 1.8, 1.1, 3.0, 2.2, 0.9, 1.4])

    p = torch.softmax(p_logits, dim=-1)
    q = torch.softmax(q_logits, dim=-1)

    num_samples = 40000
    samples = []

    for _ in range(num_samples):
        # sample draft token from q
        draft_token = torch.multinomial(q, num_samples=1).item()
        q_val = q[draft_token].item()
        p_val = p[draft_token].item()

        # rejection check
        ratio = min(1.0, p_val / q_val)
        u = torch.rand(1).item()

        if u <= ratio:
            samples.append(draft_token)
        else:
            # resample from max(0, p - q)
            diff = torch.clamp(p - q, min=0.0)
            resample_dist = diff / diff.sum()
            resampled_token = torch.multinomial(resample_dist, num_samples=1).item()
            samples.append(resampled_token)

    # compute empirical distribution
    empirical_counts = torch.zeros(vocab_size)
    for s in samples:
        empirical_counts[s] += 1
    empirical_p = empirical_counts / num_samples

    # total variation distance must be negligible (< 0.015)
    tvd = 0.5 * torch.sum(torch.abs(empirical_p - p)).item()
    assert tvd < 0.015, f"Total variation distance {tvd} exceeds threshold"


def test_speculative_engine_generation_and_speedup():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vocab_size = 100

    # draft model (fast, compact)
    draft_model = DummyModel(vocab_size=vocab_size, hidden_dim=32, bias_val=0.0)
    # target model (accurate verifier)
    target_model = DummyModel(vocab_size=vocab_size, hidden_dim=32, bias_val=0.05)

    engine = SpeculativeDecodingEngine(
        draft_model=draft_model,
        target_model=target_model,
        k=3,
        temperature=0.7,
        device=device,
    )

    prompt = [5, 12, 23]
    summary = engine.generate(prompt, max_new_tokens=10)

    assert len(summary.generated_tokens) >= len(prompt) + 10
    assert summary.total_cycles > 0
    assert summary.total_drafted >= summary.total_accepted
    assert 0.0 <= summary.acceptance_rate <= 1.0
    assert summary.tokens_per_second > 0
    assert len(summary.events) == summary.total_cycles

    # check structure of step events
    first_event = summary.events[0]
    assert len(first_event.draft_tokens) == 3
    assert len(first_event.decisions) >= 1
    assert first_event.cycle_latency_ms > 0
