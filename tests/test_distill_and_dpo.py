import pytest
import torch
import torch.nn as nn
from optiserve.train.distill import LoRALinear, StudentDistillationEngine
from optiserve.train.dpo_align import DPOTrainer


class SimpleTransformer(nn.Module):
    """Small transformer mockup for testing training mechanics and memory limits."""

    def __init__(self, vocab_size: int = 500, hidden_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.down_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        # attention projections
        q = self.q_proj(h)
        v = self.v_proj(h)
        h = h + q + v
        # mlp projections
        mlp = self.down_proj(torch.relu(self.up_proj(h)))
        h = h + mlp
        return self.head(h)


def test_lora_linear_forward_and_params():
    base = nn.Linear(64, 128)
    lora = LoRALinear(base, r=8, lora_alpha=16.0)

    # base layer weights must be frozen
    assert not lora.base_layer.weight.requires_grad
    # lora adapters must require grad
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad

    # verify forward pass
    x = torch.randn(2, 64)
    out = lora(x)
    assert out.shape == (2, 128)


def test_distillation_step_and_memory():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SimpleTransformer(vocab_size=200, hidden_dim=64)
    engine = StudentDistillationEngine(model, learning_rate=1e-3, device=device)

    # synthetic batch
    input_ids = torch.randint(0, 200, (2, 16))
    labels = input_ids.clone()

    loss_start, vram = engine.train_step(input_ids, labels)
    assert loss_start > 0

    # memory ceiling check: must strictly stay below 5.2 GB (5324 MB)
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        assert peak_mb < 5200.0, f"Memory exceeded limit: {peak_mb} MB"

    # run 5 steps to verify convergence
    loss_end = loss_start
    for _ in range(5):
        loss_end, _ = engine.train_step(input_ids, labels)

    assert loss_end < loss_start, "Loss should decrease after optimization steps"


def test_dpo_alignment_and_reward_margin():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SimpleTransformer(vocab_size=200, hidden_dim=64)
    trainer = DPOTrainer(model, beta=0.1, learning_rate=5e-4, device=device)

    # chosen: high reward prompt/response
    chosen_ids = torch.tensor([[10, 20, 30, 40, 50], [10, 25, 35, 45, 55]])
    chosen_labels = chosen_ids.clone()

    # rejected: wrong sequence
    rejected_ids = torch.tensor([[10, 99, 98, 97, 96], [10, 89, 88, 87, 86]])
    rejected_labels = rejected_ids.clone()

    initial_metrics = trainer.train_step(
        chosen_ids, chosen_labels, rejected_ids, rejected_labels
    )

    # multiple optimization steps
    for _ in range(10):
        final_metrics = trainer.train_step(
            chosen_ids, chosen_labels, rejected_ids, rejected_labels
        )

    # reward margin between chosen and rejected should increase
    assert final_metrics.reward_margin > initial_metrics.reward_margin
    assert final_metrics.loss < initial_metrics.loss
