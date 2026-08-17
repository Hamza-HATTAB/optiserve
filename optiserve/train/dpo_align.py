import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from optiserve.config import settings


@dataclass
class DPOTrainMetrics:
    step: int
    loss: float
    chosen_reward: float
    rejected_reward: float
    reward_margin: float
    accuracy: float
    vram_used_mb: float


class DPOTrainer:
    """
    Direct Preference Optimization (DPO) trainer for reasoning alignment.
    Directly optimizes the implicit reward without training an actor-critic PPO loop.
    Ref: Rafailov et al. (2023).
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        beta: float = 0.1,
        learning_rate: float = 1e-5,
        device: str = "cuda:0",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.policy_model = policy_model.to(self.device)
        self.beta = beta

        # create or freeze reference model
        if ref_model is None:
            self.ref_model = copy.deepcopy(policy_model).to(self.device)
        else:
            self.ref_model = ref_model.to(self.device)

        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        trainable = [p for p in self.policy_model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

    def _get_batch_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes per-token log-probabilities and sums over non-masked response tokens.
        """
        if hasattr(model, "forward_logits"):
            logits = model.forward_logits(input_ids)
        elif callable(getattr(model, "forward", None)):
            out = model(input_ids)
            logits = out[0] if isinstance(out, tuple) else out
        else:
            raise RuntimeError("Model missing forward method")

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_mask = shift_labels != -100
        shift_labels = shift_labels.clone()
        shift_labels[~loss_mask] = 0

        # compute log P(y_t | y_{<t}, x)
        per_token_logps = torch.gather(
            shift_logits.log_softmax(-1), dim=2, index=shift_labels.unsqueeze(2)
        ).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1)

    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # compute log-ratios for chosen and rejected sequences
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = pi_logratios - ref_logratios
        losses = -F.logsigmoid(self.beta * logits)

        # implicit rewards
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps).detach()

        return losses.mean(), chosen_rewards, rejected_rewards

    def train_step(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> DPOTrainMetrics:
        self.policy_model.train()
        self.optimizer.zero_grad()

        chosen_input_ids = chosen_input_ids.to(self.device)
        chosen_labels = chosen_labels.to(self.device)
        rejected_input_ids = rejected_input_ids.to(self.device)
        rejected_labels = rejected_labels.to(self.device)

        # policy logps
        policy_chosen_logps = self._get_batch_logps(
            self.policy_model, chosen_input_ids, chosen_labels
        )
        policy_rejected_logps = self._get_batch_logps(
            self.policy_model, rejected_input_ids, rejected_labels
        )

        # reference logps (no grad)
        with torch.no_grad():
            ref_chosen_logps = self._get_batch_logps(self.ref_model, chosen_input_ids, chosen_labels)
            ref_rejected_logps = self._get_batch_logps(
                self.ref_model, rejected_input_ids, rejected_labels
            )

        loss, chosen_rewards, rejected_rewards = self.compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.policy_model.parameters() if p.requires_grad], max_norm=1.0
        )
        self.optimizer.step()

        margin = (chosen_rewards - rejected_rewards).mean().item()
        acc = (chosen_rewards > rejected_rewards).float().mean().item()

        vram_mb = 0.0
        if self.device.type == "cuda":
            vram_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)

        return DPOTrainMetrics(
            step=0,
            loss=round(loss.item(), 4),
            chosen_reward=round(chosen_rewards.mean().item(), 4),
            rejected_reward=round(rejected_rewards.mean().item(), 4),
            reward_margin=round(margin, 4),
            accuracy=round(acc, 3),
            vram_used_mb=round(vram_mb, 2),
        )
