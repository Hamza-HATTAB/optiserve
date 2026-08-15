import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from optiserve.config import settings


class LoRALinear(nn.Module):
    """
    Low-rank adaptation layer: W = W_0 + (alpha / r) * B * A
    Base weights W_0 are frozen, only A and B are updated.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        lora_alpha: float = 32.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.r = r
        self.scaling = lora_alpha / r

        dev = device or base_layer.weight.device
        self.lora_A = nn.Parameter(torch.empty((r, in_features), dtype=torch.float32, device=dev))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r), dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        # compute low rank branch: (x @ A.T) @ B.T
        lora_out = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out.to(base_out.dtype)


@dataclass
class DistillStepMetrics:
    step: int
    loss: float
    learning_rate: float
    vram_used_mb: float
    elapsed_ms: float


class StudentDistillationEngine:
    """
    Supervised reasoning distillation engine using QLoRA parameter-efficient fine-tuning.
    Distills chain-of-thought rationale tokens from teacher trajectories into student model.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 2e-4,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        device: str = "cuda:0",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # wrap linear projection layers with LoRA adapters
        self.lora_modules = []
        self._apply_lora(self.model)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    def _apply_lora(self, module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(
                proj in name for proj in ["q_proj", "v_proj", "up_proj", "down_proj"]
            ):
                lora_layer = LoRALinear(
                    child, r=self.lora_rank, lora_alpha=self.lora_alpha, device=self.device
                )
                setattr(module, name, lora_layer)
                self.lora_modules.append(lora_layer)
            else:
                self._apply_lora(child)

    def compute_distill_loss(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        alpha_kd: float = 0.5,
    ) -> torch.Tensor:
        # standard cross-entropy loss over shift targets
        shift_logits = student_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        if teacher_logits is None:
            return ce_loss

        # soft distillation loss via KL divergence on rationale tokens
        shift_teacher = teacher_logits[:, :-1, :].contiguous()
        p_teacher = F.softmax(shift_teacher / temperature, dim=-1)
        log_p_student = F.log_softmax(shift_logits / temperature, dim=-1)
        kl_loss = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (temperature**2)

        return (1.0 - alpha_kd) * ce_loss + alpha_kd * kl_loss

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        if teacher_logits is not None:
            teacher_logits = teacher_logits.to(self.device)

        # mixed precision forward
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            if hasattr(self.model, "forward_logits"):
                logits = self.model.forward_logits(input_ids)
            elif callable(getattr(self.model, "forward", None)):
                out = self.model(input_ids)
                logits = out[0] if isinstance(out, tuple) else out
            else:
                raise RuntimeError("Model missing forward method")

            loss = self.compute_distill_loss(logits, labels, teacher_logits=teacher_logits)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
        )
        self.optimizer.step()

        vram_mb = 0.0
        if self.device.type == "cuda":
            vram_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)

        return float(loss.item()), float(vram_mb)
