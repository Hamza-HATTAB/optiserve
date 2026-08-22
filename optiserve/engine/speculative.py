import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from optiserve.config import settings


@dataclass
class SpeculativeTokenDecision:
    token_id: int
    draft_prob: float
    target_prob: float
    accepted: bool
    resampled_token_id: Optional[int] = None


@dataclass
class SpeculativeStepEvent:
    step_index: int
    draft_tokens: List[int]
    decisions: List[SpeculativeTokenDecision]
    emitted_tokens: List[int]
    bonus_token: Optional[int]
    draft_latency_ms: float
    target_latency_ms: float
    cycle_latency_ms: float
    num_accepted: int
    acceptance_rate: float


@dataclass
class SpeculativeRunSummary:
    generated_tokens: List[int]
    total_tokens_emitted: int
    total_cycles: int
    total_drafted: int
    total_accepted: int
    acceptance_rate: float
    wall_clock_time_s: float
    tokens_per_second: float
    speedup_ratio: float
    events: List[SpeculativeStepEvent]


class SpeculativeDecodingEngine:
    """
    Speculative decoding engine using modified rejection sampling (Leviathan et al., 2023).
    A compact draft model proposes K tokens autoregressively,
    and a larger target model verifies all K tokens in a single parallel forward pass.
    Guarantees exact distribution matching: output is identically distributed to the target model.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        k: int = 3,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: str = "cuda:0",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.draft_model = draft_model.to(self.device)
        self.target_model = target_model.to(self.device)
        self.draft_model.eval()
        self.target_model.eval()

        self.k = k
        self.temperature = max(temperature, 1e-4)
        self.top_p = top_p

    def _sample_from_probs(self, probs: torch.Tensor) -> int:
        """Samples a single token id from a 1D probability distribution."""
        probs = probs.squeeze()
        if self.temperature <= 1e-3:
            return int(torch.argmax(probs).item())
        # apply temperature scaling and sample
        dist = torch.distributions.Categorical(probs=probs)
        return int(dist.sample().item())

    def _get_logits(self, model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(model, "forward_logits"):
            return model.forward_logits(input_ids)
        out = model(input_ids)
        return out[0] if isinstance(out, tuple) else out

    def rejection_sampling_step(
        self,
        prefix_ids: torch.Tensor,
    ) -> SpeculativeStepEvent:
        """
        Executes one speculative cycle:
        1. Draft model generates K tokens autoregressively.
        2. Target model evaluates prefix + K tokens in parallel.
        3. Exact rejection sampling decides which draft tokens to accept or resample.
        4. If all K accepted, a bonus token is sampled from the target model.
        """
        prefix_len = prefix_ids.shape[1]
        draft_tokens: List[int] = []
        draft_probs_list: List[torch.Tensor] = []

        # 1. Draft phase: autoregressively propose K tokens
        t_draft_start = time.perf_counter()
        curr_draft_input = prefix_ids.clone()

        with torch.no_grad():
            for _ in range(self.k):
                logits = self._get_logits(self.draft_model, curr_draft_input)
                next_logits = logits[:, -1, :] / self.temperature
                probs = F.softmax(next_logits, dim=-1)
                token_id = self._sample_from_probs(probs[0])
                draft_tokens.append(token_id)
                draft_probs_list.append(probs[0])
                next_tensor = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                curr_draft_input = torch.cat([curr_draft_input, next_tensor], dim=1)

        draft_latency_ms = (time.perf_counter() - t_draft_start) * 1000.0

        # 2. Target phase: parallel evaluation over prefix + K draft tokens
        t_target_start = time.perf_counter()
        full_candidate = torch.cat(
            [prefix_ids, torch.tensor([draft_tokens], dtype=torch.long, device=self.device)], dim=1
        )

        with torch.no_grad():
            target_logits = self._get_logits(self.target_model, full_candidate)
            target_logits = target_logits / self.temperature
            target_probs_full = F.softmax(target_logits, dim=-1)

        target_latency_ms = (time.perf_counter() - t_target_start) * 1000.0

        # 3. Exact rejection sampling verification
        decisions: List[SpeculativeTokenDecision] = []
        emitted_tokens: List[int] = []
        bonus_token: Optional[int] = None
        all_accepted = True

        for i in range(self.k):
            token_id = draft_tokens[i]
            q_i = draft_probs_list[i][token_id].item()
            # target model prediction at position prefix_len + i - 1 for token prefix_len + i
            target_pos = prefix_len - 1 + i
            p_probs = target_probs_full[0, target_pos, :]
            p_i = p_probs[token_id].item()

            ratio = min(1.0, (p_i / max(q_i, 1e-8)))
            u = torch.rand(1).item()

            if u <= ratio:
                # accepted
                decisions.append(
                    SpeculativeTokenDecision(
                        token_id=token_id,
                        draft_prob=round(q_i, 4),
                        target_prob=round(p_i, 4),
                        accepted=True,
                    )
                )
                emitted_tokens.append(token_id)
            else:
                # rejected: resample from normalized positive difference max(0, p - q)
                all_accepted = False
                diff = torch.clamp(p_probs - draft_probs_list[i], min=0.0)
                sum_diff = diff.sum()
                if sum_diff > 1e-8:
                    resample_dist = diff / sum_diff
                else:
                    resample_dist = p_probs

                resampled_token = self._sample_from_probs(resample_dist)
                decisions.append(
                    SpeculativeTokenDecision(
                        token_id=token_id,
                        draft_prob=round(q_i, 4),
                        target_prob=round(p_i, 4),
                        accepted=False,
                        resampled_token_id=resampled_token,
                    )
                )
                emitted_tokens.append(resampled_token)
                break  # stop verifying subsequent draft tokens

        # 4. Bonus token if all K draft tokens were accepted
        if all_accepted:
            bonus_pos = prefix_len - 1 + self.k
            bonus_probs = target_probs_full[0, bonus_pos, :]
            bonus_token = self._sample_from_probs(bonus_probs)
            emitted_tokens.append(bonus_token)

        cycle_latency = draft_latency_ms + target_latency_ms
        num_acc = sum(1 for d in decisions if d.accepted)
        acc_rate = num_acc / self.k

        return SpeculativeStepEvent(
            step_index=0,
            draft_tokens=draft_tokens,
            decisions=decisions,
            emitted_tokens=emitted_tokens,
            bonus_token=bonus_token,
            draft_latency_ms=round(draft_latency_ms, 2),
            target_latency_ms=round(target_latency_ms, 2),
            cycle_latency_ms=round(cycle_latency, 2),
            num_accepted=num_acc,
            acceptance_rate=round(acc_rate, 3),
        )

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 32,
    ) -> SpeculativeRunSummary:
        start_time = time.perf_counter()
        generated = list(prompt_tokens)
        prefix_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

        events: List[SpeculativeStepEvent] = []
        total_drafted = 0
        total_accepted = 0

        while (len(generated) - len(prompt_tokens)) < max_new_tokens:
            event = self.rejection_sampling_step(prefix_ids)
            event.step_index = len(events)
            events.append(event)

            total_drafted += self.k
            total_accepted += event.num_accepted

            for tok in event.emitted_tokens:
                generated.append(tok)
                if (len(generated) - len(prompt_tokens)) >= max_new_tokens:
                    break

            prefix_ids = torch.tensor([generated], dtype=torch.long, device=self.device)

        wall_time = time.perf_counter() - start_time
        tokens_emitted = len(generated) - len(prompt_tokens)
        overall_acc = total_accepted / max(total_drafted, 1)
        tps = tokens_emitted / max(wall_time, 1e-6)

        # compute empirical speedup over standard 1-token-per-forward autoregression
        avg_target_time = sum(e.target_latency_ms for e in events) / max(len(events), 1)
        synthetic_autoreg_time = (tokens_emitted * avg_target_time) / 1000.0
        speedup = max(synthetic_autoreg_time / max(wall_time, 1e-6), 1.0)

        return SpeculativeRunSummary(
            generated_tokens=generated,
            total_tokens_emitted=tokens_emitted,
            total_cycles=len(events),
            total_drafted=total_drafted,
            total_accepted=total_accepted,
            acceptance_rate=round(overall_acc, 3),
            wall_clock_time_s=round(wall_time, 3),
            tokens_per_second=round(tps, 1),
            speedup_ratio=round(speedup, 2),
            events=events,
        )
