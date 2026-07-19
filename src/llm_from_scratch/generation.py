import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingParams:
    max_new_tokens: int
    stop_string: Optional[str] = None
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None


def sample_next_token(
    logits: Tensor,
    params: SamplingParams,
) -> Tensor:
    # logits: B x V
    if params.top_k == 1:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / params.temperature
    apply_top_k(logits=logits, params=params)
    apply_top_p(logits=logits, params=params)
    probs = logits.softmax(dim=-1)
    # return B x 1
    return torch.multinomial(input=probs, num_samples=1)


def apply_top_k(
    logits: Tensor,
    params: SamplingParams,
) -> Tensor:
    # todo: am I supposed to make a copy?
    # logits B x V
    if not params.top_k:
        return logits

    sorted_logit_indices = logits.sort(dim=1, descending=True).indices
    top_k_indices = sorted_logit_indices[:, params.top_k :]
    rows = torch.arange(logits.shape[0]).unsqueeze(1)
    logits[rows, top_k_indices] = float("-inf")
    # return filtered B x V
    return logits


def apply_top_p(
    logits: Tensor,
    params: SamplingParams,
) -> Tensor:
    # todo: am I supposed to make a copy?
    # logits B x V
    # TODO this is really slow
    if not params.top_p:
        return logits
    probs = logits.softmax(dim=1)
    sorted_probs = probs.sort(dim=1, descending=True)
    cum_sum_values = sorted_probs.values.cumsum(dim=1)
    cum_sum_values -= sorted_probs.values
    cond_indices = (cum_sum_values > params.top_p).nonzero()
    for r, c in cond_indices:
        logits[r, sorted_probs.indices[r][c]] = float("-inf")
    return logits

    # return filtered B x V
