import math
from collections.abc import Callable
from enum import Enum

import torch
from torch import Tensor


class AttentionBackendName(Enum):
    EAGER = "eager"
    FLASH = "flash"


AttentionBackend = Callable[..., Tensor]


def _causal_mask(
    query_length: int,
    key_length: int,
    query_start_pos: int,
    device: torch.device,
) -> Tensor:
    query_positions = query_start_pos + torch.arange(query_length, device=device)
    key_positions = torch.arange(key_length, device=device)
    # True means that this query/key pair is allowed to attend.
    return key_positions[None, :] <= query_positions[:, None]


def eager_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool,
    query_start_pos: int = 0,
) -> Tensor:
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        allowed = _causal_mask(
            query_length=q.shape[-2],
            key_length=k.shape[-2],
            query_start_pos=query_start_pos,
            device=q.device,
        )
        scores = scores.masked_fill(~allowed, float("-inf"))
    return scores.softmax(dim=-1) @ v


def create_attention_backend(name: AttentionBackendName) -> AttentionBackend:
    if not isinstance(name, AttentionBackendName):
        raise TypeError("name must be an AttentionBackendName")
    if name == AttentionBackendName.EAGER:
        return eager_attention
    if name == AttentionBackendName.FLASH:
        from llm_from_scratch.flash_attention_triton import flash_attention

        return flash_attention
    raise ValueError(f"Unknown attention backend: {name!r}")
