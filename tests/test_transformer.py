import importlib.util
import math

import pytest
import torch

from llm_from_scratch.attention import AttentionBackendName, create_attention_backend
from llm_from_scratch.transformer import MultiHeadAttention


requires_triton = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="Triton CUDA test",
)


def test_eager_attention_with_cached_keys():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 2, 4)
    k = torch.randn(2, 3, 5, 4)
    v = torch.randn_like(k)

    actual = create_attention_backend(AttentionBackendName.EAGER)(
        q=q,
        k=k,
        v=v,
        causal=True,
        query_start_pos=3,
    )

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    mask = torch.arange(5)[None, :] > (3 + torch.arange(2)[:, None])
    expected = scores.masked_fill(mask, float("-inf")).softmax(dim=-1) @ v
    torch.testing.assert_close(actual, expected)


def test_multi_head_attention_is_causal():
    torch.manual_seed(0)
    attention = MultiHeadAttention(
        embedding_dim=8,
        attention_heads=2,
        context_length=8,
        masked=True,
    ).eval()
    x = torch.randn(1, 5, 8)
    changed_future = x.clone()
    changed_future[:, 3:] = torch.randn_like(changed_future[:, 3:])

    with torch.no_grad():
        output = attention(x)
        changed_output = attention(changed_future)

    torch.testing.assert_close(output[:, :3], changed_output[:, :3])


@requires_triton
def test_flash_attention_prefill_matches_eager():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 37, 32, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with torch.no_grad():
        expected = create_attention_backend(AttentionBackendName.EAGER)(
            q=q,
            k=k,
            v=v,
            causal=True,
            query_start_pos=0,
        )
        actual = create_attention_backend(AttentionBackendName.FLASH)(
            q=q,
            k=k,
            v=v,
            causal=True,
            query_start_pos=0,
        )

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@requires_triton
def test_flash_attention_decode_matches_eager():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 1, 32, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 3, 37, 32, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)

    with torch.no_grad():
        expected = create_attention_backend(AttentionBackendName.EAGER)(
            q=q,
            k=k,
            v=v,
            causal=True,
            query_start_pos=36,
        )
        actual = create_attention_backend(AttentionBackendName.FLASH)(
            q=q,
            k=k,
            v=v,
            causal=True,
            query_start_pos=36,
        )

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
