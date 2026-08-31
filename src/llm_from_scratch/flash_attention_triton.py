import torch
from torch import Tensor
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"B_r": 16, "B_c": 16},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"B_r": 16, "B_c": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"B_r": 32, "B_c": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"B_r": 32, "B_c": 64},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["M", "N", "d"],
)
@triton.jit
def _flash_attn(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_on,
    stride_od,
    n_heads: tl.constexpr,
    sm_scale,
    query_start_pos,
    M: tl.constexpr,
    N: tl.constexpr,
    d: tl.constexpr,
    B_c: tl.constexpr,
    B_r: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = batch_head // n_heads
    head_idx = batch_head % n_heads

    # Load Q_i
    offset_qn = tl.arange(0, B_r) + (pid * B_r)
    offset_qd = tl.arange(0, triton.next_power_of_2(d))
    mask_q = (offset_qn < M)[:, None] & (offset_qd < d)[None, :]
    block_ptrs_q = q_ptr + (
        batch_idx * stride_qb
        + head_idx * stride_qh
        + (offset_qn * stride_qn)[:, None]
        + (offset_qd * stride_qd)[None, :]
    )
    Q_i = tl.load(block_ptrs_q, mask=mask_q, other=0)
    tl.static_assert(Q_i.shape == (B_r, triton.next_power_of_2(d)))

    # init
    l_i = tl.zeros((B_r, 1), dtype=tl.float32)
    m_i = tl.full((B_r, 1), -float("inf"), dtype=tl.float32)
    O_i = tl.zeros((B_r, d), dtype=tl.float32)

    for j in range(0, N, B_c):
        # Load K_j
        offset_kn = j + tl.arange(0, B_c)
        offset_kd = tl.arange(0, triton.next_power_of_2(d))
        mask_k = (offset_kn < N)[:, None] & (offset_kd < d)[None, :]
        block_ptrs_k = k_ptr + (
            batch_idx * stride_kb
            + head_idx * stride_kh
            + (offset_kn * stride_kn)[:, None]
            + (offset_kd * stride_kd)[None, :]
        )
        K_j = tl.load(block_ptrs_k, mask=mask_k, other=0)
        tl.static_assert(K_j.shape == (B_c, triton.next_power_of_2(d)))

        # Load V_j
        offset_vn = j + tl.arange(0, B_c)
        offset_vd = tl.arange(0, triton.next_power_of_2(d))
        mask_v = (offset_vn < N)[:, None] & (offset_vd < d)[None, :]
        block_ptrs_v = v_ptr + (
            batch_idx * stride_vb
            + head_idx * stride_vh
            + (offset_vn * stride_vn)[:, None]
            + (offset_vd * stride_vd)[None, :]
        )
        V_j = tl.load(block_ptrs_v, mask=mask_v, other=0)
        tl.static_assert(V_j.shape == (B_c, triton.next_power_of_2(d)))

        # Compute attention
        S_ij = tl.dot(Q_i, tl.trans(K_j))
        S_ij *= sm_scale
        tl.static_assert(S_ij.shape == (B_r, B_c))

        mask_s = (offset_kn < N)[None, :] & (
            offset_kn[None, :] <= query_start_pos + offset_qn[:, None]
        )
        S_ij = tl.where(mask_s, S_ij, float("-inf"))

        m_ij = tl.max(S_ij, axis=1, keep_dims=True)
        tl.static_assert(m_ij.shape == (B_r, 1))

        # The where avoids NaNs when a causal row has no keys in this block.
        P_ij = tl.where(mask_s, tl.exp(S_ij - m_ij), 0.0)
        tl.static_assert(P_ij.shape == (B_r, B_c))

        l_ij = tl.sum(P_ij, axis=1, keep_dims=True)
        tl.static_assert(l_ij.shape == (B_r, 1))

        m_i_new = tl.maximum(m_i, m_ij)
        tl.static_assert(m_i_new.shape == (B_r, 1))

        l_i_new = (tl.exp(m_i - m_i_new) * l_i) + (tl.exp(m_ij - m_i_new) * l_ij)
        tl.static_assert(l_i_new.shape == (B_r, 1))

        _norm_o_i = l_i * tl.exp(m_i - m_i_new) * O_i
        tl.static_assert(_norm_o_i.shape == (B_r, d))

        _pv = tl.exp(m_ij - m_i_new) * tl.dot(P_ij.to(tl.float16), V_j)
        tl.static_assert(_pv.shape == (B_r, d))

        _new_o_i = (1 / l_i_new) * (_norm_o_i + _pv)
        tl.static_assert(_new_o_i.shape == (B_r, d))

        O_i = _new_o_i
        l_i = l_i_new
        m_i = m_i_new

    # Store O_i to HBM
    offset_on = tl.arange(0, B_r) + (pid * B_r)
    offset_od = tl.arange(0, triton.next_power_of_2(d))
    mask_o = (offset_on < M)[:, None] & (offset_od < d)[None, :]
    block_ptrs_o = o_ptr + (
        batch_idx * stride_ob
        + head_idx * stride_oh
        + (offset_on * stride_on)[:, None]
        + (offset_od * stride_od)[None, :]
    )
    tl.store(block_ptrs_o, value=O_i, mask=mask_o)


def flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool,
    query_start_pos: int = 0,
) -> Tensor:
    if not causal:
        raise ValueError(
            "The custom FlashAttention backend only supports causal attention"
        )
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("Q, K and V must have shape [B, H, sequence, d]")
    if q.shape[:2] != k.shape[:2] or q.shape[3] != k.shape[3]:
        raise ValueError("Q, K and V must have matching batch, head and d dimensions")
    if q.device.type != "cuda":
        raise ValueError("The custom FlashAttention backend requires CUDA")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise ValueError("The custom FlashAttention backend requires FP16 tensors")
    if any(tensor.requires_grad for tensor in (q, k, v)):
        raise RuntimeError("The custom FlashAttention backend has no backward pass")

    B, H, M, d = q.shape
    N = k.shape[2]
    if d not in (16, 32, 64, 128):
        raise ValueError("Supported head dimensions are 16, 32, 64 and 128")

    output = torch.zeros_like(q)

    def grid(meta):
        return (triton.cdiv(M, meta["B_r"]), B * H)

    _flash_attn[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        o_ptr=output,
        stride_qb=q.stride(0),
        stride_qh=q.stride(1),
        stride_qn=q.stride(2),
        stride_qd=q.stride(3),
        stride_kb=k.stride(0),
        stride_kh=k.stride(1),
        stride_kn=k.stride(2),
        stride_kd=k.stride(3),
        stride_vb=v.stride(0),
        stride_vh=v.stride(1),
        stride_vn=v.stride(2),
        stride_vd=v.stride(3),
        stride_ob=output.stride(0),
        stride_oh=output.stride(1),
        stride_on=output.stride(2),
        stride_od=output.stride(3),
        n_heads=H,
        sm_scale=d**-0.5,
        query_start_pos=query_start_pos,
        M=M,
        N=N,
        d=d,
    )
    return output
