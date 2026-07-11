from dataclasses import dataclass
import math
import torch
from torch.nn import (
    Linear,
    Module,
    ModuleList,
    Embedding,
    LayerNorm,
    ReLU,
    Dropout,
)
from torch import Tensor
from llm_from_scratch.kv_cache import KVCache


@dataclass
class TransformerConfig:
    embedding_dim: int
    context_length: int
    attention_heads: int
    ff_hidden_dim: int
    n_decoders: int
    p_dropout: float


class FeedForward(Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.l1 = Linear(embedding_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, embedding_dim)
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        # input B x T x C
        # B x T x H
        x = self.l1(x)
        x = self.relu(x)
        # B x T x C
        return self.l2(x)


class Decoder(Module):
    def __init__(
        self,
        embedding_dim: int,
        ff_hidden_dim: int,
        attention_heads: int,
        context_length: int,
        p_dropout: float,
        vectorized: bool = True,
    ):
        super().__init__()
        if vectorized:  # for benchmarking
            self.masked_multi_head_attention = MultiHeadAttention(
                embedding_dim=embedding_dim,
                attention_heads=attention_heads,
                context_length=context_length,
                masked=True,
            )
        else:
            self.masked_multi_head_attention = MultiHeadAttentionSeq(
                embedding_dim=embedding_dim,
                attention_heads=attention_heads,
                context_length=context_length,
                masked=True,
            )
        self.layer_norm_1 = LayerNorm(embedding_dim)
        self.layer_norm_2 = LayerNorm(embedding_dim)
        self.ff = FeedForward(embedding_dim=embedding_dim, hidden_dim=ff_hidden_dim)
        self.dropout = Dropout(p=p_dropout)

    def forward(
        self,
        x: Tensor,
        layer_idx: int,
        kv_cache=None,
        request_ids: list[str] | None = None,
        start_positions: Tensor | None = None,
        use_cache: bool = False,
    ) -> Tensor:
        # B x T x C
        x = (
            self.dropout(
                self.masked_multi_head_attention(
                    x=x,
                    layer_idx=layer_idx,
                    kv_cache=kv_cache,
                    request_ids=request_ids,
                    start_positions=start_positions,
                    use_cache=use_cache,
                )
            )
            + x
        )
        x = self.layer_norm_1(x)
        x = self.dropout(self.ff(x)) + x
        return self.layer_norm_2(x)


class Attention(Module):
    """Attention as described by Attention is all you need"""

    mask: Tensor

    def __init__(
        self,
        embedding_dim: int,
        head_dim: int,
        context_length: int,
        masked: bool,
    ):
        super().__init__()
        self.masked = masked
        self.head_dim = head_dim
        self.wq = Linear(embedding_dim, head_dim, bias=False)
        self.wk = Linear(embedding_dim, head_dim, bias=False)
        self.wv = Linear(embedding_dim, head_dim, bias=False)

        mask = torch.ones(context_length, context_length, dtype=torch.bool).triu(
            diagonal=1
        )
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        # input is B x T x C
        # each B x T x d
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        # B x T x T
        q_kt = q @ k.transpose(1, 2)
        q_kt = q_kt / math.sqrt(self.head_dim)
        if self.masked:  # can't attend future tokens
            T = x.shape[1]
            q_kt = q_kt.masked_fill(
                self.mask[:T, :T], float("-inf")
            )  # -inf instead of 0 as we can have negatives
        q_kt = q_kt.softmax(dim=-1)
        # B x T x d
        return q_kt @ v


class MultiHeadAttentionSeq(Module):
    """Non vectorized MultiHeadAttention"""

    def __init__(
        self,
        embedding_dim: int,
        attention_heads: int,
        context_length: int,
        masked: bool,
    ):
        super().__init__()
        assert embedding_dim % attention_heads == 0, (
            "embedding_dim must be divisible by attention_heads"
        )
        self.heads = ModuleList(
            [
                Attention(
                    embedding_dim=embedding_dim,
                    head_dim=embedding_dim // attention_heads,
                    context_length=context_length,
                    masked=masked,
                )
                for _ in range(attention_heads)
            ]
        )
        self.wo = Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        x: Tensor,
        layer_idx: int = 0,
        kv_cache=None,
        request_ids: list[str] | None = None,
        start_positions: Tensor | None = None,
        use_cache: bool = False,
    ) -> Tensor:
        # input is B x T x C
        # each B x T x d
        x_heads = [head(x) for head in self.heads]
        # B x T x C
        x = torch.cat(tensors=x_heads, dim=-1)
        # B x T x C
        return self.wo(x)


class MultiHeadAttention(Module):
    """Vectorized MultiHeadAttention"""

    mask: Tensor

    def __init__(
        self,
        embedding_dim: int,
        attention_heads: int,
        context_length: int,
        masked: bool,
    ):
        super().__init__()
        assert embedding_dim % attention_heads == 0, (
            "embedding_dim must be divisible by attention_heads"
        )
        self.attention_heads = attention_heads
        self.head_dim = embedding_dim // attention_heads
        self.wq = Linear(
            embedding_dim,
            embedding_dim,  # d * h
            bias=False,
        )
        self.wk = Linear(
            embedding_dim,
            embedding_dim,  # d * h
            bias=False,
        )
        self.wv = Linear(
            embedding_dim,
            embedding_dim,  # d * h
            bias=False,
        )

        self.masked = masked
        mask = torch.ones(context_length, context_length, dtype=torch.bool).triu(
            diagonal=1
        )
        self.register_buffer("mask", mask)
        self.wo = Linear(embedding_dim, embedding_dim)

    def _shape_qkv(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        B, T, _ = x.shape
        # B x T x h * d
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        # B x T x h x d
        q, k, v = (
            q.reshape(B, T, self.attention_heads, self.head_dim),
            k.reshape(B, T, self.attention_heads, self.head_dim),
            v.reshape(B, T, self.attention_heads, self.head_dim),
        )
        # B x h x T x d
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        return q, k, v

    def _attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        # B x h x T_new x T
        q_kt = q @ k.transpose(-2, -1)
        q_kt = q_kt / math.sqrt(self.head_dim)
        if attention_mask is not None:
            q_kt = q_kt.masked_fill(
                attention_mask, float("-inf")
            )  # -inf instead of 0 as we can have negatives
        q_kt = q_kt.softmax(dim=-1)
        # B x h x T_new x d
        return q_kt @ v

    def forward(
        self,
        x: Tensor,
        layer_idx: int = 0,
        kv_cache: KVCache | None = None,
        request_ids: list[str] | None = None,
        start_positions: Tensor | None = None,
        use_cache: bool = False,
    ) -> Tensor:
        # x is B x T x C
        # start_positions is B
        B, T, C = (
            x.shape[0],
            x.shape[1],
            x.shape[2],
        )
        if use_cache:
            if kv_cache is None:
                raise ValueError("use_cache true but no kv_cache passed")
            if start_positions is None:
                raise ValueError("use_cache true but no start_positions passed")
            # start pos is index of first token inside model context window that we need to process
            # ! during decode x != full context as we only pass the last/new token !
            # todo: make this work correctly with batches
            start_pos = int(start_positions[0].item())
            # each B x h x T x d
            q_new, k_new, v_new = self._shape_qkv(x=x)
            kv_cache.append_batch(
                layer_idx=layer_idx,
                keys=k_new,
                values=v_new,
                start_positions=start_positions,
            )
            # do we also need to modify here to correctly handle batches?
            # each B x h x start_positions + T x d
            k, v = kv_cache.get_batch(
                layer_idx=layer_idx, end_positions=start_positions + T
            )
            attention_mask = None
            if self.masked:
                attention_mask = self.mask[start_pos : start_pos + T, : start_pos + T]
            # B x h x T x d
            x = self._attention(q=q_new, k=k, v=v, attention_mask=attention_mask)
        else:
            # each B x h x T x d
            q, k, v = self._shape_qkv(x=x)
            attention_mask = None
            if self.masked:
                attention_mask = self.mask[:T, :T]
            # B x h x T x d
            x = self._attention(q=q, k=k, v=v, attention_mask=attention_mask)

        # B x T x h x d
        x = x.transpose(1, 2)
        # B x T x C
        x = x.reshape(B, T, C)
        # B x T x C
        return self.wo(x)


class PosEncoding(Module):
    pos_encoding: Tensor

    def __init__(self, context_length: int, embedding_dim: int, p_dropout: float):
        super().__init__()
        self.dropout = Dropout(p=p_dropout)
        # T x C
        pos_encoding = torch.zeros(context_length, embedding_dim)

        for pos in range(context_length):
            for i in range(embedding_dim):
                if i % 2 == 0:
                    pos_encoding[pos, i] = math.sin(
                        pos / (10000 ** ((2 * (i // 2)) / embedding_dim))
                    )
                else:
                    pos_encoding[pos, i] = math.cos(
                        pos / (10000 ** ((2 * (i // 2)) / embedding_dim))
                    )

        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x: Tensor, start_positions: Tensor | None = None) -> Tensor:
        assert x.shape[1] <= self.pos_encoding.shape[0], (
            "Passed more tokens than model context length"
        )
        # B x T x C
        # todo: make this work correctly with batches
        start_pos = 0
        if start_positions is not None:
            start_pos = int(start_positions[0].item())
        # slicing because during inference we don't have all T yet.
        return self.dropout(
            x + self.pos_encoding[start_pos : start_pos + x.shape[1], :]
        )


class Transformer(Module):
    """Decoder-only Transformer as presented in Attention is all you need paper"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
        attention_heads: int,
        ff_hidden_dim: int,
        n_decoders: int,
        p_dropout: float,
    ):
        super().__init__()
        self.embedding = Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.pos_encoding = PosEncoding(
            context_length=context_length,
            embedding_dim=embedding_dim,
            p_dropout=p_dropout,
        )
        self.decoders = ModuleList(
            [
                Decoder(
                    embedding_dim=embedding_dim,
                    attention_heads=attention_heads,
                    context_length=context_length,
                    ff_hidden_dim=ff_hidden_dim,
                    p_dropout=p_dropout,
                )
                for _ in range(n_decoders)
            ]
        )
        self.linear = Linear(in_features=embedding_dim, out_features=vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        kv_cache=None,
        request_ids: list[str] | None = None,
        start_positions: Tensor | None = None,
        use_cache: bool = False,
    ) -> Tensor:
        # input: B x T
        # B x T x C
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        # B x T x C
        x = self.pos_encoding(x=x, start_positions=start_positions)
        # B x T x C
        for i, decoder in enumerate(self.decoders):
            x = decoder(
                x=x,
                layer_idx=i,
                kv_cache=kv_cache,
                request_ids=request_ids,
                start_positions=start_positions,
                use_cache=use_cache,
            )
        # B x T x V
        logits = self.linear(x)
        return logits
