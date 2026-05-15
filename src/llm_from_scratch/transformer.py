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

    def forward(self, x: Tensor) -> Tensor:
        # B x T x C
        x = self.dropout(self.masked_multi_head_attention(x)) + x
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

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        # input is B x T x C
        B, T, C = (
            x.shape[0],
            x.shape[1],
            x.shape[2],
        )
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
        # B x h x T x T
        q_kt = q @ k.transpose(-2, -1)
        q_kt = q_kt / math.sqrt(self.head_dim)
        if self.masked:  # can't attent future tokens
            q_kt = q_kt.masked_fill(
                self.mask[:T, :T], float("-inf")
            )  # -inf instead of 0 as we can have negatives
        q_kt = q_kt.softmax(dim=-1)

        # B x h x T x d
        v_q_kt = q_kt @ v
        # B x T x h x d
        v_q_kt = v_q_kt.transpose(1, 2)
        # B x T x C
        v_q_kt = v_q_kt.reshape(B, T, C)
        # B x T x C
        return self.wo(v_q_kt)


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

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] <= self.pos_encoding.shape[0], (
            "Passed more tokens than model context length"
        )
        # B x T x C
        # slicing because during inference we don't have all T yet.
        return self.dropout(x + self.pos_encoding[: x.shape[1], :])


class Transformer(Module):
    """Decoder-only Transformed as presented in Attention is all you need paper"""

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
        self.linear = Linear(in_features=embedding_dim, out_features=vocab_size)
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

    def forward(self, input_ids: Tensor) -> Tensor:
        # B x T x C
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        # B x T x C
        x = self.pos_encoding(x)
        # B x T x C
        for decoder in self.decoders:
            x = decoder(x)
        # B x T x V
        logits = self.linear(x)
        return logits
