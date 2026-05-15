import torch

from llm_from_scratch.transformer import MultiHeadAttention, MultiHeadAttentionSeq


def test_vectorized_multi_head_attention_matches_sequential():
    torch.manual_seed(0)

    batch_size = 2
    context_length = 8
    sequence_length = 4
    embedding_dim = 8
    attention_heads = 2

    seq = MultiHeadAttentionSeq(
        embedding_dim=embedding_dim,
        attention_heads=attention_heads,
        context_length=context_length,
        masked=True,
    )
    vec = MultiHeadAttention(
        embedding_dim=embedding_dim,
        attention_heads=attention_heads,
        context_length=context_length,
        masked=True,
    )

    with torch.no_grad():
        vec.wq.weight.copy_(torch.cat([head.wq.weight for head in seq.heads], dim=0))
        vec.wk.weight.copy_(torch.cat([head.wk.weight for head in seq.heads], dim=0))
        vec.wv.weight.copy_(torch.cat([head.wv.weight for head in seq.heads], dim=0))
        vec.wo.weight.copy_(seq.wo.weight)
        vec.wo.bias.copy_(seq.wo.bias)

    x = torch.randn(batch_size, sequence_length, embedding_dim)

    torch.testing.assert_close(vec(x), seq(x))
