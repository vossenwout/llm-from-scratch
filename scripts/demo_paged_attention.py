import math
import torch
from llm_from_scratch.paged_kv_cache import PagedKVCache, PagedKVCacheConfig

request_id = "req1"
prompt_len = 6
block_size = 4
n_layers = 3
n_heads = 2
head_dim = 8

cache = PagedKVCache(
    PagedKVCacheConfig(
        num_blocks=8,
        block_size=block_size,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        device="cpu",
        dtype=torch.float32,
    )
)
cache.create_request(request_id)

# Prefill: every layer writes K/V for the same logical prompt positions.
for layer_idx in range(n_layers):
    keys = torch.randn(n_heads, prompt_len, head_dim)
    values = torch.randn(n_heads, prompt_len, head_dim)
    cache.append(
        request_id=request_id,
        layer_idx=layer_idx,
        key=keys,
        value=values,
        start_pos=0,
    )

table = cache.get_block_table(request_id)
print(f"after prefill: {table.token_count} cached tokens")
for logical_block, physical_block in enumerate(table.block_ids):
    start = logical_block * block_size
    end = min(start + block_size, table.token_count)
    print(f"logical tokens {start}-{end - 1} -> physical block {physical_block}")

# Decode: every layer writes K/V for the same new token at position 6.
decode_pos = table.token_count
for layer_idx in range(n_layers):
    key = torch.randn(n_heads, 1, head_dim)
    value = torch.randn(n_heads, 1, head_dim)
    cache.append(
        request_id=request_id,
        layer_idx=layer_idx,
        key=key,
        value=value,
        start_pos=decode_pos,
    )

print(f"after decode: {table.token_count} cached tokens")
print(f"physical blocks in use: {cache.num_used_blocks()}")

# Toy paged attention gathers one layer's scattered blocks first.
layer_idx = 0
gathered_keys, gathered_values = cache.get(request_id, layer_idx)
query = torch.randn(n_heads, 1, head_dim)

scores = query @ gathered_keys.transpose(-2, -1)
weights = torch.softmax(scores / math.sqrt(head_dim), dim=-1)
attention_output = weights @ gathered_values

print(f"gathered K shape: {tuple(gathered_keys.shape)}")
print(f"attention output shape: {tuple(attention_output.shape)}")

cache.free_request(request_id)
print(f"free blocks after free_request: {cache.num_free_blocks()}")
