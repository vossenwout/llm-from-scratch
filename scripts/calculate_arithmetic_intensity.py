# My Model: 8 layers, d_model=1024, 8 heads, d_ff=4096, context=4096
# Tokenizer: BPE, vocabulary size 4096
# Device: Apple MPS
# Dtype: float32

B = 100
# my model
L = 8
C = 1024
FF_C = 4096
T = 4096
V = 4096

# Prefill / decode without KV cache

I_prefill = (B * T * (1 + (8 * L * C) + (4 * L * T) + (4 * L * FF_C) + (2 * V))) / (
    4 * (T * (B + 1) + (4 * L * C) + (2 * L * FF_C) + V)
)
print(f"Arithmetic intensity prefill: {I_prefill}")
# Decode with a KV cache

I_decode = (B * (1 + (8 * L * C) + (4 * L * T) + (4 * L * FF_C) + (2 * V))) / (
    4 * (B + 1 + (4 * L * C) + (2 * L * FF_C) + V + (2 * L * B * T))
)

print(f"Arithmetic intensity decode: {I_decode}")

kv_cache_size = (4 * (2 * B * L * T * C)) / 1e9

print(f"kv cache size at max_content_len = t gb {kv_cache_size}")
