import torch
from torch import Tensor
from llm_from_scratch.transformer import Transformer, TransformerConfig
from llm_from_scratch.utils import fetch_device
from llm_from_scratch.tokenizer import (
    CARD_START,
    CARD_END,
    CharTokenizer,
    TokenizerConfig,
    TokenizerType,
    BPETokenizer,
)
from pathlib import Path

# --- Params ---

MODEL_PATH = "model/2026-05-14_12-09-58/model.pt"
MAX_TOKENS_TO_GENERATE = 1000
TEMPERATURE = 1.0

# --- End params ---

device = fetch_device()
model_checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
tokenizer_config = TokenizerConfig(**model_checkpoint["tokenizer_config"])

if tokenizer_config.tokenizer_type == TokenizerType.CHAR:
    tokenizer = CharTokenizer(mapping_path=Path(tokenizer_config.mapping_path))
elif tokenizer_config.tokenizer_type == TokenizerType.BPE:
    tokenizer = BPETokenizer(mapping_path=Path(tokenizer_config.mapping_path))
else:
    raise ValueError(f"Invalid tokenizer type: {tokenizer_config.tokenizer_type}")

model_config = TransformerConfig(**model_checkpoint["model_config"])
model = Transformer(
    vocab_size=tokenizer.vocab_size(),
    embedding_dim=model_config.embedding_dim,
    context_length=model_config.context_length,
    ff_hidden_dim=model_config.ff_hidden_dim,
    attention_heads=model_config.attention_heads,
    n_decoders=model_config.n_decoders,
    p_dropout=0,
).to(device)

model.load_state_dict(model_checkpoint["model_state_dict"])

print(f"Successfully loaded {MODEL_PATH}")
print("# Model:")
print(f" - Embedding dim: {model_config.embedding_dim}")
print(f" - Context length: {model_config.context_length}")
print(f" - FF hidden dim: {model_config.ff_hidden_dim}")
print(f" - Attention Heads: {model_config.attention_heads}")
print(f" - N decoders: {model_config.n_decoders}")
print("# Tokenizer:")
print(f" - Vocab_size: {tokenizer.vocab_size()}")
print(f" - Type: {tokenizer.get_type()}")
model.eval()


def sample_next_token(
    model: Transformer, input_tokens: Tensor, temperature: float
) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    with torch.no_grad():
        # 1 x T x V
        logits = model(input_tokens)
        # 1 x V
        logits = logits[:, -1, :]
        logits = logits / temperature
        probs = logits.softmax(dim=-1)
        # 1 x 1
        return torch.multinomial(probs, num_samples=1)


seed_text = input("Enter seed text to complete: ").lower() or f"{CARD_START}\n"
input_tokens = tokenizer.encode(seed_text).to(device)

if len(input_tokens) > model_config.context_length:
    raise ValueError(
        f"{len(input_tokens)} input tokens > {model_config.context_length} model context length"
    )

# 1 x T
generated = input_tokens.unsqueeze(dim=0)

for _ in range(MAX_TOKENS_TO_GENERATE):
    cur_context = generated[:, -model_config.context_length :]
    # 1 x 1
    next_token = sample_next_token(
        model=model, input_tokens=cur_context, temperature=TEMPERATURE
    )
    # 1 x T
    generated = torch.cat((generated, next_token), dim=-1)
    generated_text = tokenizer.decode(generated.squeeze())
    print(generated_text)
    if CARD_END in generated_text:
        break
