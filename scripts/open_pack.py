import textwrap
import time
from pathlib import Path

import torch
from torch import Tensor

from llm_from_scratch.tokenizer import (
    BPETokenizer,
    CARD_END,
    CARD_START,
    CharTokenizer,
    TokenizerConfig,
    TokenizerType,
)
from llm_from_scratch.transformer import Transformer, TransformerConfig
from llm_from_scratch.utils import fetch_device

# --- Params ---

MODEL_PATH = "model/small-model/best_model.pt"
PACK_SIZE = 5
MAX_TOKENS_TO_GENERATE = 500
TEMPERATURE = 0.9

# --- End params ---

RESET = "\033[0m"
BOLD = "\033[1m"
WHITE = "\033[37m"
BLUE = "\033[34m"
PURPLE = "\033[35m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
FIELD_NAMES = {
    "type",
    "sub_type",
    "attribute",
    "rank",
    "attack",
    "defense",
    "description",
}


def rarity_color(rarity: str) -> str:
    rarity = rarity.lower()
    if "secret" in rarity:
        return CYAN
    if "ultra" in rarity:
        return YELLOW
    if "super" in rarity:
        return PURPLE
    if "rare" in rarity:
        return BLUE
    return WHITE


def extract_field(card_text: str, field: str) -> str:
    for line in card_text.splitlines():
        if line.startswith(f"{field}:"):
            return line.split(":", maxsplit=1)[1].strip()
    return "unknown"


def sample_next_token(
    model: Transformer, input_tokens: Tensor, temperature: float
) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    with torch.no_grad():
        logits = model(input_tokens)
        logits = logits[:, -1, :]
        logits = logits / temperature
        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1)


def generate_card(
    model: Transformer, tokenizer, device: str, context_length: int
) -> str:
    input_tokens = tokenizer.encode(f"{CARD_START}\n").to(device)
    generated = input_tokens.unsqueeze(dim=0)

    for _ in range(MAX_TOKENS_TO_GENERATE):
        cur_context = generated[:, -context_length:]
        next_token = sample_next_token(
            model=model, input_tokens=cur_context, temperature=TEMPERATURE
        )
        generated = torch.cat((generated, next_token), dim=-1)
        generated_text = tokenizer.decode(generated.squeeze())
        if CARD_END in generated_text:
            break

    return generated_text


def print_card(card_text: str, index: int, total: int):
    name = extract_field(card_text, "name")
    rarity = extract_field(card_text, "rarity")
    color = rarity_color(rarity)
    width = 78
    header_left = f"CARD {index}/{total}"
    header_right = rarity.upper()
    body_lines = [
        line
        for line in card_text.splitlines()
        if line not in (CARD_START, CARD_END)
        and not line.startswith("name:")
        and not line.startswith("rarity:")
    ]

    print(f"{color}{BOLD}")
    print("+" + "-" * (width + 2) + "+")
    print(f"| {header_left:<{width - len(header_right)}}{header_right} |")
    print(f"| {name.title()[:width].center(width)} |")
    print("+" + "-" * (width + 2) + "+")
    print(RESET, end="")

    for line in body_lines:
        for wrapped_line in textwrap.wrap(line, width=width) or [""]:
            field = next(
                (
                    field_name
                    for field_name in FIELD_NAMES
                    if wrapped_line.startswith(f"{field_name}:")
                ),
                None,
            )
            if field:
                _, value = wrapped_line.split(":", maxsplit=1)
                visible_line = f"{field}: {value.lstrip()}"
                styled_line = f"{BOLD}{field}:{RESET} {value.lstrip()}"
                print(f"| {styled_line}{' ' * (width - len(visible_line))} |")
            else:
                print(f"| {wrapped_line:<{width}} |")
    print("+" + "-" * (width + 2) + "+\n")


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
model.eval()

print(f"Loaded {MODEL_PATH}")
input("Press Enter to open pack...")
print("Opening pack", end="", flush=True)
for _ in range(3):
    time.sleep(0.4)
    print(".", end="", flush=True)
print("\n")

cards = []
for i in range(PACK_SIZE):
    input(f"Press Enter to reveal card {i + 1}/{PACK_SIZE}...")
    card = generate_card(
        model=model,
        tokenizer=tokenizer,
        device=device,
        context_length=model_config.context_length,
    )
    cards.append(card)
    print_card(card_text=card, index=i + 1, total=PACK_SIZE)

print("Pack Summary")
print("------------")
for i, card in enumerate(cards, start=1):
    name = extract_field(card, "name")
    rarity = extract_field(card, "rarity")
    color = rarity_color(rarity)
    print(f"{i}. {color}{rarity.title():<18}{RESET} {name}")
