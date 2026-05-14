from tokenizers.tokenizers import models, pre_tokenizers, trainers, processors, decoders
from tokenizers import Tokenizer
from llm_from_scratch.tokenizer import (
    CARD_START,
    CARD_END,
)

# Params
DATASET_PATH = "data/processed/yugioh/v001/all.txt"
VOCAB_SIZE = 4096  # Original transformer paper mentions 37k

tokenizer = Tokenizer(models.BPE())
# 1. Split text into chunks using regex
# 2. Convert chunks into individual UTF-8 bytes
# 3. Map each byte to visible Unicode symbol (so vocab size is at least 256 tokens)
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=[CARD_START, CARD_END],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
tokenizer.train([DATASET_PATH], trainer=trainer)

test_input = """
<card>
name: evil mind
type: spell
description: if you control a fiend monster: activate 1 of these effects, based on the number of monsters in your opponent's gy;●1+: draw 1 card.●4+: add 1 "hero" monster or 1 "dark fusion" from your deck to your hand.●10+: add 1 "polymerization" spell or 1 "fusion" spell from your deck to your hand.you can only activate 1 "evil mind" per turn.
</card>
"""

print("input")
print(f"Chars of input {len(test_input)}")
print(test_input)

encoding = tokenizer.encode(test_input)
print(encoding.tokens)

tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

tokenizer.decoder = decoders.ByteLevel()

output = tokenizer.decode(encoding.ids, skip_special_tokens=False)
print("decoded")
print(f"Tokens of input {len(encoding.tokens)}")
print(output)
