from pathlib import Path

from llm_from_scratch.tokenizer import CharTokenizer, BPETokenizer


DATASET_PATH = Path("data/processed/yugioh/v001/all.txt")
CHAR_TOKENIZER_PATH = Path("data/processed/yugioh/v001/tokenizer-char.json")
BPE_TOKENIZER_PATH = Path("data/processed/yugioh/v001/tokenizer-bpe.json")

tokenizer = CharTokenizer()
tokenizer.build_mapping(
    input_dataset_path=DATASET_PATH,
    mapping_save_path=CHAR_TOKENIZER_PATH,
)

VOCAB_SIZE = 4096
tokenizer = BPETokenizer()
tokenizer.build_mapping(
    input_dataset_path=DATASET_PATH,
    mapping_save_path=BPE_TOKENIZER_PATH,
    vocab_size=VOCAB_SIZE,
)
