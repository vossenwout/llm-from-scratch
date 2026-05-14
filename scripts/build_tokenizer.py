from pathlib import Path

from llm_from_scratch.tokenizer import CharTokenizer


DATASET_PATH = Path("data/processed/yugioh/v001/all.txt")
TOKENIZER_PATH = Path("data/processed/yugioh/v001/tokenizer-char.json")


tokenizer = CharTokenizer()
tokenizer.build_mapping(
    input_dataset_path=DATASET_PATH,
    mapping_save_path=TOKENIZER_PATH,
)
