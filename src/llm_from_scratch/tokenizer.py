from dataclasses import dataclass
from enum import Enum
from abc import abstractmethod, ABC
from pathlib import Path
from torch import tensor, Tensor
from typing import Optional
import json


class TokenizerType(Enum):
    CHAR = "char"
    BPE = "bpe"


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, s: str) -> Tensor:
        pass

    @abstractmethod
    def decode(self, ix: Tensor) -> str:
        pass

    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def get_type(self) -> TokenizerType:
        pass


@dataclass
class TokenizerConfig:
    mapping_path: str
    tokenizer_type: TokenizerType


class CharTokenizer(Tokenizer):
    def __init__(self, mapping_path: Optional[Path] = None):
        self.mapping_path = mapping_path
        if mapping_path:
            with open(mapping_path, "r") as f:
                self.c_to_i = json.load(f)
                self.i_to_c = {}
                for c, i in self.c_to_i.items():
                    self.i_to_c[i] = c

    def build_mapping(self, input_dataset_path: Path, mapping_save_path: Path):
        with open(input_dataset_path, "r") as f:
            dataset_str = f.read()

        c_to_i = {c: i for i, c in enumerate(set(dataset_str))}
        with open(mapping_save_path, "w") as f:
            json.dump(c_to_i, f, ensure_ascii=False)
            print(f"Total tokenizer keys {len(c_to_i)}")
            print(f"Saved tokenizer mapping to {mapping_save_path}")

    def vocab_size(self) -> int:
        if not self.c_to_i:
            raise Exception("Forgot to load tokenizer mapping")
        return len(self.c_to_i)

    def encode(self, s: str) -> Tensor:
        if not self.c_to_i:
            raise Exception("Forgot to load tokenizer mapping")
        return tensor([self.c_to_i[c] for c in s])

    def decode(self, ix: Tensor) -> str:
        if not self.i_to_c:
            raise Exception("Forgot to load tokenizer mapping")
        return "".join(self.i_to_c[i.item()] for i in ix)

    def get_type(self) -> TokenizerType:
        return TokenizerType.CHAR


def build_tokenizer(input_dataset_path: Path):
    print(input_dataset_path)
    tokenizer = CharTokenizer()
    tokenizer.build_mapping(
        input_dataset_path=input_dataset_path,
        mapping_save_path=input_dataset_path.parent
        / (str(input_dataset_path.stem) + "-tokenizer.json"),
    )


# Todo move this to a pytest
def sanity_check(test_input: str, tokenizer_mapping_path: str):
    char_tokenizer = CharTokenizer(mapping_path=Path(tokenizer_mapping_path))
    encoded = char_tokenizer.encode(test_input)
    print(f"encoded {encoded}")
    decoded = char_tokenizer.decode(encoded)
    print(f"decoded {decoded}")


text_sample = """
name: a case for k9
description: when this card is activated: you can add 1 "k9" monster from your deck to your hand. "k9" monsters you control gain 900 atk during any turn in which your opponent has activated a monster effect in the hand or gy. if this card in the spell & trap zone is destroyed by card effect: you can set 1 "k9" quick-play spell from your deck or gy. you can only use this effect of ""a case for k9"" once per turn. you can only activate 1 ""a case for k9"" per turn.

name: a cell breeding device
description: during each of your standby phases, put 1 a-counter on 1 face-up monster your opponent controls.

name: a cell incubator
description: each time an a-counter(s) is removed from play by a card effect, place 1 a-counter on this card. when this card is destroyed, distribute the a-counters on this card among face-up monsters.

name: a cell recombination device
description: target 1 face-up monster on the field; send 1 "alien" monster from your deck to the graveyard, and if you do, place a-counters on that monster equal to the level of the sent monster. during your main phase, except the turn this card was sent to the graveyard: you can banish this card from your graveyard; add 1 "alien" monster from your deck to your hand.
"""
# sanity_check(text_sample, "data/dataset-tokenizer.json")

# build_tokenizer(Path("./data/dataset.txt"))
