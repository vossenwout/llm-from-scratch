from torch.utils.data import DataLoader
from pathlib import Path
from torch.nn import Linear, Module, Embedding
from torch import Tensor, accelerator

from llm_from_scratch.tokenizer import CharTokenizer
from llm_from_scratch.dataset import YuGiOhCardsDataset

TRAIN_DATASET = "data/train_set.txt"
TOKENIZER = CharTokenizer(mapping_path=Path("data/dataset-tokenizer.json"))
EMBEDDING_DIM = 512  # Attention all you need paper
BATCH_SIZE = 8
CONTEXT_LENGTH = 16  # pretty small for now


def device():
    current_accelerator = accelerator.current_accelerator()
    if current_accelerator and accelerator.is_available():
        return current_accelerator.type
    return "cpu"


class Decoder(Module):
    def __init__(self):
        super().__init__()


class Transformer(Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)


def train(device: str):
    train_dataloader = DataLoader(
        YuGiOhCardsDataset(
            dataset_path=TRAIN_DATASET,
            context_length=CONTEXT_LENGTH,
            tokenizer=TOKENIZER,
        ),
        batch_size=BATCH_SIZE,
    )
    transformer = Transformer(
        vocab_size=TOKENIZER.vocab_size(), embedding_dim=EMBEDDING_DIM
    ).to(device)

    for batch_i, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        # B x S x C
        pred = transformer(X)
        print(pred.shape)
        if batch_i == 0:
            break


train(device=device())
