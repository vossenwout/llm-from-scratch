import torch
from torch.utils.data import DataLoader
from pathlib import Path
from torch.nn import Linear, Module, Embedding, CrossEntropyLoss
from torch import Tensor, accelerator

from llm_from_scratch.tokenizer import CharTokenizer
from llm_from_scratch.dataset import YuGiOhCardsDataset
from typing import Optional

torch.manual_seed(42)

TRAIN_DATASET = "data/train_set.txt"
TEST_DATASET = "data/test_set.txt"
TOKENIZER = CharTokenizer(mapping_path=Path("data/dataset-tokenizer.json"))
EMBEDDING_DIM = 512  # Attention all you need paper
BATCH_SIZE = 8
CONTEXT_LENGTH = 16  # pretty small for now


def device():
    current_accelerator = accelerator.current_accelerator()
    if current_accelerator and accelerator.is_available():
        return current_accelerator.type
    return "cpu"


def _create_target(y: Tensor, device: str):
    # For if we want to give class probabilities as target
    # TODO: how do we do this in a tensor way instead of looping
    # B x T x V
    target = torch.empty(
        (
            BATCH_SIZE,
            CONTEXT_LENGTH,
            TOKENIZER.vocab_size(),
        ),
        device=device,
    )
    for b in range(BATCH_SIZE):
        for t in range(CONTEXT_LENGTH):
            target[b, t, y[b, t]] = 1

    # B x V x T
    return target.permute(0, 2, 1)


class Decoder(Module):
    def __init__(self):
        super().__init__()


class Transformer(Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.linear = Linear(in_features=embedding_dim, out_features=vocab_size)
        self.loss = CrossEntropyLoss()

    def forward(
        self, input_ids: Tensor, target: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        # B x T x C
        x = self.embedding(input_ids)
        # B x T x V
        logits = self.linear(x)
        # some sort of transform (reshape??)
        # B x V x T
        logits = logits.permute(0, 2, 1)
        if target is not None:
            loss = self.loss(logits, target)
            return logits, loss
        return logits


def decode_pred(pred: Tensor):
    # B x V x T
    if not len(pred.shape) == 3:
        raise Exception("Strange format")
    # B x T
    b_token_preds = pred.argmax(1)
    for i, token_preds in enumerate(b_token_preds):
        print(f"--- Batch {i} ----")
        predicted_string = TOKENIZER.decode(token_preds)
        print(predicted_string)


def decode_input(X: Tensor):
    # B x T
    if not len(X.shape) == 2:
        raise Exception("Strange format")
    # B x T
    for i, token_preds in enumerate(X):
        print(f"--- Batch {i} ----")
        predicted_string = TOKENIZER.decode(token_preds)
        print(predicted_string)


def train(device: str):
    train_dataloader = DataLoader(
        YuGiOhCardsDataset(
            dataset_path=TRAIN_DATASET,
            context_length=CONTEXT_LENGTH,
            tokenizer=TOKENIZER,
        ),
        batch_size=BATCH_SIZE,
    )

    for batch_i, (X, y) in enumerate(train_dataloader):
        # B x T
        X = X.to(device)
        # B x T
        y = y.to(device)
        # B x T x V, B
        pred, loss = transformer(X, y)
        decode_pred(pred)

        if batch_i == 0:
            break


def test(device: str):
    test_dataloader = DataLoader(
        YuGiOhCardsDataset(
            dataset_path=TEST_DATASET,
            context_length=CONTEXT_LENGTH,
            tokenizer=TOKENIZER,
        ),
        batch_size=1,
    )
    transformer = Transformer(
        vocab_size=TOKENIZER.vocab_size(), embedding_dim=EMBEDDING_DIM
    ).to(device)

    for batch_i, (X, y) in enumerate(test_dataloader):
        # B x T
        X = X.to(device)
        decode_input(X)
        # B x T
        y = y.to(device)
        # B x T x V
        pred = transformer(X)
        decode_pred(pred)

        if batch_i == 0:
            break


transformer = Transformer(
    vocab_size=TOKENIZER.vocab_size(), embedding_dim=EMBEDDING_DIM
).to(device)


train(transformer, device=device())
test(transformer, device=device())
