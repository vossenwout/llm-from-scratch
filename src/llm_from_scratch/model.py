import math
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from torch.nn import Linear, Module, Embedding, CrossEntropyLoss, Parameter
from torch import Tensor, accelerator, optim

from llm_from_scratch.tokenizer import CharTokenizer
from llm_from_scratch.dataset import YuGiOhCardsDataset

torch.manual_seed(42)

TRAIN_DATASET = "data/train_set.txt"
TEST_DATASET = "data/test_set.txt"
TOKENIZER = CharTokenizer(mapping_path=Path("data/dataset-tokenizer.json"))
EMBEDDING_DIM = 512  # Attention all you need paper
BATCH_SIZE = 64
CONTEXT_LENGTH = 64  # pretty small for now


def fetch_device():
    current_accelerator = accelerator.current_accelerator()
    if current_accelerator and accelerator.is_available():
        return current_accelerator.type
    return "cpu"


class Decoder(Module):
    def __init__(self):
        super().__init__()


class Attention(Module):
    """Attention as described by Attention is all you need"""

    def __init__(
        self, embedding_dim: int, device: str, h: int = 8, masked: bool = True
    ):
        super().__init__()
        self.d = embedding_dim // h
        self.masked = masked
        self.wq = Linear(embedding_dim, self.d, bias=False).to(device)
        self.wk = Linear(embedding_dim, self.d, bias=False).to(device)
        self.wv = Linear(embedding_dim, self.d, bias=False).to(device)

    def forward(self, x: Tensor) -> Tensor:
        # input is B x T x C
        # each B x T x d
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        # B x T x T
        q_kt = q @ k.transpose(1, 2)
        if self.masked:  # can't attend future tokens
            mask = torch.ones_like(q_kt, dtype=torch.bool).triu(diagonal=1)
            q_kt = q_kt.masked_fill(
                mask, float("-inf")
            )  # -inf instead of 0 as we can have negatives
        q_kt = q_kt / math.sqrt(self.d)
        q_kt = q_kt.softmax(2)
        # B x T x d
        return q_kt @ v


class PosEncoding(Module):
    def __init__(self, context_length: int, embedding_dim: int, device: str):
        super().__init__()
        # TODO: can we do this more efficient without looping?
        # T x C
        self.pos_encoding = torch.zeros(context_length, embedding_dim).to(device)
        for pos in range(context_length):
            for i in range(embedding_dim):
                if i % 2 == 0:
                    self.pos_encoding[pos, i] = math.sin(
                        i / (10000 ** ((2 * i) / embedding_dim))
                    )
                else:
                    self.pos_encoding[pos, i] = math.cos(
                        i / (10000 ** ((2 * i) / embedding_dim))
                    )

    def forward(self, x: Tensor) -> Tensor:
        # B x T x C
        return x + self.pos_encoding


class Transformer(Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int, context_length: int, device: str
    ):
        super().__init__()
        self.embedding = Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.pos_encoding = PosEncoding(
            context_length=context_length, embedding_dim=embedding_dim, device=device
        )
        self.linear = Linear(in_features=embedding_dim, out_features=vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        # B x T x C
        x = self.embedding(input_ids)
        # B x T x C
        x = self.pos_encoding(x)
        # B x T x V
        logits = self.linear(x)
        # some sort of transform (reshape??)
        # B x V x T
        logits = logits.permute(0, 2, 1)
        return logits


def decode_pred(pred: Tensor):
    # B x V x T
    if not len(pred.shape) == 3:
        raise Exception("Strange format")
    # B x T
    b_token_preds = pred.argmax(1)
    for i, token_preds in enumerate(b_token_preds):
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


def train(model: Module, loss_fn: Module, optimizer: optim.Optimizer, device: str):
    train_dataloader = DataLoader(
        YuGiOhCardsDataset(
            dataset_path=TRAIN_DATASET,
            context_length=CONTEXT_LENGTH,
            tokenizer=TOKENIZER,
        ),
        batch_size=BATCH_SIZE,
    )

    model.train()

    for batch, (X, y) in enumerate(train_dataloader):
        # B x T, B x T
        X, y = X.to(device), y.to(device)
        # B x V x T, B
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            print(f"--- Batch: {batch} / {len(train_dataloader)} | Loss: {loss.item()}")
            decode_pred(pred[0:1])


def test(model: Module, device: str):
    test_dataloader = DataLoader(
        YuGiOhCardsDataset(
            dataset_path=TEST_DATASET,
            context_length=CONTEXT_LENGTH,
            tokenizer=TOKENIZER,
        ),
        batch_size=1,
    )

    model.eval()
    with torch.no_grad():
        for batch_i, (X, y) in enumerate(test_dataloader):
            # B x T, B x T
            X, y = X.to(device), y.to(device)
            decode_input(X)
            # B x T x V
            pred = model(X)
            decode_pred(pred)

            if batch_i == 0:
                break


# Setup
device = fetch_device()
model = Transformer(
    vocab_size=TOKENIZER.vocab_size(),
    embedding_dim=EMBEDDING_DIM,
    context_length=CONTEXT_LENGTH,
    device=device,
).to(device)

loss_fn = CrossEntropyLoss()
# params from attention is all you need paper (without learning rate schedule)
optimizer = optim.Adam(params=model.parameters(), betas=(0.9, 0.98), eps=10e-9)

train(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
# test(model, device=device)
