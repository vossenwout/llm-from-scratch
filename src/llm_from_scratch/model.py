import math
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from torch.nn import Linear, Module, Embedding, CrossEntropyLoss, LayerNorm, ReLU
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
FF_HIDDEN_DIM = 2048  # Attention all you need paper
ATTENTION_HEADS = 8  # Attention all you need paper


def fetch_device():
    current_accelerator = accelerator.current_accelerator()
    if current_accelerator and accelerator.is_available():
        return current_accelerator.type
    return "cpu"


class FeedForward(Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, device: str):
        super().__init__()
        self.l1 = Linear(embedding_dim, hidden_dim, device=device)
        self.l2 = Linear(hidden_dim, embedding_dim, device=device)
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        # input B x T x C
        # B x T x H
        x = self.l1(x)
        x = self.relu(x)
        # B x T x C
        return self.l2(x)


class Decoder(Module):
    def __init__(
        self, embedding_dim: int, ff_hidden_dim: int, attention_heads: int, device: str
    ):
        super().__init__()
        self.masked_multi_head_attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            attention_heads=attention_heads,
            device=device,
            masked=True,
        )
        self.layer_norm_1 = LayerNorm(embedding_dim, device=device)
        self.layer_norm_2 = LayerNorm(embedding_dim, device=device)
        self.ff = FeedForward(
            embedding_dim=embedding_dim, hidden_dim=ff_hidden_dim, device=device
        )

    def forward(self, x: Tensor) -> Tensor:
        # B x T x C
        x = self.masked_multi_head_attention(x) + x
        x = self.layer_norm_1(x)
        x = self.ff(x) + x
        return self.layer_norm_2(x)


class Attention(Module):
    """Attention as described by Attention is all you need"""

    def __init__(
        self, embedding_dim: int, attention_heads: int, device: str, masked: bool
    ):
        super().__init__()
        self.d = embedding_dim // attention_heads  # attention all you need paper
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
        q_kt = q_kt / math.sqrt(self.d)
        if self.masked:  # can't attend future tokens
            mask = torch.ones_like(q_kt, dtype=torch.bool).triu(diagonal=1)
            q_kt = q_kt.masked_fill(
                mask, float("-inf")
            )  # -inf instead of 0 as we can have negatives
        q_kt = q_kt.softmax(2)
        # B x T x d
        return q_kt @ v


class MultiHeadAttention(Module):
    def __init__(
        self, embedding_dim: int, attention_heads: int, device: str, masked: bool
    ):
        super().__init__()
        self.heads = [
            Attention(
                embedding_dim=embedding_dim,
                attention_heads=attention_heads,
                device=device,
                masked=masked,
            )
            for _ in range(attention_heads)
        ]
        self.wo = Linear(embedding_dim, embedding_dim, device=device)

    def forward(self, x: Tensor) -> Tensor:
        # input is B x T x C
        # todo can we easily make this fully parallel?
        # each B x T x d
        x_heads = [head(x) for head in self.heads]
        # B x T x C
        x = torch.cat(tensors=x_heads, dim=2)
        # B x T x C
        return self.wo(x)


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
        self,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
        attention_heads: int,
        ff_hidden_dim: int,
        device: str,
    ):
        super().__init__()
        self.embedding = Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.pos_encoding = PosEncoding(
            context_length=context_length, embedding_dim=embedding_dim, device=device
        )
        self.decoder = Decoder(
            embedding_dim=embedding_dim,
            attention_heads=attention_heads,
            ff_hidden_dim=ff_hidden_dim,
            device=device,
        )
        self.linear = Linear(in_features=embedding_dim, out_features=vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        # B x T x C
        x = self.embedding(input_ids)
        # B x T x C
        x = self.pos_encoding(x)
        # B x T x V
        x = self.decoder(x)
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
    ff_hidden_dim=FF_HIDDEN_DIM,
    attention_heads=ATTENTION_HEADS,
    device=device,
).to(device)

loss_fn = CrossEntropyLoss()
# params from attention is all you need paper (without learning rate schedule)
optimizer = optim.Adam(params=model.parameters(), betas=(0.9, 0.98), eps=10e-9)

train(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
# test(model, device=device)
