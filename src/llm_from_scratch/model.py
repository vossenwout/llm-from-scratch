import torch
from torch.utils.data import DataLoader
from pathlib import Path
from torch.nn import Linear, Module, Embedding, CrossEntropyLoss
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

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        # B x T x C
        x = self.embedding(input_ids)
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
model = Transformer(vocab_size=TOKENIZER.vocab_size(), embedding_dim=EMBEDDING_DIM).to(
    device
)
loss_fn = CrossEntropyLoss()
# params from attention is all you need paper (without learning rate schedule)
optimizer = optim.Adam(params=model.parameters(), betas=(0.9, 0.98), eps=10e-9)

train(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
# test(model, device=device)
