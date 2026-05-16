from torch.optim.lr_scheduler import LRScheduler
from dataclasses import dataclass, asdict
import math
import time
import torch
import wandb
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from typing import Any
from torch.nn import (
    Module,
    CrossEntropyLoss,
)
from torch import optim
from llm_from_scratch.tokenizer import (
    CharTokenizer,
    BPETokenizer,
    Tokenizer,
    TokenizerConfig,
    TokenizerType,
)
from llm_from_scratch.dataset import YuGiOhCardsDataset
from llm_from_scratch.transformer import Transformer, TransformerConfig
from llm_from_scratch.utils import fetch_device
from llm_from_scratch.checkpoint import load_checkpoint, save_checkpoint

torch.manual_seed(42)


@dataclass
class DataConfig:
    train_dataset: str
    val_dataset: str


@dataclass
class TrainConfig:
    train_epochs: int
    batch_size: int
    warmup_steps: int
    val_batches: int
    log_every_steps: int
    checkpoint_every_steps: int
    resume_checkpoint_path: str | None
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    use_wandb: bool
    wandb_entity: str | None
    wandb_project: str | None


# --- Define train params ---
# TOKENIZER_CONFIG = TokenizerConfig(
#    mapping_path="data/processed/yugioh/v001/tokenizer-char.json",
#    tokenizer_type=TokenizerType.CHAR,
# )

TOKENIZER_CONFIG = TokenizerConfig(
    mapping_path="data/processed/yugioh/v001/tokenizer-bpe.json",
    tokenizer_type=TokenizerType.BPE,
)

DATA_CONFIG = DataConfig(
    train_dataset="data/processed/yugioh/v001/train.txt",
    val_dataset="data/processed/yugioh/v001/val.txt",
)


TRAIN_CONFIG = TrainConfig(
    train_epochs=1,
    batch_size=32,
    val_batches=20,
    log_every_steps=100,
    checkpoint_every_steps=1000,
    resume_checkpoint_path=None,
    adam_beta1=0.9,  # Attention is all you need paper
    adam_beta2=0.98,  # Attention is all you need paper
    adam_eps=1e-9,  # Attention is all you need paper
    warmup_steps=2000,  # Attention is all you need paper
    use_wandb=True,
    wandb_entity="pookie",
    wandb_project="yugioh-card-generator",
)

MODEL_CONFIG = TransformerConfig(
    embedding_dim=512,  # Attention is all you need paper,
    context_length=128,
    attention_heads=8,  # Attention is all you need paper
    ff_hidden_dim=2048,  # Attention is all you need paper
    n_decoders=6,  # Attention is all you need paper
    p_dropout=0.1,  # Attention is all you need paper
)

# --- End train params ---


def estimate_val_loss(
    model: Module,
    loss_fn: Module,
    val_dataloader: DataLoader,
    val_batches: int,
    device: str,
) -> tuple[float, float]:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(val_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            losses.append(loss_fn(pred.permute(0, 2, 1), y).item())

            if batch == val_batches - 1:
                break
    loss = sum(losses) / len(losses)
    model.train()
    return loss, math.exp(loss)


def train(
    model: Module,
    loss_fn: Module,
    optimizer: optim.Optimizer,
    scheduler: LRScheduler,
    train_dataloader: DataLoader,
    train_epochs: int,
    val_dataloader: DataLoader,
    val_batches: int,
    device: str,
    tokenizer: Tokenizer,
    run_dir: Path,
    log_every_steps: int,
    checkpoint_every_steps: int,
    start_epoch: int,
    global_step: int,
    best_val_loss: float,
    wandb_run: Any | None,
):
    model.train()
    last_train_loss = float("nan")
    end_epoch = start_epoch + train_epochs
    for epoch in range(start_epoch, end_epoch):
        print(f"[epoch] start epoch={epoch}")
        epoch_start = time.perf_counter()
        log_start = epoch_start
        log_tokens = 0
        log_batches = 0
        for batch, (X, y) in enumerate(train_dataloader):
            # B x T, B x T
            X, y = X.to(device), y.to(device)
            # B x T x V
            pred = model(X)
            loss = loss_fn(pred.permute(0, 2, 1), y)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            last_train_loss = loss.item()
            log_tokens += X.numel()
            log_batches += 1

            if global_step % log_every_steps == 0:
                log_time = time.perf_counter() - log_start
                tokens_per_second = log_tokens / log_time
                tokens_per_second_str = (
                    f"{tokens_per_second / 1000:.1f}k"
                    if tokens_per_second >= 1000
                    else f"{tokens_per_second:.0f}"
                )
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[train] epoch={epoch} batch={batch}/{len(train_dataloader)} "
                    f"step={global_step} loss={loss.item():.4f} "
                    f"lr={current_lr:.2e} batch_time={log_time / log_batches:.3f}s "
                    f"tok/s={tokens_per_second_str}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": loss.item(),
                            "train/lr": current_lr,
                            "train/tokens_per_second": tokens_per_second,
                            "train/batch_time": log_time / log_batches,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
                log_start = time.perf_counter()
                log_tokens = 0
                log_batches = 0

            if global_step % checkpoint_every_steps == 0:
                val_loss, perplexity = estimate_val_loss(
                    model=model,
                    loss_fn=loss_fn,
                    val_dataloader=val_dataloader,
                    val_batches=val_batches,
                    device=device,
                )
                last_val_loss = val_loss
                print(
                    f"[eval] step={global_step} val_loss={val_loss:.4f} "
                    f"ppl={perplexity:.2f}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "val/loss": val_loss,
                            "val/perplexity": perplexity,
                            "val/best_loss": min(best_val_loss, val_loss),
                        },
                        step=global_step,
                    )
                checkpoint_filename = "checkpoint.pt"
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_filename = "best_model.pt"

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    run_dir=run_dir,
                    filename=checkpoint_filename,
                    epoch=epoch,
                    global_step=global_step,
                    last_train_loss=last_train_loss,
                    last_val_loss=last_val_loss,
                    best_val_loss=best_val_loss,
                    model_config=asdict(MODEL_CONFIG),
                    data_config=asdict(DATA_CONFIG),
                    training_config=asdict(TRAIN_CONFIG),
                    tokenizer_config=asdict(TOKENIZER_CONFIG),
                )
                target_text = tokenizer.decode(y[0]).replace("\n", "\\n")[:300]
                predicted_text = tokenizer.decode(pred[0].argmax(dim=-1)).replace(
                    "\n", "\\n"
                )[:300]
                print("[sample]")
                print(f"target:    {target_text}...")
                print(f"predicted: {predicted_text}...")
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "sample/target": target_text,
                            "sample/predicted": predicted_text,
                        },
                        step=global_step,
                    )

        epoch_time = time.perf_counter() - epoch_start
        print(f"[epoch] done epoch={epoch} time={epoch_time:.2f}s")

    last_val_loss, perplexity = estimate_val_loss(
        model=model,
        loss_fn=loss_fn,
        val_dataloader=val_dataloader,
        val_batches=val_batches,
        device=device,
    )
    print(
        f"[final] step={global_step} val_loss={last_val_loss:.4f} ppl={perplexity:.2f}"
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "val/loss": last_val_loss,
                "val/perplexity": perplexity,
                "val/best_loss": min(best_val_loss, last_val_loss),
            },
            step=global_step,
        )
    checkpoint_filenames = ["checkpoint.pt"]
    if last_val_loss < best_val_loss:
        best_val_loss = last_val_loss
        checkpoint_filenames.append("best_model.pt")

    for checkpoint_filename in checkpoint_filenames:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            run_dir=run_dir,
            filename=checkpoint_filename,
            epoch=end_epoch,
            global_step=global_step,
            last_train_loss=last_train_loss,
            last_val_loss=last_val_loss,
            best_val_loss=best_val_loss,
            model_config=asdict(MODEL_CONFIG),
            data_config=asdict(DATA_CONFIG),
            training_config=asdict(TRAIN_CONFIG),
            tokenizer_config=asdict(TOKENIZER_CONFIG),
        )


def lr_schedule(step: int) -> float:
    """Attention is all you need paper"""
    step = max(step, 1)  # prevent step 0
    return (MODEL_CONFIG.embedding_dim**-0.5) * min(
        step**-0.5, step * TRAIN_CONFIG.warmup_steps**-1.5
    )


def print_run_config(model: Module, tokenizer: Tokenizer, device: str, run_dir: Path):
    n_params = sum(p.numel() for p in model.parameters())

    print("# Run")
    print(f" - Device: {device}")
    print(f" - Run dir: {run_dir}")
    print(f" - Resume checkpoint: {TRAIN_CONFIG.resume_checkpoint_path}")
    print("# Data")
    print(f" - Train dataset: {DATA_CONFIG.train_dataset}")
    print(f" - Val dataset: {DATA_CONFIG.val_dataset}")
    print("# Tokenizer")
    print(f" - Type: {TOKENIZER_CONFIG.tokenizer_type}")
    print(f" - Mapping path: {TOKENIZER_CONFIG.mapping_path}")
    print(f" - Vocab size: {tokenizer.vocab_size()}")
    print("# Model")
    print(f" - Embedding dim: {MODEL_CONFIG.embedding_dim}")
    print(f" - Context length: {MODEL_CONFIG.context_length}")
    print(f" - Attention heads: {MODEL_CONFIG.attention_heads}")
    print(f" - FF hidden dim: {MODEL_CONFIG.ff_hidden_dim}")
    print(f" - Decoders: {MODEL_CONFIG.n_decoders}")
    print(f" - Dropout: {MODEL_CONFIG.p_dropout}")
    print(f" - Parameters: {n_params:,}")
    print("# Training")
    print(f" - Epochs: {TRAIN_CONFIG.train_epochs}")
    print(f" - Batch size: {TRAIN_CONFIG.batch_size}")
    print(f" - Warmup steps: {TRAIN_CONFIG.warmup_steps}")
    print(f" - Val batches: {TRAIN_CONFIG.val_batches}")
    print(f" - Log every steps: {TRAIN_CONFIG.log_every_steps}")
    print(f" - Checkpoint every steps: {TRAIN_CONFIG.checkpoint_every_steps}")
    print(f" - W&B: {TRAIN_CONFIG.use_wandb}")
    print(f" - W&B entity: {TRAIN_CONFIG.wandb_entity}")
    print(f" - W&B project: {TRAIN_CONFIG.wandb_project}")


def init_wandb(run_dir: Path):
    if not TRAIN_CONFIG.use_wandb:
        return None

    if TRAIN_CONFIG.wandb_entity is None or TRAIN_CONFIG.wandb_project is None:
        raise ValueError("Set wandb_entity and wandb_project when use_wandb=True")

    return wandb.init(
        entity=TRAIN_CONFIG.wandb_entity,
        project=TRAIN_CONFIG.wandb_project,
        name=run_dir.name,
        config={
            "model": asdict(MODEL_CONFIG),
            "training": asdict(TRAIN_CONFIG),
            "data": asdict(DATA_CONFIG),
            "tokenizer": {
                "mapping_path": TOKENIZER_CONFIG.mapping_path,
                "tokenizer_type": TOKENIZER_CONFIG.tokenizer_type.value,
            },
        },
    )


device = fetch_device()

if TOKENIZER_CONFIG.tokenizer_type == TokenizerType.CHAR:
    tokenizer = CharTokenizer(mapping_path=Path(TOKENIZER_CONFIG.mapping_path))
elif TOKENIZER_CONFIG.tokenizer_type == TokenizerType.BPE:
    tokenizer = BPETokenizer(mapping_path=Path(TOKENIZER_CONFIG.mapping_path))
else:
    raise ValueError(f"Invalid tokenizer type: {TOKENIZER_CONFIG.tokenizer_type}")

model = Transformer(
    vocab_size=tokenizer.vocab_size(),
    embedding_dim=MODEL_CONFIG.embedding_dim,
    context_length=MODEL_CONFIG.context_length,
    ff_hidden_dim=MODEL_CONFIG.ff_hidden_dim,
    attention_heads=MODEL_CONFIG.attention_heads,
    n_decoders=MODEL_CONFIG.n_decoders,
    p_dropout=MODEL_CONFIG.p_dropout,
).to(device)

train_dataloader = DataLoader(
    YuGiOhCardsDataset(
        dataset_path=DATA_CONFIG.train_dataset,
        context_length=MODEL_CONFIG.context_length,
        tokenizer=tokenizer,
    ),
    batch_size=TRAIN_CONFIG.batch_size,
    shuffle=True,
)

val_dataloader = DataLoader(
    YuGiOhCardsDataset(
        dataset_path=DATA_CONFIG.val_dataset,
        context_length=MODEL_CONFIG.context_length,
        tokenizer=tokenizer,
    ),
    batch_size=TRAIN_CONFIG.batch_size,
    shuffle=True,
)

loss_fn = CrossEntropyLoss()
optimizer = optim.Adam(
    params=model.parameters(),
    lr=1.0,  # we rely on scheduler
    betas=(TRAIN_CONFIG.adam_beta1, TRAIN_CONFIG.adam_beta2),
    eps=TRAIN_CONFIG.adam_eps,
)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

start_epoch = 0
global_step = 0
best_val_loss = float("inf")

if TRAIN_CONFIG.resume_checkpoint_path is None:
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("model") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
else:
    run_dir = Path(TRAIN_CONFIG.resume_checkpoint_path).parent
    start_epoch, global_step, best_val_loss = load_checkpoint(
        checkpoint_path=TRAIN_CONFIG.resume_checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
    print(f"Loaded checkpoint from {TRAIN_CONFIG.resume_checkpoint_path}")

print_run_config(model=model, tokenizer=tokenizer, device=device, run_dir=run_dir)
wandb_run = init_wandb(run_dir=run_dir)

train(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    train_dataloader=train_dataloader,
    val_batches=TRAIN_CONFIG.val_batches,
    train_epochs=TRAIN_CONFIG.train_epochs,
    val_dataloader=val_dataloader,
    device=device,
    tokenizer=tokenizer,
    run_dir=run_dir,
    log_every_steps=TRAIN_CONFIG.log_every_steps,
    checkpoint_every_steps=TRAIN_CONFIG.checkpoint_every_steps,
    start_epoch=start_epoch,
    global_step=global_step,
    best_val_loss=best_val_loss,
    wandb_run=wandb_run,
)

if wandb_run is not None:
    wandb_run.finish()
