from pathlib import Path

import torch
from torch import optim
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler


def save_checkpoint(
    model: Module,
    optimizer: optim.Optimizer,
    scheduler: LRScheduler,
    run_dir: Path,
    filename: str,
    epoch: int,
    global_step: int,
    last_train_loss: float,
    last_val_loss: float,
    best_val_loss: float,
    model_config: dict,
    data_config: dict,
    training_config: dict,
    tokenizer_config: dict,
):
    save_path = run_dir / filename
    torch.save(
        {
            "model_config": model_config,
            "data_config": data_config,
            "training_config": training_config,
            "tokenizer_config": tokenizer_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "last_train_loss": last_train_loss,
            "last_val_loss": last_val_loss,
            "best_val_loss": best_val_loss,
        },
        save_path,
    )
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: Module,
    optimizer: optim.Optimizer,
    scheduler: LRScheduler,
    device: str,
) -> tuple[int, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return (
        checkpoint["epoch"],
        checkpoint["global_step"],
        checkpoint["best_val_loss"],
    )
