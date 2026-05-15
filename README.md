# LLM From Scratch

Small decoder-only Transformer implemented in PyTorch, trained from scratch to generate Yu-Gi-Oh cards. The decoder architecture is based on [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## Folders

- `src/llm_from_scratch/`: model, tokenizer, dataset, and checkpoint code.
- `scripts/`: runnable scripts for data prep, training, and inference.
- `data/`: raw and processed datasets.
- `model/`: saved training runs and checkpoints.

## Setup

```bash
uv sync
```

Optional W&B logging:

```bash
uv run wandb login
```

Set `use_wandb=True`, `wandb_entity`, and `wandb_project` in `scripts/train.py` to enable logging. Training and model settings live in `TRAIN_CONFIG` and `MODEL_CONFIG`.

## Scripts

Run in order:

```bash
uv run scripts/download_data.py
uv run scripts/create_dataset.py
uv run scripts/build_tokenizer.py
uv run scripts/train.py
uv run scripts/inference.py
uv run scripts/open_pack.py
```

- `download_data.py`: downloads the raw Yu-Gi-Oh card dataset.
- `create_dataset.py`: converts raw card rows into train/validation text files.
- `build_tokenizer.py`: builds the char and BPE tokenizers.
- `train.py`: trains the Transformer and writes checkpoints to `model/`.
- `inference.py`: loads a checkpoint and generates card text.
- `open_pack.py`: opens a booster pack of generated cards from your cli :).

Use `checkpoint.pt` to resume training and `best_model.pt` for inference.
