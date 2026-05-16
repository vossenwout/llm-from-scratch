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


### Sample card pack 

```bash
Press Enter to reveal card 1/5...

+--------------------------------------------------------------------------------+
| CARD 1/5                                                                COMMON |
|                                Oracle Of Light                                 |
+--------------------------------------------------------------------------------+
| type: monster                                                                  |
| sub_type: [fairy／effect]                                                       |
| attribute: light                                                               |
| rank: level 4                                                                  |
| attack: 0                                                                      |
| defense: 2000                                                                  |
| description: if your opponent controls a monster, you can special summon this  |
| card (from your hand). if this card is normal or special summoned from the     |
| hand: you can draw 1 card. during your main phase: you can activate this       |
| effect; you can add 1 level 4 or lower light fairy monster from your gy to     |
| your hand, except "contorteration". you can only use each effect of "tis the   |
| dragon ninja" once per turn.                                                   |
+--------------------------------------------------------------------------------+

Press Enter to reveal card 2/5...

+--------------------------------------------------------------------------------+
| CARD 2/5                                                                COMMON |
|                          Neo Blue Carrier Fuborawler                           |
+--------------------------------------------------------------------------------+
| type: monster                                                                  |
| sub_type: [insect／effect]                                                      |
| attribute: light                                                               |
| rank: level 5                                                                  |
| attack: 1700                                                                   |
| defense: 1200                                                                  |
| description: you can special summon this card (from your hand) by tributing 1  |
| insect or plant monster, except "insect armor ninja getsuga". if this card is  |
| special summoned from the gy: you can special summon 1 level 4 or lower insect |
| monster from your hand. you can only special summon "assaraiza the hidden      |
| star" once per turn this way. if this card in its owner's control is destroyed |
| by an opponent's card (by battle or card effect): you can target 1 level 4 or  |
| lower insect monster in your gy; add it to your hand. you can only use this    |
| effect of "nimble darklord" once per turn.                                     |
+--------------------------------------------------------------------------------+

Press Enter to reveal card 3/5...

+--------------------------------------------------------------------------------+
| CARD 3/5                                                            ULTRA RARE |
|                           Wishes Of The White Forest                           |
+--------------------------------------------------------------------------------+
| type: monster                                                                  |
| sub_type: [illusion／effect]                                                    |
| attribute: light                                                               |
| rank: level 3                                                                  |
| attack: 300                                                                    |
| defense: 1200                                                                  |
| description: (quick effect): you can tribute this card; special summon 1       |
| "white forest" monster from your deck, by tributing monsters from your hand or |
| field whose total levels equal 8 or more. you can only use each effect of      |
| "white steuder" once per turn. each time a spell card is activated, place 1    |
| spell counter on this card when that spell resolves.                           |
+--------------------------------------------------------------------------------+

Press Enter to reveal card 4/5...

+--------------------------------------------------------------------------------+
| CARD 4/5                                                                COMMON |
|                                   Tree Shark                                   |
+--------------------------------------------------------------------------------+
| type: monster                                                                  |
| sub_type: [fish／normal]                                                        |
| attribute: water                                                               |
| rank: level 9                                                                  |
| attack: 900                                                                    |
| defense: 2900                                                                  |
| description: this fru aries of en script enemies and went in nothing. this     |
| bird is extingused, but also still a spell/trap card.)                         |
+--------------------------------------------------------------------------------+

Press Enter to reveal card 5/5...

+--------------------------------------------------------------------------------+
| CARD 5/5                                                            SUPER RARE |
|                                 Bujingi Cugin                                  |
+--------------------------------------------------------------------------------+
| type: monster                                                                  |
| sub_type: [winged beast／effect]                                                |
| attribute: light                                                               |
| rank: level 2                                                                  |
| attack: 800                                                                    |
| defense: 400                                                                   |
| description: if you control a "bujin" monster, you can special summon this     |
| card (from your hand). you can only special summon "bujingi centipede" once    |
| per turn this way. if this face-up card on the field would be destroyed, you   |
| can destroy another face-up "gusto" monster card you control instead. you can  |
| only use each of the following effects of "bujingi sinyoug" once per turn. if  |
| this card is sent from the field to the graveyard: you can special summon 1    |
| "green gadget" from your deck.                                                 |
+--------------------------------------------------------------------------------+

Pack Summary
------------
1. Common             oracle of light
2. Common             neo blue carrier fuborawler
3. Ultra Rare         wishes of the white forest
4. Common             tree shark
5. Super Rare         bujingi cugin
```
