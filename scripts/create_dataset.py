import pandas as pd
import json
from pathlib import Path

from pandas.core.series import Series
from sklearn.model_selection import train_test_split
from llm_from_scratch.tokenizer import CARD_START, CARD_END

DATASET = "hammadus/yugioh-full-card-database-index-august-1st-2025"
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "yugioh" / "v001"
VAL_SIZE = 0.1

dataset_paths = sorted(RAW_DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
dataset_path = dataset_paths[0]

dataset = pd.read_csv(dataset_path)
dataset = dataset.drop_duplicates(subset="name")
train_df, val_df = train_test_split(
    dataset,
    random_state=42,
    test_size=VAL_SIZE,
)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def serialize_card(card_row: Series):
    card_name = card_row["name"].replace('"', "")
    card_description = card_row["description"]
    card_fields = [
        ("name", card_name),
        ("type", card_row["type"]),
        ("sub_type", card_row["sub_type"]),
        ("attribute", card_row["attribute"]),
        ("rank", card_row["rank"]),
        ("attack", card_row["attack"]),
        ("defense", card_row["defense"]),
        ("description", card_description),
    ]

    formatted_lines = [CARD_START]
    for field, value in card_fields:
        if not pd.isna(value):
            formatted_lines.append(f"{field}: {value}")
    formatted_lines.append(CARD_END)

    formatted = "\n".join(formatted_lines).lower() + "\n\n"
    return card_name, formatted


card_set = set()
n_val_set = 0
n_train_set = 0

with open(PROCESSED_DATA_DIR / "all.txt", "w") as dataset_f:
    with open(PROCESSED_DATA_DIR / "train.txt", "w") as train_f:
        for _, row in train_df.iterrows():
            card_name, formatted = serialize_card(row)
            if card_name not in card_set:
                dataset_f.write(formatted)
                train_f.write(formatted)
                n_train_set += 1
                card_set.add(card_name)
    with open(PROCESSED_DATA_DIR / "val.txt", "w") as val_f:
        for _, row in val_df.iterrows():
            card_name, formatted = serialize_card(row)
            if card_name not in card_set:
                dataset_f.write(formatted)
                val_f.write(formatted)
                n_val_set += 1
                card_set.add(card_name)
with open(PROCESSED_DATA_DIR / "metadata.json", "w") as metadata_f:
    json.dump(
        {
            "source": str(dataset_path),
            "val_size": VAL_SIZE,
            "n_train_cards": n_train_set,
            "n_val_cards": n_val_set,
        },
        metadata_f,
        indent=2,
    )

print(f"Wrote {n_val_set + n_train_set} cards to {str(PROCESSED_DATA_DIR / 'all.txt')}")
print(f"Wrote {n_train_set} cards to {str(PROCESSED_DATA_DIR / 'train.txt')}")
print(f"Wrote {n_val_set} cards to {str(PROCESSED_DATA_DIR / 'val.txt')}")
