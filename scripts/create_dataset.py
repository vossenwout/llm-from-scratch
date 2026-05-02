import pandas as pd
from pathlib import Path

from pandas.core.series import Series

DATASET = "hammadus/yugioh-full-card-database-index-august-1st-2025"
DATA_DIR = Path("data")

dataset_path = sorted(
    DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
)[0]

df = pd.read_csv(dataset_path)


def serialize_card(card_row: Series):
    card_name = card_row["name"].replace('"', "")
    card_description = card_row["description"]
    # include these later
    card_rarity = card_row["rarity"]
    card_price = card_row["price"]
    card_type = card_row["type"]
    card_rank = card_row["rank"]
    card_attack = card_row["attack"]
    card_defense = card_row["defense"]

    formatted = f"name: {card_name}\ndescription: {card_description}\n\n"
    return card_name, formatted


card_set = set()
with open(DATA_DIR / "dataset.txt", "w") as f:
    for _, row in df.iterrows():
        card_name, formatted = serialize_card(row)
        if card_name not in card_set:
            f.write(formatted)
            card_set.add(card_name)

print(f"Wrote {len(card_set)} cards to {str(DATA_DIR / 'dataset.txt')}")
