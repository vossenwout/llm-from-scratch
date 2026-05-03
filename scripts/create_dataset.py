import pandas as pd
from pathlib import Path

from pandas.core.series import Series
from sklearn.model_selection import train_test_split

DATASET = "hammadus/yugioh-full-card-database-index-august-1st-2025"
DATA_DIR = Path("data")
TEST_SIZE = 0.1

dataset_path = sorted(
    DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
)[0]

dataset = pd.read_csv(dataset_path)
train_df, test_df = train_test_split(
    dataset,
    random_state=42,
    test_size=TEST_SIZE,
)


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

    formatted = f"name: {card_name}\ndescription: {card_description}\n\n".lower()
    return card_name, formatted


card_set = set()
n_test_set = 0
n_train_set = 0

with open(DATA_DIR / "dataset.txt", "w") as dataset_f:
    with open(DATA_DIR / "train_set.txt", "w") as train_f:
        for _, row in train_df.iterrows():
            card_name, formatted = serialize_card(row)
            if card_name not in card_set:
                dataset_f.write(formatted)
                train_f.write(formatted)
                n_train_set += 1
                card_set.add(card_name)
    with open(DATA_DIR / "test_set.txt", "w") as test_f:
        for _, row in test_df.iterrows():
            card_name, formatted = serialize_card(row)
            if card_name not in card_set:
                dataset_f.write(formatted)
                test_f.write(formatted)
                n_test_set += 1
                card_set.add(card_name)

print(f"Wrote {n_test_set + n_train_set} cards to {str(DATA_DIR / 'dataset.txt')}")
print(f"Wrote {n_train_set} cards to {str(DATA_DIR / 'train_set.txt')}")
print(f"Wrote {n_test_set} cards to {str(DATA_DIR / 'test_set.txt')}")
