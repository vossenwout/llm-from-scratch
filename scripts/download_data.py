from pathlib import Path

import kagglehub

DATASET = "hammadus/yugioh-full-card-database-index-august-1st-2025"
DATA_DIR = Path("data")

DATA_DIR.mkdir(exist_ok=True)

downloaded_path = kagglehub.dataset_download(DATASET, output_dir=str(DATA_DIR))

print("Downloaded dataset to:", downloaded_path)
