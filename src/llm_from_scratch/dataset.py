from torch.utils.data import Dataset
from llm_from_scratch.tokenizer import Tokenizer


class YuGiOhCardsDataset(Dataset):
    def __init__(self, dataset_path: str, context_length: int, tokenizer: Tokenizer):
        with open(dataset_path, "r") as f:
            self.yugioh_dataset = f.read()
        self.context_length = context_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.yugioh_dataset) - self.context_length

    def __getitem__(self, index):
        # TODO: apply tokenization before this so we can make the dataset independent of the tokenizer?
        X = self.yugioh_dataset[index : index + self.context_length]
        y = self.yugioh_dataset[index + 1 : index + 1 + self.context_length]
        return self.tokenizer.encode(X), self.tokenizer.encode(y)
