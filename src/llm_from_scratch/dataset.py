from torch.utils.data import Dataset
from llm_from_scratch.tokenizer import Tokenizer


class YuGiOhCardsDataset(Dataset):
    def __init__(self, dataset_path: str, context_length: int, tokenizer: Tokenizer):
        with open(dataset_path, "r") as f:
            text = f.read()
        self.context_length = context_length
        self.tokenizer = tokenizer
        # tokenize up front (for larger datasets we would tokenize ahead)
        self.token_ids = self.tokenizer.encode(text)

    def __len__(self):
        return len(self.token_ids) - self.context_length

    def __getitem__(self, index):
        X = self.token_ids[index : index + self.context_length]
        y = self.token_ids[index + 1 : index + 1 + self.context_length]
        return X, y
