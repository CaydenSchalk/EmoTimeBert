import torch
from torch.utils.data import Dataset

class EmpatheticDialoguesDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_len=128):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv = self.conversations[idx]

        return {
            "texts": conv["texts"],
            "labels": torch.tensor(conv["labels"], dtype=torch.long),
            "timestamps": torch.tensor(conv["timestamps"], dtype=torch.long),
            "speakers": torch.tensor(conv["speakers"], dtype=torch.long)
        }