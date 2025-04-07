import torch
from torch.utils.data import Dataset
import pandas as pd

class StruGeneDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        """
        :param csv_file: 预处理后的CSV文件路径（含 input_text, label）
        :param tokenizer: 实例化后的 SimpleTokenizer
        :param max_length: 最大序列长度
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["input_text"]
        label = int(row["label"])

        encoding = self.tokenizer.encode(text)  # returns dict with input_ids and attention_mask

        return {
            "input_ids": encoding["input_ids"],               # Tensor, shape [max_length]
            "attention_mask": encoding["attention_mask"],     # Tensor, shape [max_length]
            "label": torch.tensor(label, dtype=torch.long)     # Tensor
        }
