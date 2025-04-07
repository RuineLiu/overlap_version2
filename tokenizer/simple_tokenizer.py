import torch

class SimpleTokenizer:
    def __init__(self, vocab_path, max_length=128):
        self.token2id = {}
        self.id2token = {}
        self.max_length = max_length

        with open(vocab_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip()
                self.token2id[token] = idx
                self.id2token[idx] = token

        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.pad_id = self.token2id[self.pad_token]
        self.unk_id = self.token2id[self.unk_token]

    def encode(self, text):
        """
        将 token 分割文本转为 input_ids 和 attention_mask
        """
        tokens = text.strip().split()
        input_ids = [self.token2id.get(tok, self.unk_id) for tok in tokens]

        # 截断
        input_ids = input_ids[:self.max_length]
        attention_mask = [1] * len(input_ids)

        # padding
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.pad_id] * pad_len
            attention_mask += [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }

    def decode(self, ids):
        """
        将 input_ids 转回 token 序列（调试用）
        """
        return [self.id2token.get(i, self.unk_token) for i in ids]
