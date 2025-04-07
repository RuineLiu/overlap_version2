import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd

        self.pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape(x):  # (B, T, D) → (B, num_heads, T, head_dim)
            return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = map(reshape, (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, T, T)

        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :] == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        context = attn @ v  # (B, heads, T, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(context)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=2, dim_ff=256, num_classes=3, max_len=128,dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dim_ff, dropout=dropout) for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        x = self.token_emb(input_ids)
        x = self.pos_emb(x)

        for block in self.blocks:
            x = block(x, attention_mask)

        # 使用 mean pooling 作为句向量表示
        mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)  # (B, d_model)

        logits = self.classifier(x)
        return logits
