"""Small decoder-only Transformer and tokenizers for Popcorn AI research."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    def validate(self):
        if self.vocab_size < 3: raise ValueError("vocab_size must be at least 3")
        if self.n_head < 1 or self.n_embd % self.n_head: raise ValueError("n_embd must be divisible by a positive n_head")
        if min(self.block_size, self.n_embd, self.n_layer) < 1: raise ValueError("Model dimensions must be positive")
        if not 0 <= self.dropout < 1: raise ValueError("dropout must be in [0, 1)")

class ByteTokenizer:
    """UTF-8 byte tokenizer: every byte is representable, including source code."""
    eos_token_id = 256
    vocab_size = 257
    tokenizer_type = "byte-v1"
    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        return ids + ([self.eos_token_id] if add_eos else [])
    def decode(self, ids: Iterable[int]) -> str:
        values = []
        for token in ids:
            token = int(token)
            if token == self.eos_token_id: break
            if 0 <= token <= 255: values.append(token)
        return bytes(values).decode("utf-8", errors="replace")
    def to_dict(self): return {"type": self.tokenizer_type, "vocab_size": self.vocab_size, "eos_token_id": self.eos_token_id}

class CharTokenizer:
    """Legacy tokenizer retained so version-1 experimental checkpoints remain readable."""
    UNK, EOS, tokenizer_type = "<unk>", "<eos>", "char-v1"
    def __init__(self, vocabulary: list[str]):
        if len(vocabulary) < 3 or vocabulary[0] != self.UNK or vocabulary[1] != self.EOS: raise ValueError("Vocabulary must start with <unk>, <eos>")
        if len(set(vocabulary)) != len(vocabulary): raise ValueError("Vocabulary entries must be unique")
        self.vocabulary = vocabulary
        self.stoi = {token: index for index, token in enumerate(vocabulary)}
        self.vocab_size, self.eos_token_id = len(vocabulary), 1
    @classmethod
    def from_texts(cls, texts: Iterable[str]): return cls([cls.UNK, cls.EOS, *sorted({c for text in texts for c in text})])
    def encode(self, text: str, add_eos: bool = False):
        ids = [self.stoi.get(char, 0) for char in text]
        return ids + ([self.eos_token_id] if add_eos else [])
    def decode(self, ids: Iterable[int]):
        output = []
        for index in ids:
            token = self.vocabulary[int(index)] if 0 <= int(index) < len(self.vocabulary) else self.UNK
            if token == self.EOS: break
            if token != self.UNK: output.append(token)
        return "".join(output)
    def to_dict(self): return {"type": self.tokenizer_type, "vocabulary": self.vocabulary, "vocab_size": self.vocab_size, "eos_token_id": self.eos_token_id}

def tokenizer_from_checkpoint(checkpoint: dict):
    metadata = checkpoint.get("tokenizer")
    if metadata and metadata.get("type") == ByteTokenizer.tokenizer_type: return ByteTokenizer()
    vocabulary = metadata.get("vocabulary") if metadata else checkpoint.get("vocabulary")
    if not vocabulary: raise ValueError("Checkpoint has no supported tokenizer metadata")
    return CharTokenizer(vocabulary)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head, dropout=config.dropout, batch_first=True)
        self.register_buffer("causal_mask", torch.triu(torch.ones(config.block_size, config.block_size, dtype=torch.bool), diagonal=1), persistent=False)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        length = x.size(1)
        output, _ = self.attn(x, x, x, attn_mask=self.causal_mask[:length, :length], need_weights=False)
        return self.dropout(output)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1, self.attention = nn.LayerNorm(config.n_embd), CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.feed_forward = nn.Sequential(nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(), nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.dropout))
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))

class PopcornBrain(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.language_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.language_head.weight = self.token_embedding.weight
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, token_ids, targets=None):
        _, length = token_ids.shape
        if length > self.config.block_size: raise ValueError(f"Input length {length} exceeds block size {self.config.block_size}")
        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        logits = self.language_head(self.final_norm(self.blocks(x)))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss
    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens=160, temperature=0.8, top_k=40, eos_token_id=1):
        if temperature <= 0: raise ValueError("temperature must be greater than zero")
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(token_ids[:, -self.config.block_size:])
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
            if next_token.item() == eos_token_id: break
        return token_ids

def model_config_from_dict(values: dict):
    allowed = set(ModelConfig.__dataclass_fields__)
    return ModelConfig(**{key: value for key, value in values.items() if key in allowed})
