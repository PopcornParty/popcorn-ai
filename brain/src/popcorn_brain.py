"""A compact, character-level decoder-only Transformer for Popcorn AI experiments."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
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

    def validate(self) -> None:
        if self.vocab_size < 3:
            raise ValueError("vocab_size must include UNK, EOS and at least one character")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if min(self.block_size, self.n_embd, self.n_head, self.n_layer) < 1:
            raise ValueError("Model dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class CharTokenizer:
    UNK = "<unk>"
    EOS = "<eos>"

    def __init__(self, vocabulary: list[str]):
        if len(vocabulary) < 3 or vocabulary[0] != self.UNK or vocabulary[1] != self.EOS:
            raise ValueError("Vocabulary must start with <unk>, <eos>")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("Vocabulary entries must be unique")
        self.vocabulary = vocabulary
        self.stoi = {token: index for index, token in enumerate(vocabulary)}

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "CharTokenizer":
        chars = sorted({char for text in texts for char in text})
        return cls([cls.UNK, cls.EOS, *chars])

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = [self.stoi.get(char, 0) for char in text]
        if add_eos:
            ids.append(1)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        output = []
        for index in ids:
            token = self.vocabulary[int(index)] if 0 <= int(index) < len(self.vocabulary) else self.UNK
            if token == self.EOS:
                break
            if token != self.UNK:
                output.append(token)
        return "".join(output)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head, dropout=config.dropout, batch_first=True)
        self.register_buffer("causal_mask", torch.triu(torch.ones(config.block_size, config.block_size, dtype=torch.bool), diagonal=1), persistent=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        output, _ = self.attn(x, x, x, attn_mask=self.causal_mask[:length, :length], need_weights=False)
        return self.dropout(output)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        batch, length = token_ids.shape
        if length > self.config.block_size:
            raise ValueError(f"Input length {length} exceeds block size {self.config.block_size}")
        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        x = self.final_norm(self.blocks(x))
        logits = self.language_head(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int = 160, temperature: float = 0.8, top_k: int = 40) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        self.eval()
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
            if next_token.item() == 1:
                break
        return token_ids


def model_config_from_dict(values: dict) -> ModelConfig:
    allowed = set(ModelConfig.__dataclass_fields__)
    return ModelConfig(**{key: value for key, value in values.items() if key in allowed})
