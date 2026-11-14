from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from src.popcorn_brain import CharTokenizer, ModelConfig, PopcornBrain


def load_texts(path: Path) -> list[str]:
    texts = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError(f"Line {line_number} must contain a non-empty string field named 'text'")
        texts.append(item["text"])
    if len(texts) < 2:
        raise ValueError("Provide at least two non-empty training examples")
    return texts


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA GPU")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the experimental Popcorn character-level Transformer")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("brain/checkpoints/experiment"))
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_iters < 1 or args.batch_size < 1 or args.block_size < 8 or args.eval_interval < 1 or args.learning_rate <= 0:
        parser.error("Training values must be positive; block size must be at least 8")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    texts = load_texts(args.data)
    tokenizer = CharTokenizer.from_texts(texts)
    stream = torch.tensor([token for text in texts for token in tokenizer.encode(text, add_eos=True)], dtype=torch.long)
    if len(stream) < args.block_size + 2:
        raise ValueError(f"Dataset has only {len(stream)} tokens; need at least {args.block_size + 2}. Add more text or lower --block-size.")
    split = max(args.block_size + 1, int(len(stream) * 0.9))
    split = min(split, len(stream) - 1)
    train_data, val_data = stream[:split], stream[split:]
    if len(val_data) < args.block_size + 1:
        val_data = train_data[-(args.block_size + 1):]
    device = choose_device(args.device)

    config = ModelConfig(vocab_size=len(tokenizer.vocabulary), block_size=args.block_size)
    model = PopcornBrain(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.out.mkdir(parents=True, exist_ok=True)

    def get_batch(source: torch.Tensor):
        max_start = len(source) - args.block_size - 1
        starts = torch.randint(max_start + 1, (args.batch_size,))
        x = torch.stack([source[int(i):int(i) + args.block_size] for i in starts]).to(device)
        y = torch.stack([source[int(i) + 1:int(i) + args.block_size + 1] for i in starts]).to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        result = {}
        for name, source in (("train", train_data), ("validation", val_data)):
            losses = []
            for _ in range(10):
                x, y = get_batch(source)
                _, loss = model(x, y)
                losses.append(float(loss.item()))
            result[name] = sum(losses) / len(losses)
        model.train()
        return result

    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}; vocabulary: {len(tokenizer.vocabulary)}")
    history = []
    for step in range(args.max_iters):
        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            metrics = estimate_loss()
            history.append({"step": step, **metrics})
            print(f"step {step:5d} | train loss {metrics['train']:.4f} | validation loss {metrics['validation']:.4f}")
            checkpoint = {
                "format_version": 1,
                "model_config": config.__dict__,
                "vocabulary": tokenizer.vocabulary,
                "model_state": model.state_dict(),
                "training": {"step": step, "seed": args.seed, "device": str(device), "metrics": history},
            }
            torch.save(checkpoint, args.out / "latest.pt")
            (args.out / "metrics.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        x, y = get_batch(train_data)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print(f"Saved checkpoint: {args.out / 'latest.pt'}")
    print("Training finished. Compare validation metrics and run independent tests before trusting a checkpoint.")


if __name__ == "__main__":
    main()
