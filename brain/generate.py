import argparse
from pathlib import Path

import torch

from src.popcorn_brain import CharTokenizer, PopcornBrain, model_config_from_dict
from train import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a Popcorn Brain checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported or invalid Popcorn Brain checkpoint")
    tokenizer = CharTokenizer(checkpoint["vocabulary"])
    config = model_config_from_dict(checkpoint["model_config"])
    model = PopcornBrain(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    prompt_ids = tokenizer.encode(args.prompt) or [1]
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    result = model.generate(context, args.max_new_tokens, args.temperature, args.top_k)
    print(tokenizer.decode(result[0].tolist()))


if __name__ == "__main__":
    main()
