import argparse
import json
from pathlib import Path

import torch

from src.popcorn_brain import CharTokenizer, PopcornBrain, model_config_from_dict
from train import choose_device, load_texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure held-out next-character loss for a Popcorn Brain checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported or invalid Popcorn Brain checkpoint")
    tokenizer = CharTokenizer(checkpoint["vocabulary"])
    config = model_config_from_dict(checkpoint["model_config"])
    model = PopcornBrain(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    text = "\n".join(load_texts(args.data))
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        raise ValueError("Evaluation text is too short")
    encoded = torch.tensor(ids, dtype=torch.long, device=device)
    losses = []
    with torch.no_grad():
        for start in range(0, len(encoded) - 1, config.block_size):
            x = encoded[start:start + config.block_size]
            y = encoded[start + 1:start + config.block_size + 1]
            if len(x) != len(y) or len(x) == 0:
                continue
            _, loss = model(x.unsqueeze(0), y.unsqueeze(0))
            losses.append(float(loss.item()))
    mean_loss = sum(losses) / len(losses)
    print(json.dumps({
        "checkpoint": str(args.checkpoint),
        "evaluation_file": str(args.data),
        "mean_next_character_loss": round(mean_loss, 6),
        "perplexity": round(float(torch.exp(torch.tensor(mean_loss)).item()), 6),
        "warning": "Loss/perplexity measure character prediction on this dataset; they do not establish factual accuracy, reasoning or coding competence."
    }, indent=2))


if __name__ == "__main__":
    main()
