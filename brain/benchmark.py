import argparse
import json
from pathlib import Path

import torch

from src.popcorn_brain import CharTokenizer, PopcornBrain, model_config_from_dict
from train import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple, non-executing coding-output smoke checks")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmarks", type=Path, default=Path("brain/benchmarks/coding_smoke.jsonl"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=160)
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

    cases = []
    for line_number, line in enumerate(args.benchmarks.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item.get("prompt"), str) or not isinstance(item.get("must_include"), list):
            raise ValueError(f"Benchmark line {line_number} needs prompt and must_include fields")
        cases.append(item)
    if not cases:
        raise ValueError("No benchmark cases found")

    results = []
    for case in cases:
        prompt_ids = tokenizer.encode(case["prompt"]) or [1]
        context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(context, max_new_tokens=args.max_new_tokens, temperature=0.01, top_k=1)
        answer = tokenizer.decode(generated[0].tolist())[len(case["prompt"]):]
        missing = [fragment for fragment in case["must_include"] if fragment not in answer]
        results.append({"id": case.get("id", f"case-{len(results) + 1}"), "passed": not missing, "missing_fragments": missing, "output": answer[:500]})

    passed = sum(1 for result in results if result["passed"])
    print(json.dumps({
        "passed": passed,
        "total": len(results),
        "pass_rate": round(passed / len(results), 4),
        "results": results,
        "warning": "This is a tiny string-presence smoke test, not a measure of general programming ability. It does not execute generated code or verify semantic correctness."
    }, indent=2))


if __name__ == "__main__":
    main()
