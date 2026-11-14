"""Small, reproducible lexical evaluation harness for experimental Popcorn checkpoints.

This reports observable string checks only. It does not judge semantic correctness.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import torch
from src.popcorn_brain import PopcornBrain, model_config_from_dict, tokenizer_from_checkpoint
from train import choose_device


def load_cases(path: Path) -> list[dict]:
    cases = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on evaluation line {line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation line {line_number} must be a JSON object")
        case_id = item.get("id")
        category = item.get("category")
        prompt = item.get("prompt")
        required = item.get("must_include", [])
        forbidden = item.get("must_not_include", [])
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError(f"Evaluation line {line_number} has a missing or duplicate id")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"Evaluation line {line_number} needs a category")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Evaluation line {line_number} needs a prompt")
        if not isinstance(required, list) or not all(isinstance(x, str) and x for x in required):
            raise ValueError(f"Evaluation line {line_number} has invalid must_include terms")
        if not isinstance(forbidden, list) or not all(isinstance(x, str) and x for x in forbidden):
            raise ValueError(f"Evaluation line {line_number} has invalid must_not_include terms")
        seen.add(case_id)
        cases.append({**item, "id": case_id, "category": category, "prompt": prompt,
                      "must_include": required, "must_not_include": forbidden})
    if not cases:
        raise ValueError("Evaluation file contains no cases")
    return cases


def score_response(case: dict, response: str) -> dict:
    folded = response.casefold()
    missing = [term for term in case.get("must_include", []) if term.casefold() not in folded]
    forbidden = [term for term in case.get("must_not_include", []) if term.casefold() in folded]
    return {
        "id": case["id"],
        "category": case["category"],
        "passed": not missing and not forbidden,
        "missing_required_terms": missing,
        "found_forbidden_terms": forbidden,
        "response": response[:1000],
    }


def summarise(results: list[dict]) -> dict:
    grouped = defaultdict(list)
    for result in results:
        grouped[result["category"]].append(result)
    categories = {
        name: {"passed": sum(x["passed"] for x in items), "total": len(items),
               "pass_rate": round(sum(x["passed"] for x in items) / len(items), 4)}
        for name, items in sorted(grouped.items())
    }
    passed = sum(x["passed"] for x in results)
    return {
        "passed": passed,
        "total": len(results),
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "categories": categories,
        "results": results,
        "limitations": [
            "Checks are case-insensitive substring matches, not semantic grading.",
            "A passing phrase check does not prove factual accuracy, reasoning, safety or useful coding.",
            "Use the same fixed evaluation set across candidate checkpoints and keep it out of training data.",
            "This tiny hand-written suite is a development smoke test, not a recognised benchmark."
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Run a fixed multi-category Popcorn Brain evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("brain/benchmarks/general_quality.jsonl"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("format_version") not in (1, 2):
        raise ValueError("Unsupported checkpoint format")
    tokenizer = tokenizer_from_checkpoint(checkpoint)
    config = model_config_from_dict(checkpoint["model_config"])
    model = PopcornBrain(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    results = []
    for case in load_cases(args.cases):
        prompt_ids = tokenizer.encode(case["prompt"]) or [tokenizer.eos_token_id]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            generated = model.generate(input_ids, args.max_new_tokens, temperature=1.0,
                                       top_k=1, eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(generated[0, len(prompt_ids):].tolist())
        results.append(score_response(case, response))

    report = summarise(results)
    report["checkpoint"] = str(args.checkpoint)
    report["evaluation_file"] = str(args.cases)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
