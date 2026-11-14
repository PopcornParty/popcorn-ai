# Popcorn Brain Lab — first original model

This folder is the first **from-scratch model experiment** for Popcorn AI. It implements a small decoder-only Transformer language model in PyTorch. Its architecture, tokenizer, training loop, checkpoint format and sampling code are ours in this repository; PyTorch supplies tensor operations and GPU kernels.

## Honest scope

This is a research prototype, not a ChatGPT/Gemini replacement. A randomly initialised model trained on a small dataset will initially produce poor text. Useful conversational ability requires a large, legally usable dataset, substantial compute, careful evaluation and many training iterations. The existing website's live chat still uses the configured Cloudflare Workers AI model; this experiment does not silently replace it.

## Requirements

- Python 3.10+
- PyTorch 2.2+
- A GPU is helpful but not required. CPU training is much slower.
- Training data you have permission to use. Do not put private conversations, secrets, credentials or personal data in training files.

## Run on Windows

From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r brain/requirements.txt
python brain/train.py --data brain/data/starter.jsonl --out brain/checkpoints/first-run --max-iters 300
python brain/generate.py --checkpoint brain/checkpoints/first-run/latest.pt --prompt "Explain what a Python function is:"
python brain/evaluate.py --checkpoint brain/checkpoints/first-run/latest.pt --data brain/data/starter.jsonl
```

If PowerShell blocks activation, run the commands in Command Prompt instead, replacing activation with `.venv\Scripts\activate.bat`.

## Data format

One JSON object per line, with a `text` string:

```json
{"text":"A Python function is defined with def, followed by its name and parameters."}
```

The included starter file is only a tiny smoke-test corpus. It is not enough to create a capable assistant. Use curated, licensed and reviewed datasets for serious experiments.

## Training controls

- `--max-iters`: number of optimiser steps (default 1000)
- `--batch-size`: examples per step (default 32)
- `--block-size`: context length in characters (default 128)
- `--eval-interval`: validation interval (default 100)
- `--device auto|cpu|cuda|mps`: compute device
- `--seed`: repeatable initialisation

Checkpoints include the model weights, vocabulary and model configuration. Keep multiple checkpoints; do not overwrite a known-good model.

## Evaluation and safe improvement

`evaluate.py` measures held-out next-character loss/perplexity. That is a useful training signal, **not proof of factual accuracy or coding ability**. `benchmark.py` runs a tiny coding-output smoke test using required text fragments; it does not execute code or verify semantic correctness. Use broader human-reviewed coding and conversation tests before considering a model improvement successful.

Run the coding smoke test after training:

```powershell
python brain/benchmark.py --checkpoint brain/checkpoints/first-run/latest.pt
```

Run the unit tests from the brain directory:

```powershell
Set-Location brain
python -m unittest discover -s tests -v
Set-Location ..
```

This lab deliberately does not let the model rewrite or deploy its own code, access GitHub credentials, execute generated code, or promote new weights automatically. A future evolution service should run experiments in isolation, compare against fixed regression benchmarks, retain rollback checkpoints and require explicit approval before production changes.

## Architecture

- `src/popcorn_brain.py`: character tokenizer, causal self-attention, Transformer blocks and model.
- `train.py`: training/validation loop and checkpointing.
- `generate.py`: autoregressive text sampling.
- `evaluate.py`: held-out loss evaluation.
- `tests/`: basic architecture and causal-mask tests.

Vercel remains the web host. It is not the right place to run long GPU training jobs; train on a suitable machine or dedicated compute host, then store/version the resulting checkpoint separately. The web app should only connect to a model-serving endpoint when one is deliberately configured.
