# Popcorn Brain Lab

An experimental decoder-only Transformer and supporting intelligence-platform components. The model architecture, tokenizer, training loop, evaluation code and supporting systems are implemented in this repository; PyTorch supplies tensor operations and GPU kernels.

## Honest status

This is an early research prototype, not a frontier assistant. A small randomly initialised model trained on the included starter corpus will not be a useful general-purpose chatbot. Natural conversation and strong coding ability require substantial licensed training data, compute, iteration and independent evaluation. The existing website's live chat remains on its existing Cloudflare model; this lab does not replace it.

## Browser-only cloud workflow

The repository includes a GitHub Actions workflow called **Popcorn Brain Cloud Training**. It can run a small training job on a GitHub-hosted runner and save checkpoints as an Actions artifact. Training happens on GitHub's runner, not your PC. The starter corpus is only a pipeline smoke-test corpus; it is not enough for meaningful intelligence. The workflow is manual and does not automatically deploy model weights to Vercel.

To run it without installing anything locally:
1. Open this repository on GitHub.
2. Select **Actions**.
3. Select **Popcorn Brain Cloud Training**.
4. Choose **Run workflow** and use the safe defaults for the first smoke test.
5. Open the completed run and inspect its logs and checkpoint artifact.

GitHub-hosted CPU runners are suitable for small experiments, not frontier-model training. Larger training needs suitable hosted GPU compute and a properly licensed dataset. Availability and free quotas depend on the provider.

## Model and tokenizer

- The model is a decoder-only Transformer with causal self-attention and residual blocks.
- Byte-level UTF-8 tokenisation represents source code and all UTF-8 text without an unknown-character vocabulary.
- The previous character tokenizer remains readable for older experimental checkpoints.
- Training supports document-level train/validation splitting, resumable optimizer state, latest and best checkpoints, gradient clipping and metrics.
- Generation, held-out next-token loss and a multi-language heuristic coding smoke benchmark are included.
- The coding benchmark does not execute generated code or prove semantic correctness.
- Dataset validation checks JSONL structure, duplicates and likely secrets. Manual rights and privacy review is still required.

## Supporting intelligence systems

- Memory prototype: SQLite storage; saving requires explicit confirmation and likely credentials are rejected. It is not connected to live chat and needs a durable server database adapter before production use.
- Reasoning planner: validates structured plans against an allowlist. Plans do not execute tools.
- Coding workspace core: validates coding tasks and proposed patches, blocks unsafe paths and likely secrets. It does not write files or execute code.
- Evolution gate: compares candidate scores and regression metrics. A passing candidate is only eligible for human review; it is never automatically promoted or deployed.

## Data and checkpoints

Each JSONL line must contain a text string. Only use data you have permission to train on. Do not include private conversations, personal data, credentials or secrets. The included starter corpus is a tiny smoke-test dataset.

Checkpoint files use PyTorch's pickle-backed format. Only load checkpoints you created or otherwise trust. The cloud workflow stores artifacts temporarily; it does not connect them to the production website.

## Safety and deployment boundaries

No autonomous repository writes, arbitrary shell execution, credential access, self-deployment or automatic production model promotion is enabled. The live chat and Vercel deployment remain unchanged. Memory is a prototype and is not synced between devices. GitHub Actions installs dependencies on its hosted runner; nothing needs to be downloaded or installed on your PC.

## Repeatable model evaluation

Run the fixed, hand-written development suite against a checkpoint:

\`\`\`bash
python brain/eval_suite.py --checkpoint path/to/latest.pt --output eval-report.json
\`\`\`

The suite covers conversation, basic reasoning, coding syntax, instruction following, basic knowledge and simple safety responses. It uses deterministic greedy decoding and reports results by category. Checks are case-insensitive phrase matches only; they are not semantic grading and cannot establish factuality, reasoning ability or safe behaviour. Keep this evaluation set separate from training data and use the same cases when comparing checkpoints. A small score is a debugging signal, not a claim of intelligence.


## Bounded agent runtime core

`brain/src/agent_runtime.py` adds a small orchestration layer for future
trusted development-tool adapters. It runs at most 20 steps, accepts only
actions named by the constitution, requires a positive per-action approval
callback for protected actions, blocks unknown actions, stops on the first
failure, rejects credential-shaped payload fields, and records a minimal
audit trail without copying task payloads or handler exception text.

**Important boundary:** this is not yet an autonomous repository editor or
sandbox. It does not open files, run shell commands, access GitHub, call the
network, or deploy anything. A host application must register narrowly scoped
handlers and implement filesystem isolation, path/symlink checks, resource
limits and test-command allowlists before any real tools are connected.
