# Popcorn AI

Popcorn AI is a personal, browser-based AI chat workspace with an experimental from-scratch model research lab.

## Live setup

- Frontend: Vercel
- Existing AI endpoint: Cloudflare Worker
- Existing model: Cloudflare Workers AI @cf/zai-org/glm-4.7-flash
- Conversation history: current browser local storage
- Optional server-side conversation logging: Cloudflare D1

The browser calls Vercel /api/chat, which forwards requests to the existing Cloudflare Worker. The experimental model is not connected to live chat and does not replace this route.

## Existing interface

- Searchable chat history; create, rename and delete conversations
- Export history as JSON; copy assistant replies and code blocks; regenerate replies
- Text/code attachments up to 150 KB per file
- Responsive desktop and mobile layout
- Existing popcorn_chats browser history format retained

## Experimental Popcorn Brain

The brain/ directory contains a small decoder-only Transformer, byte-level tokenizer, resumable training pipeline, checkpoints, evaluation, coding smoke benchmark, explicit-consent memory prototype, validated task planning, safe patch proposals and an evidence-based improvement gate.

The GitHub Actions workflow named Popcorn Brain Cloud Training can run small CPU experiments on GitHub-hosted runners and save checkpoint artifacts. This requires no local installation or download on your PC. These experiments are not frontier-model training; the included dataset is only a smoke-test corpus.

See brain/README.md for architecture, cloud workflow and limitations.

## Important boundaries

The live website still relies on its configured Cloudflare model and quotas. The experimental Popcorn model is not yet a production chatbot. The project does not autonomously execute generated code, access GitHub credentials, write to repositories, or deploy new model weights. Memory is a prototype and is not synced between devices.
