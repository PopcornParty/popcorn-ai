# Popcorn AI

Popcorn AI is a personal, browser-based AI chat workspace.

## Live setup

- Frontend: Vercel
- AI endpoint: Cloudflare Worker
- Model: Cloudflare Workers AI `@cf/zai-org/glm-4.7-flash`
- Conversation history: saved in the browser's local storage
- Optional server-side conversation logging: Cloudflare D1

The browser calls the Vercel `/api/chat` endpoint. The Vercel function forwards requests to the Cloudflare Worker, so the Cloudflare endpoint is not called directly by client-side code.

## Current interface features

- Chat history with search
- Create, rename and delete conversations
- Export conversation history as JSON
- Copy assistant replies and code blocks
- Regenerate an assistant response
- Attach supported text/code files (up to 150 KB per file; contents are included in the prompt)
- Responsive desktop and mobile layout
- Existing `popcorn_chats` browser history format is retained

## Deployment

The frontend is deployed through Vercel from GitHub. Changes pushed to the production branch deploy to the main site; changes on feature branches may receive a Vercel preview deployment if preview deployments are enabled.

The API proxy is in `api/chat.js`. It forwards chat requests to the Cloudflare Worker at `https://popcorn-ai.popcornparty1yt.workers.dev/api/chat`.

The Cloudflare Worker requires:
- Workers AI binding named `AI`
- D1 binding named `DB` (used for rate limits and optional message logging)

## Important limitations

This version is an AI chat workspace, not yet an autonomous coding agent. It does not currently have permission to inspect or edit GitHub repositories, run code, or deploy projects on its own. File attachments currently support text-based files only. Chat history in the sidebar is stored in the current browser and does not automatically sync between devices.

Cloudflare's free usage is subject to provider quotas and limits.

## Experimental from-scratch model lab

The repository now includes `brain/`, a separate PyTorch character-level Transformer experiment with its own tokenizer, training loop, checkpointing, text generation and validation-loss evaluator. It is an early research prototype, not a production chat model. The live chat path remains unchanged and continues to use the Cloudflare Worker model until a separately hosted Popcorn model is trained, evaluated and deliberately integrated. See [brain/README.md](brain/README.md) for setup, dataset format and limitations.
