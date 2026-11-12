# Popcorn AI
A ChatGPT-style web chatbot with a server-side OpenRouter connection.

## Deploy (Vercel)
1. Upload this project to a GitHub repository.
2. Import that repository in Vercel.
3. In Vercel Project Settings → Environment Variables, add `OPENROUTER_API_KEY` with the API key. Optionally add `APP_URL` with your deployed URL.
4. Redeploy. Open the site and chat.

The site uses `openrouter/free`; provider availability, model selection and free quotas may change. The API key is only read by the server function and must never be put in `index.html`.

## Local development
Install Node.js and run `npm install`, then `npm run dev` (Vercel CLI). Configure `OPENROUTER_API_KEY` in the local environment first.
