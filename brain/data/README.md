# Training data policy

Each JSONL line must contain a text string and provenance metadata:

```json
{"text":"A complete training example.","metadata":{"source":"Exact source","license":"Exact licence or rights basis","permission":"Why model training is permitted","reviewed":true}}
```

Before adding data:
- Use only text you wrote or material whose licence/permission explicitly allows model training.
- Record the exact source and licence/rights basis for every record.
- Set `reviewed` to `true` only after a person checks the source, licence and content.
- Do not use private chats, personal records, emails, credentials, secrets or scraped material with unclear rights.
- Keep held-out evaluation material separate from training material.
- A passing validator is not legal clearance; it cannot verify ownership or licence terms.

The starter corpus is marked for repository smoke-test use only. It is deliberately tiny and is not a meaningful language-model training set.
