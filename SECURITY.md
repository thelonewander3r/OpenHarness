# Security

## Secrets

**Do not commit:**

- `.env` or any file containing real API keys
- `.venv/`, credentials JSON, PEM files, tokens in config

This repository ships **`.env.example`** with placeholders only. Copy to `.env` locally.

## Before you push

```bash
git status
# Ensure .env is NOT listed

git check-ignore -v .env
# Should show .gitignore rule
```

If `.env` was ever committed, remove it from history, **rotate the key** at [OpenRouter](https://openrouter.ai/keys), and force-push only after history rewrite.

## Reporting

Open an issue on [github.com/eman1369a/OpenHarness](https://github.com/eman1369a/OpenHarness/issues) for security concerns (no public key material in the issue body).
