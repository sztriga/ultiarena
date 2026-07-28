# oldtawer-web

Vite + React + TypeScript SPA for the Ulti research console.
Architecture and tab walkthrough: [`../../docs/UI.md`](../../docs/UI.md).

## Setup

Python deps + Cython compile happen at the repo root — see the project
[`README.md`](../../README.md) first. Then for this package:

```bash
cd apps/web
npm install            # one-time
```

## Run (dev)

From the repo root, launch the API and Vite together:

```bash
./apps/dev.sh          # API :8000, Vite :5173 — Ctrl-C stops both
```

Or split across terminals:

```bash
# 1. API
PYTHONPATH=. uvicorn apps.api.main:app --reload --port 8000

# 2. Web
cd apps/web && npm run dev
```

## Requirements

- Python 3.10+ (root `requirements.txt` covers this)
- Node 20+ (Vite 8 / React 19)

## Build

```bash
npm run build          # emits apps/web/dist/
npm run preview        # preview the built bundle on :4173
```
