# ReviewerMatch

**Semantic researcher discovery — ranked reviewer & collaborator shortlists from a paper abstract, in under a second.**

[![API Status](https://img.shields.io/badge/API-live-22c55e?style=flat-square)](https://reviewermatch.vercel.app)
[![Web App](https://img.shields.io/badge/web-reviewermatch.vercel.app-000?style=flat-square&logo=vercel)](https://reviewermatch.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-1.8-0467DF?style=flat-square)](https://github.com/facebookresearch/faiss)
[![Sentence Transformers](https://img.shields.io/badge/sentence--transformers-all--MiniLM--L6--v2-FFB300?style=flat-square)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Railway](https://img.shields.io/badge/deploy-Railway-0B0D0E?style=flat-square&logo=railway&logoColor=white)](https://railway.app)
[![Vercel](https://img.shields.io/badge/frontend-Vercel-000?style=flat-square&logo=vercel)](https://vercel.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-A31F34?style=flat-square)](LICENSE)

**[Live Demo](https://reviewermatch.vercel.app)** · **[API Docs](#api-reference)** · **[Roadmap](#roadmap)** · **[Security](#security)** · **[License](#license)**

---

![ReviewerMatch — Hero](docs/screenshots/Hero.png)

*Drop in a paper abstract, get a ranked shortlist of active ML / CS researchers — by meaning, not keywords.*

---

## What is ReviewerMatch?

ReviewerMatch is a full-stack ML application that scores every indexed researcher against a paper abstract or research description — instantly. Unlike keyword-based author search tools, ReviewerMatch embeds both the query and researcher profiles into a shared semantic space, retrieves the closest candidates with a FAISS vector index, and re-ranks them using semantic similarity, **h-index**, and **recency**.

**No keyword guessing. No trawling Google Scholar. A ranked shortlist in < 1 s — reviewers, collaborators, citation candidates, or PhD supervisors.**

---

## Screenshots

| Landing / query | Ranked results |
| --- | --- |
| ![Landing](docs/screenshots/Hero.png) | ![Results](docs/screenshots/Results.png) |

*Left: Landing Page for the Web app running the ML-API.  Right: ranked researcher cards with h-index / works / citations pills and recent-work explanations.*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Static web + Edge proxy · Vercel                           │
│  reviewermatch.vercel.app                                   │
│                                                             │
│  web/index.html · Build on modern CSS components                   │
│  api/match.js   · Vercel Edge → proxies to Railway          │
│         │  POST /api/match  (X-API-Key injected server-side)│
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI · Python 3.11 · Docker · Railway                   │
│                                                             │
│  1. Embed query  → sentence-transformers (all-MiniLM-L6-v2) │
│  2. Retrieve     → FAISS IndexFlatIP (cosine via L2-norm)   │
│  3. Filter       → h-index floor · active-since year ·      │
│                    country allow-list · exclude co-authors  │
│  4. Rerank       → weighted: similarity + h-index + recency │
│  5. Hydrate      → recent works (explain the match)         │
│         │                                                   │
│  PostgreSQL (Railway)  ← author rows + embedding bytes      │
└─────────────────────────────────────────────────────────────┘
                           ▲
                           │  seeded once, offline
                           │
              ┌──────────────────────────┐
              │  Colab · T4 GPU          │
              │  sentence-transformers   │
              │  → embedding bytes       │
              │  → scripts/seed_postgres │
              └──────────────────────────┘
```

All ML inference happens **offline on Colab** and seeds Railway Postgres with pre-computed embedding vectors. Railway's free tier (1 GB RAM, 2 CPU) only has to reconstruct the FAISS index from those bytes at boot — it never runs the model.

---

## Data Source

| Source | Records | What it contains |
| --- | --- | --- |
| [OpenAlex](https://openalex.org) — ML / CS researchers | ~150 k (active, 2022+) | Authors, institutions, h-index, works, concepts, co-authorship |

All data is sourced from free public OpenAlex endpoints — no paid subscriptions required. Polite-pool access via `OPENALEX_EMAIL` is strongly recommended.

---

## ML Pipeline

```
Abstract / research description
    │
    ▼
sentence-transformers (all-MiniLM-L6-v2)   — 384-dim embedding, L2-normalised
    │
    ▼
FAISS IndexFlatIP search  (top-K candidates, cosine via inner product)
    │
    ▼
Hard filters
    ├── min_h_index           (drop below floor)
    ├── require_active_since  (drop authors with no works since year)
    ├── countries             (ISO allow-list)
    └── exclude_openalex_ids  (e.g. your own co-authors)
    │
    ▼
Rerank (weighted heuristic → score ∈ [0, 100])
    ├── semantic_similarity   (dominant weight)
    ├── h_index (log-scaled)
    └── recency (last_known_activity_year)
    │
    ▼
Hydrate with matching_works  (recent titles that explain the score)
```

The reranker lives in `app/services/ranker.py` and is deliberately transparent — no black-box ML between retrieval and the user. A trained cross-encoder re-ranker is on the [roadmap](#roadmap).

---

## Live Services

| Service      | URL                                             | Platform |
| ------------ | ----------------------------------------------- | -------- |
| Web app      | <https://reviewermatch.vercel.app>              | Vercel   |
| API          | *(private — proxied via Vercel edge function)*  | Railway  |
| Swagger UI   | `/docs` on the Railway deployment               | Railway  |
| Health check | `/api/v1/health`                                | Railway  |

---

## API Reference

### Authentication

All protected endpoints require an API key in the request header:

```
X-API-Key: <your-api-key>
```

In production the key is held by the Vercel edge function (`api/match.js`) and never exposed to the browser.

### Endpoints

| Method | Path                    | Auth | Description                                                      |
| ------ | ----------------------- | ---- | ---------------------------------------------------------------- |
| GET    | `/`                     | —    | Service banner                                                   |
| GET    | `/health`               | —    | Lightweight liveness probe                                       |
| GET    | `/api/v1/`              | —    | API metadata (name, version, docs link)                          |
| GET    | `/api/v1/health`        | —    | Full health: model loaded, authors in DB, FAISS vectors, last ingestion |
| POST   | `/api/v1/match`         | ✓    | **Core endpoint** — match an abstract against the researcher index |
| GET    | `/api/v1/author/{id}`   | ✓    | Retrieve a single author by database ID                          |
| GET    | `/docs`                 | —    | Interactive Swagger UI                                           |

---

### `POST /api/v1/match`

Match an abstract against the researcher index and receive a ranked shortlist with scores and matching recent works.

**Request body**

```json
{
  "query": "We develop spiking neural networks for energy-efficient anomaly detection in network intrusion datasets, comparing timestep–performance tradeoffs on NSL-KDD and UNSW-NB15. Our method achieves comparable F1 to dense ANNs with 8× fewer spikes.",
  "top_n": 20,
  "min_h_index": 10,
  "countries": ["GB", "US"],
  "exclude_openalex_ids": ["A5012345678"],
  "require_active_since": 2022
}
```

| Field                   | Type              | Required | Description                                                              |
| ----------------------- | ----------------- | -------- | ------------------------------------------------------------------------ |
| `query` / `abstract`    | string, 50–5000   | **Yes**  | Abstract or research description used for semantic matching              |
| `top_n`                 | int 1–50          | No       | Results to return (default: 20)                                          |
| `min_h_index`           | int 0–200         | No       | Drop researchers below this h-index                                      |
| `min_works`             | int ≥ 0           | No       | Drop researchers with fewer works                                        |
| `countries`             | string[]          | No       | ISO-2 country allow-list, e.g. `["GB", "US"]`                            |
| `exclude_openalex_ids`  | string[]          | No       | OpenAlex author IDs to exclude (useful for your own co-authors)          |
| `require_active_since`  | int year          | No       | Require at least one work in or after this year (default: 2022)          |

**Response**

```json
{
  "query_summary": "We develop spiking neural networks for energy-efficient anomaly detection...",
  "total_matched": 20,
  "processing_time_ms": 142.7,
  "data_freshness": "2026-04-16",
  "results": [
    {
      "openalex_id": "A5012345678",
      "display_name": "Dr Jane Doe",
      "orcid": "0000-0002-1825-0097",
      "institution_name": "University of Edinburgh",
      "institution_country": "GB",
      "h_index": 34,
      "i10_index": 78,
      "works_count": 112,
      "cited_by_count": 4821,
      "top_concepts": ["spiking neural networks", "anomaly detection", "neuromorphic"],
      "score": 91.4,
      "similarity": 0.782,
      "matching_works": [
        { "id": "W4...", "title": "Energy-efficient SNNs for intrusion detection", "year": 2024, "venue": "NeurIPS" }
      ],
      "profile_url": "https://openalex.org/A5012345678"
    }
  ]
}
```

**curl example**

```bash
curl -s -X POST https://<your-railway-app>/api/v1/match \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "query": "spiking neural networks for energy-efficient anomaly detection on NSL-KDD and UNSW-NB15",
    "top_n": 10,
    "min_h_index": 10,
    "require_active_since": 2022
  }' | python -m json.tool
```

**PowerShell example (Windows)**

```powershell
$body = @{
    query                = "spiking neural networks for energy-efficient anomaly detection on NSL-KDD and UNSW-NB15"
    top_n                = 10
    min_h_index          = 10
    require_active_since = 2022
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "https://<your-railway-app>/api/v1/match" `
  -Method POST `
  -Headers @{ "X-API-Key" = "YOUR_API_KEY"; "Content-Type" = "application/json" } `
  -Body $body
```

---

## Local Development

### Prerequisites

- Python 3.11+
- PostgreSQL (or SQLite for dev — default)
- *(Optional)* Google Colab with T4 GPU for full embedding runs

### Setup

```bash
git clone https://github.com/samiurk70/reviewermatch-api.git
cd reviewermatch-api

# Install Python dependencies
pip install -r requirements.txt

# Copy and edit environment variables
cp .env.example .env
# → set OPENALEX_EMAIL (required), API_KEY, DATABASE_URL

# Quick smoke test — ingest ~500 authors, CPU-only
INGEST_SAMPLE_SIZE=500 python -m scripts.ingest_sample   # → data/authors_raw.jsonl
python -m scripts.load_jsonl                             # → SQLite DB
python -m scripts.build_index                            # → data/authors.faiss

# Start the API server
uvicorn app.main:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs   (Swagger UI)
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

### Docker

```bash
docker-compose up --build
# API at http://localhost:8000
```

The Dockerfile pre-downloads `all-MiniLM-L6-v2` at build time so the container starts quickly and never risks Railway's healthcheck window.

---

## Data Pipeline — seeding Railway

> Railway's free tier cannot run ML training or re-embedding. All embedding work happens on Colab; Railway only rebuilds the FAISS index from pre-computed vectors stored in Postgres.

### 1. Generate vectors on Colab (T4 GPU)

Open `scripts/colab_notebook.ipynb` in Google Colab, run all cells, download `reviewermatch_bundle.tar.gz`, and extract:

```bash
tar -xzf reviewermatch_bundle.tar.gz -C data/
# produces: data/authors_meta.json  data/authors.faiss
```

### 2. Seed Railway Postgres (once, locally)

```bash
DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/railway" \
    python -m scripts.seed_postgres
```

This uploads every author row **plus** its embedding vector bytes. Railway will never need to run ML inference — it rebuilds FAISS from these bytes on boot.

### 3. Redeploy Railway

On startup the API will:

1. Detect no `data/authors.faiss` (the container is stateless).
2. Read `embedding_vector` bytes from Postgres.
3. Reconstruct the FAISS index in RAM (seconds, no ML).
4. Serve requests.

Check `/api/v1/health` — you should see `authors_in_db > 0` and `index_built: true`.

---

## Environment Variables

### API (`.env`)

| Variable                 | Default                                    | Description                                                              |
| ------------------------ | ------------------------------------------ | ------------------------------------------------------------------------ |
| `DATABASE_URL`           | `sqlite+aiosqlite:///data/reviewermatch.db`| SQLAlchemy async URL. Postgres: `postgresql+asyncpg://...`               |
| `EMBEDDING_MODEL`        | `all-MiniLM-L6-v2`                         | HuggingFace sentence-transformer model name                              |
| `FAISS_INDEX_PATH`       | `data/authors.faiss`                       | Path to FAISS index file                                                 |
| `API_KEY`                | `changeme`                                 | Bearer key for protected endpoints (`X-API-Key` header)                  |
| `MAX_RESULTS`            | `50`                                       | Hard cap on results returned per request                                 |
| `OPENALEX_EMAIL`         | —                                          | **Required** for polite-pool access to OpenAlex                          |
| `OPENALEX_BASE_URL`      | `https://api.openalex.org`                 | OpenAlex API base URL                                                    |
| `OPENALEX_API_KEY`       | —                                          | *(Optional)* premium OpenAlex key — **store in Railway dashboard only**  |
| `ANTHROPIC_API_KEY`      | —                                          | *(Optional)* Claude key for natural-language explanations                |
| `FREE_TIER_DAILY_LIMIT`  | `3`                                        | Per-IP daily search cap on the free tier                                 |

> ⚠️ `API_KEY`, `DATABASE_URL`, `OPENALEX_API_KEY`, and `ANTHROPIC_API_KEY` are secrets. Store them in Railway/Vercel dashboards only. `.env` is gitignored — never commit real values.

### Web / Edge proxy (Vercel)

| Variable                | Description                                                          |
| ----------------------- | -------------------------------------------------------------------- |
| `REVIEWERMATCH_API_URL` | Base URL of the Railway API (no trailing slash)                      |
| `REVIEWERMATCH_API_KEY` | API key forwarded server-side by the edge function — never reaches the browser |

---

## Repository Structure

```
reviewermatch-api/
├── app/
│   ├── api/
│   │   └── routes.py          # All FastAPI endpoints
│   ├── models/
│   │   ├── db_models.py       # SQLAlchemy Author model (incl. embedding_vector)
│   │   └── schemas.py         # Pydantic MatchRequest / MatchResponse / Health
│   ├── services/
│   │   ├── embedder.py        # sentence-transformers singleton
│   │   ├── matcher.py         # FAISS search → filters → rerank pipeline
│   │   └── ranker.py          # Weighted scorer (similarity + h-index + recency)
│   ├── utils/
│   ├── config.py              # Pydantic Settings (reads .env)
│   ├── database.py            # Async SQLAlchemy engine + session factory
│   └── main.py                # FastAPI app, lifespan: rebuild FAISS from Postgres
├── api/
│   └── match.js               # Vercel Edge proxy → Railway (hides API_KEY)
├── data/                      # Generated artifacts (gitignored)
├── docs/
│   └── screenshots/           # Hero.png, Results.png
├── scripts/
│   ├── build_index.py         # Embed authors → build FAISS index
│   ├── colab_notebook.ipynb   # Full GPU ingestion + embedding on Colab
│   ├── ingest_sample.py       # Sample ingestion from OpenAlex
│   ├── load_jsonl.py          # Load authors_raw.jsonl into local DB
│   ├── load_metadata.py       # Attach metadata to author rows
│   ├── seed_postgres.py       # Upload rows + vector bytes to Railway Postgres
│   └── test_match.py          # Manual /match smoke script
├── tests/                     # pytest test suite
├── web/
│   └── index.html             # Static frontend (pearly marble design)
├── Dockerfile
├── docker-compose.yml
├── Procfile
├── railway.json
├── vercel.json
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── pytest.ini
└── .env.example
```

---

## Deployment

### Railway (API)

Railway builds directly from the `Dockerfile`. The `all-MiniLM-L6-v2` model weights are pre-downloaded at image build time so cold starts stay well inside Railway's healthcheck window. The FAISS index is **not** baked into the image — it's rebuilt in RAM from Postgres on every boot, so data refreshes don't require a redeploy.

**After a major author data refresh:**

```bash
# 1. Re-run ingestion + embedding on Colab (T4 GPU)
#    → downloads reviewermatch_bundle.tar.gz

# 2. Seed Railway Postgres with the new rows + vectors
DATABASE_URL="postgresql+asyncpg://..." \
    python -m scripts.seed_postgres

# 3. Restart Railway — the next boot rebuilds FAISS from the fresh vectors
```

### Vercel (Web app + Edge proxy)

The static site at `web/` and the edge function at `api/match.js` are deployed together. Set two environment variables in the Vercel dashboard:

| Variable                 | Value                                   |
| ------------------------ | --------------------------------------- |
| `REVIEWERMATCH_API_URL`  | `https://<your-service>.up.railway.app` |
| `REVIEWERMATCH_API_KEY`  | The same value as Railway's `API_KEY`   |

---

## Tech Stack

| Layer             | Technology                                                             |
| ----------------- | ---------------------------------------------------------------------- |
| **API framework** | FastAPI 0.115, Pydantic v2, Python 3.11                                |
| **Database**      | PostgreSQL (Railway prod) / SQLite (local dev) via SQLAlchemy async    |
| **Embeddings**    | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim, ~90 MB)         |
| **Vector search** | FAISS `IndexFlatIP` — cosine via L2-normalised inner product           |
| **Reranker**      | Transparent weighted heuristic (similarity + h-index + recency)        |
| **Ingestion**     | OpenAlex public API via `httpx` + `tenacity` retries                   |
| **Container**     | Docker on Railway — CPU-only PyTorch, HF model pre-baked               |
| **Frontend**      | Static HTML + vanilla JS       |
| **Fonts**         | Inter Tight (sans) + Fraunces (italic accent), system Helvetica Neue   |
| **Edge proxy**    | Vercel Edge Function — hides API key, rate-limits free tier            |

---

## Pricing

| Tier      | Price        | Includes                                                                   |
| --------- | ------------ | -------------------------------------------------------------------------- |
| **Free**  | £0 / month   | 3 searches / day · Top 10 results · h-index & recent works                 |
| **Pro**   | £9 / month   | Unlimited searches · Top 50 results · CSV export · COI filters             |
| **Lab**   | £49 / month  | 5 seats · Shared workspace · Reviewer lists with COI · Priority support    |

Pro and Lab tiers are **launching soon** — join the waitlist from the pricing section of the web app.

---

## Roadmap

### Next up — ingestion v2 *(in progress)*

- [ ] Incremental OpenAlex sync — nightly delta pulls via `from_updated_date`, upsert into Postgres, re-embed only new/changed authors.
- [ ] Scheduled Colab / GitHub Actions workflow (GPU runner) that re-embeds diffs and pushes fresh vector bytes into Railway Postgres automatically.
- [ ] Per-author `last_embedded_at` tracking in Postgres for rolling backfills of stale rows.
- [ ] Domain expansion beyond ML / CS — opt-in packs for bio, physics, economics.
- [ ] Conflict-of-interest signals pulled during ingest: co-author graph, shared-institution history, advisor–advisee edges.

### Later

- [ ] Author-level recency decay and topic-drift detection.
- [ ] Dedup OpenAlex author IDs that refer to the same person.
- [ ] Cross-encoder re-ranker for the top-K shortlist (trained on labelled reviewer data).
- [ ] Batch-match endpoint — upload a CSV of abstracts, get ranked shortlists.
- [ ] CSV export of results (Pro tier).
- [ ] Lab workspaces with shared reviewer lists and COI filters.
- [ ] Public dataset card describing coverage, freshness, and biases of the index.
- [ ] Claude-powered natural-language explanations for each match (`ANTHROPIC_API_KEY`).

---

## Contributing

1. Fork the repo and create a feature branch
2. Run `pip install -r requirements-dev.txt` for dev tooling (pytest, ruff, mypy)
3. Run tests: `pytest tests/`
4. Open a pull request against `main`

Before opening a PR, please skim the [Security](#security) section below and make sure your changes don't commit real secrets, database dumps, or API keys.

---

## Security

ReviewerMatch handles API keys, a live Postgres connection, and optional third-party keys (OpenAlex, Anthropic). A few rules of the road:

- **Never commit real secrets.** `.env`, `.env.*` (except `.env.example`), `*.log`, `*.sqlite`, and generated data files are gitignored. Real values for `API_KEY`, `DATABASE_URL`, `OPENALEX_API_KEY`, and `ANTHROPIC_API_KEY` belong only in the Railway and Vercel dashboards.
- **The browser never sees the API key.** The Vercel edge function at `api/match.js` reads `REVIEWERMATCH_API_KEY` server-side and injects the `X-API-Key` header before forwarding to Railway. Do not call Railway directly from the frontend in production.
- **Placeholders vs real values.** Strings like `postgresql+asyncpg://user:pass@host:5432/railway` in the README and scripts are intentional placeholders. If you see a string of the same shape but with a real-looking host/password in a PR, block it.
- **Free-tier rate limits.** `FREE_TIER_DAILY_LIMIT` (default `3`) gates public search volume per IP. Treat this as defence-in-depth, not a replacement for API-key auth on `POST /api/v1/match`.
- **Reporting a vulnerability.** Please do **not** open a public GitHub issue for security problems. Instead, open a private report via [GitHub Security Advisories](https://github.com/samiurk70/reviewermatch-api/security/advisories/new) or email the maintainer directly. You'll get an acknowledgement within a few working days.
- **If a key does leak**, rotate it immediately in the relevant dashboard (Railway / Vercel / OpenAlex / Anthropic), then consider scrubbing history with `git filter-repo --replace-text` and force-pushing — treat any previously pushed secret as permanently compromised.

Enable GitHub's **secret scanning** and **push protection** (Settings → Code security) on your fork to catch accidental commits automatically.

---

## License

Released under the [MIT License](LICENSE) — free to use, modify, and distribute with attribution.

```
Copyright (c) 2026 Samiur Khan
```

Data sourced from OpenAlex is published under [CC0](https://creativecommons.org/publicdomain/zero/1.0/); please cite OpenAlex when redistributing derivatives of the index.

---

Built with public data from **[OpenAlex](https://openalex.org)**. Forked from [grantmatch-api](https://github.com/samiurk70/grantmatch-api); grant-specific ingestion and the ML reranker were removed. Not affiliated with any publisher or journal.

[reviewermatch.vercel.app](https://reviewermatch.vercel.app) · API on Railway
