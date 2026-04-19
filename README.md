# ReviewerMatch

Semantic researcher discovery. Paste a paper abstract or topic and get a
ranked shortlist of active ML / CS researchers matched by meaning — not
keywords — for peer review, collaboration, citations, or supervisor search.

Live: [reviewermatch.vercel.app](https://reviewermatch.vercel.app) ·
Status: **private beta**.

![ReviewerMatch hero](docs/screenshots/Hero.png)

---

## Table of contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Screenshots](#screenshots)
- [Local development](#local-development)
- [Data pipeline — getting authors into Railway](#data-pipeline--getting-authors-into-railway)
- [API reference](#api-reference)
- [Deployment](#deployment)
- [Roadmap](#roadmap)
- [Credits](#credits)

---

## What it does

- Indexes active ML / CS researchers from [OpenAlex](https://openalex.org) into a shared
  384-dimensional semantic space using `sentence-transformers` (`all-MiniLM-L6-v2`).
- Embeds your query abstract with the same model and runs a FAISS cosine-similarity
  search across the full index.
- Reranks the shortlist by semantic similarity, **h-index**, and **recency**, and
  returns each match with their institution, citation stats, and the recent works
  that explain the score.

## Architecture

```
┌────────────────────┐      ┌──────────────────────────┐      ┌──────────────────┐
│  Vercel (static)   │      │   Railway (FastAPI)      │      │  Railway Postgres │
│  web/ + api/match  │ ───► │   app/ — embed + FAISS   │ ───► │  authors + vecs  │
│  (edge proxy)      │      │   rebuilds index in RAM  │      │                  │
└────────────────────┘      └──────────────────────────┘      └──────────────────┘
           ▲                             ▲
           │                             │ seeded once, offline
           │                             │
      end users                ┌──────────────────────┐
                               │  Colab (T4 GPU)       │
                               │  sentence-transformers│
                               │  → embedding bytes    │
                               └──────────────────────┘
```

All ML inference happens **offline on Colab** and seeds Railway Postgres with
pre-computed embedding vectors. Railway's free tier (1 GB RAM, 2 CPU) only has
to reconstruct the FAISS index from those bytes at startup — it never runs the
model.

## Screenshots

**Landing / query**

![Hero](docs/screenshots/Hero.png)

**Ranked results**

![Results](docs/screenshots/Results.png)

---

## Local development

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env — at minimum set OPENALEX_EMAIL (required for OpenAlex polite-pool).
# Leave OPENALEX_API_KEY empty unless you have a premium key.
```

Run the API locally against SQLite:

```bash
uvicorn app.main:app --reload
# → http://127.0.0.1:8000
# → http://127.0.0.1:8000/docs   (Swagger)
```

Tests:

```bash
pytest
```

The static frontend lives in `web/index.html` and talks to the FastAPI service.
In production the Vercel edge function at `api/match.js` proxies requests to
Railway so the API key is never exposed to the browser.

---

## Data pipeline — getting authors into Railway

> Railway's free tier cannot run ML training or re-embedding. All embedding
> work happens on Colab; Railway only rebuilds the FAISS index from
> pre-computed vectors stored in Postgres.

### Step 1 — Generate data on Colab (T4 GPU)

Open `scripts/colab_notebook.ipynb` in Google Colab. Run all cells. At the end
download `reviewermatch_bundle.tar.gz` and extract it:

```bash
tar -xzf reviewermatch_bundle.tar.gz -C data/
# produces: data/authors_meta.json  data/authors.faiss
```

For a quick smoke test first (~500 authors, CPU-only, local):

```bash
INGEST_SAMPLE_SIZE=500 python -m scripts.ingest_sample   # → data/authors_raw.jsonl
python -m scripts.load_jsonl                             # → local SQLite DB
python -m scripts.build_index                            # → data/authors.faiss
```

### Step 2 — Seed Railway Postgres (run once, locally)

Copy your Railway Postgres connection string from the Railway dashboard
(Postgres service → Connect → "Postgres Connection URL"). Replace
`postgresql://` with `postgresql+asyncpg://` if needed.

```bash
DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/railway" \
    python -m scripts.seed_postgres
```

This uploads every author row **plus** its embedding vector bytes into Postgres.
Railway will never need to run ML inference — it rebuilds FAISS from these bytes
on boot.

### Step 3 — Set Railway env vars

In the Railway dashboard → your API service → **Variables**:

| Variable            | Value                                                |
| ------------------- | ---------------------------------------------------- |
| `DATABASE_URL`      | your Railway Postgres URL (asyncpg form)             |
| `API_KEY`           | must match `REVIEWERMATCH_API_KEY` on Vercel         |
| `OPENALEX_EMAIL`    | your email (for polite-pool access)                  |
| `OPENALEX_API_KEY`  | *(optional)* your premium OpenAlex key — **never commit this** |

> ⚠️ Treat `API_KEY`, `DATABASE_URL`, and `OPENALEX_API_KEY` as secrets. They
> live in Railway/Vercel dashboards only. `.env` is gitignored; do not paste
> real values into this README, commit messages, or issues.

### Step 4 — Redeploy Railway

Push to git or trigger a manual redeploy. On startup the API will:

1. Detect no `data/authors.faiss` file (the container is stateless).
2. Read `embedding_vector` bytes from Postgres.
3. Reconstruct the FAISS index in RAM (~seconds, no ML).
4. Serve requests.

Check `/api/v1/health` — you should see `authors_in_db > 0` and
`index_built: true`.

---

## API reference

Base path: `/api/v1`. Authenticated endpoints expect `X-API-Key: <API_KEY>`.

| Method | Path                        | Description                                                              |
| ------ | --------------------------- | ------------------------------------------------------------------------ |
| `GET`  | `/api/v1/`                  | Service info (name, version).                                            |
| `GET`  | `/api/v1/health`            | DB + index status: author count, last ingestion, FAISS vector count.     |
| `POST` | `/api/v1/match`             | Body: `{ "query": "...", "top_n": 10, "min_h_index": 0 }`. `abstract` is also accepted as an alias for `query`. |
| `GET`  | `/api/v1/author/{id}`       | Full author detail including recent works.                                |

Example:

```bash
curl -X POST https://<your-railway-app>/api/v1/match \
     -H 'Content-Type: application/json' \
     -H "X-API-Key: $API_KEY" \
     -d '{"query":"spiking neural networks for anomaly detection","top_n":10}'
```

---

## Deployment

- **Railway** — FastAPI service (`app/`) + Postgres. See `railway.json`, `Procfile`, `Dockerfile`.
- **Vercel** — static frontend in `web/` and an edge proxy at `api/match.js`.
  Set `REVIEWERMATCH_API_URL` and `REVIEWERMATCH_API_KEY` in the Vercel project
  so the browser never sees your Railway API key.

---

## Roadmap

Ingestion is the next big area of work — the current index is a snapshot from
Colab and needs to become self-refreshing.

**Next up — ingestion v2 (in progress)**

- [ ] Incremental OpenAlex sync: nightly delta pulls via `from_updated_date`,
      upsert into Postgres, re-embed only new/changed authors.
- [ ] Scheduled Colab / GitHub Actions workflow (with GPU runner) that re-embeds
      diffs and pushes fresh vector bytes into Railway Postgres automatically.
- [ ] Per-author "last embedded at" tracking in Postgres so we can backfill
      stale rows on a rolling basis.
- [ ] Broader coverage beyond ML / CS — opt-in domain packs (bio, physics, econ).
- [ ] Conflict-of-interest signals during ingest: co-author graph, shared
      institution history, advisor–advisee edges.

**Later**

- [ ] Author-level recency decay and topic-drift detection.
- [ ] Deduping OpenAlex author IDs that refer to the same person.
- [ ] Batch-match endpoint (upload a CSV of abstracts, get ranked shortlists).
- [ ] CSV export of results (Pro tier).
- [ ] Lab workspaces with shared reviewer lists and COI filters.
- [ ] Public dataset card describing coverage, freshness, and biases of the index.

Tracking is rough — open an issue or ping me if you want a specific item
prioritised.

---

## Credits

Built with public data from [OpenAlex](https://openalex.org). Forked from
[grantmatch-api](https://github.com/samiurk70/grantmatch-api); grant-specific
ingestion and the ML reranker were removed. Not affiliated with any publisher
or journal.
