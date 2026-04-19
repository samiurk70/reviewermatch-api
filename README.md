# ReviewerMatch

ML-powered researcher discovery. Paste a paper abstract or topic and get ranked author matches for reviewers, collaborators, and citations.

Status: private beta.

## Dev

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set OPENALEX_EMAIL and (optionally) OPENALEX_API_KEY
```

## Data pipeline — get authors into Railway

**Railway free tier (1 GB RAM, 2 CPU) cannot run ML training or re-embedding.**
All embedding work happens on Colab (T4 GPU). Railway only reads pre-computed
vectors from Postgres and reconstructs the FAISS index at startup — no ML needed.

### Step 1 — Generate data on Colab (T4 GPU)

Open `scripts/colab_notebook.ipynb` in Google Colab.
Run all cells. At the end download `reviewermatch_bundle.tar.gz` and extract it:

```bash
tar -xzf reviewermatch_bundle.tar.gz -C data/
# produces: data/authors_meta.json  data/authors.faiss
```

For a quick smoke test first (~500 authors, CPU-only, local):
```bash
INGEST_SAMPLE_SIZE=500 python -m scripts.ingest_sample   # → data/authors_raw.jsonl
python -m scripts.load_jsonl                              # → local SQLite DB
python -m scripts.build_index                            # → data/authors.faiss
```

### Step 2 — Seed Railway Postgres (run once, locally)

Copy your Railway Postgres connection string from the Railway dashboard
(Postgres service → Connect → "Postgres Connection URL").
Replace `postgresql://` with `postgresql+asyncpg://` if needed.

```bash
DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/railway" \
    python -m scripts.seed_postgres
```

This uploads every author row **plus** their embedding vector bytes into Postgres.
Railway will never need to run ML inference — it rebuilds FAISS from these bytes.

### Step 3 — Set Railway env vars

In Railway dashboard → your API service → Variables:

| Variable | Value |
|---|---|
| `DATABASE_URL` | your Railway Postgres URL (asyncpg form) |
| `API_KEY` | must match `REVIEWERMATCH_API_KEY` on Vercel |
| `OPENALEX_EMAIL` | your email (for polite OpenAlex access) |
| `OPENALEX_API_KEY` | `iy5PMNgpq7AjdJypQiASnr` (your OpenAlex key) |

### Step 4 — Redeploy Railway

Push to git or trigger a manual redeploy. On startup the API will:
1. Detect no `data/authors.faiss` file (container is stateless)
2. Read `embedding_vector` bytes from Postgres
3. Reconstruct the FAISS index in RAM (~seconds, no ML)
4. Serve requests

Check `/api/v1/health` — you should see `authors_in_db > 0` and `index_built: true`.

## API

- `GET /api/v1/` — service info
- `GET /api/v1/health` — DB + index status (author count, last ingestion, FAISS vectors)
- `POST /api/v1/match` — body: `{ "query": "...", "top_n": 10 }` (`abstract` also accepted; header `X-API-Key`)
- `GET /api/v1/author/{id}` — author detail

Forked from [grantmatch-api](https://github.com/samiurk70/grantmatch-api); grant-specific ingestion and ML reranker removed.
