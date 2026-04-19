# ReviewerMatch

ML-powered researcher discovery. Paste a paper abstract or topic and get ranked author matches for reviewers, collaborators, and citations.

Status: private beta.

## Dev

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set OPENALEX_EMAIL and (optionally) OPENALEX_API_KEY
```

## Data pipeline — get authors into the database

Generated files (`data/*.faiss`, `data/authors_raw.jsonl`, `data/authors_meta.json`) are **gitignored** and must be produced offline. There are two paths:

### Path A — Local (no GPU needed, ~500 authors for smoke test)

```bash
# 1. Ingest author profiles from OpenAlex
INGEST_SAMPLE_SIZE=500 python -m scripts.ingest_sample   # or: python -m data.ingest_openalex for full 150k

# 2. Load into local DB (auto-assigns IDs)
python -m scripts.load_jsonl                             # reads data/authors_raw.jsonl

# 3. Build FAISS index from DB
python -m scripts.build_index

# 4. Start the API
uvicorn app.main:app --reload
```

### Path B — Google Colab (recommended for full 150k authors with GPU embeddings)

Open `scripts/colab_notebook.ipynb` in Google Colab, run the full pipeline, download
`reviewermatch_bundle.tar.gz`, extract into `data/`, then:

```bash
python -m scripts.load_metadata    # loads data/authors_meta.json into DB
# data/authors.faiss is already in place from the bundle
uvicorn app.main:app --reload
```

### Railway deployment notes

1. Set these env vars in the Railway dashboard → Variables:
   - `DATABASE_URL` — Railway PostgreSQL URL (auto-provided if you add a Postgres plugin)
   - `API_KEY` — must match `REVIEWERMATCH_API_KEY` on Vercel
   - `OPENALEX_EMAIL` — your email for OpenAlex polite pool
   - `OPENALEX_API_KEY` — your OpenAlex premium API key (optional, for higher rate limits)

2. Populate the Railway DB via the Railway shell or by pointing `DATABASE_URL` at Railway Postgres locally:
   ```bash
   DATABASE_URL=postgresql+asyncpg://... python -m scripts.load_jsonl
   DATABASE_URL=postgresql+asyncpg://... python -m scripts.build_index
   ```

3. FAISS index: on first startup the API auto-rebuilds the FAISS index from the DB
   if the index file is missing. For Railway with no persistent disk this happens every
   restart — acceptable for small corpora. Add a Railway Volume mounted at `/app/data`
   to persist the index across restarts.

## API

- `GET /api/v1/` — service info
- `GET /api/v1/health` — DB + index status (author count, last ingestion, FAISS vectors)
- `POST /api/v1/match` — body: `{ "query": "...", "top_n": 10 }` (`abstract` also accepted; header `X-API-Key`)
- `GET /api/v1/author/{id}` — author detail

Forked from [grantmatch-api](https://github.com/samiurk70/grantmatch-api); grant-specific ingestion and ML reranker removed.
