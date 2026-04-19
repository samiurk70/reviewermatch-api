# ReviewerMatch

ML-powered researcher discovery. Paste a paper abstract or topic and get ranked author matches for reviewers, collaborators, and citations.

Status: private beta.

## Dev

```bash
pip install -r requirements.txt
cp .env.example .env
# Set OPENALEX_EMAIL in .env for OpenAlex polite pool
python -m scripts.ingest_sample              # smoke: 500 authors (INGEST_SAMPLE_SIZE to change)
# Full corpus + GPU embeddings: open scripts/colab_notebook.ipynb in Google Colab (see notebook intro).
# Local full ingest (slow): OPENALEX_EMAIL=... python -m data.ingest_openalex
uvicorn app.main:app --reload
```

Forked from [grantmatch-api](https://github.com/samiurk70/grantmatch-api); grant-specific ingestion and ML reranker removed.

## API

- `GET /api/v1/` — service info
- `GET /api/v1/health` — DB + index status
- `POST /api/v1/match` — body: `{ "abstract": "...", "top_n": 10 }` (header `X-API-Key`)
- `GET /api/v1/author/{id}` — author detail
# reviewermatch-api
