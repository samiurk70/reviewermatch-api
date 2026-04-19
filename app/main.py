import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import get_settings
from app.database import AsyncSessionLocal, create_all_tables
from app.api.routes import router
from app.services.embedder import get_embedder
from app.services.matcher import get_matcher

logger = logging.getLogger(__name__)


async def _rebuild_faiss_from_db_vectors() -> None:
    """
    Reconstruct the FAISS index from pre-computed embedding_vector bytes stored in Postgres.

    This runs at startup when the index file is missing (e.g. Railway container restart).
    It does NOT run any ML inference — it just reads stored float32 bytes and packs them
    into a FAISS IndexIDMap.  On 5k authors this takes ~1 s; on 150k authors ~10 s.
    """
    settings = get_settings()
    index_path = Path(settings.faiss_index_path)
    if index_path.exists():
        return

    import faiss
    import numpy as np
    from sqlalchemy import select
    from app.models.db_models import Author

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Author.id, Author.embedding_vector)
            .where(Author.embedding_vector.isnot(None))
            .order_by(Author.id)
        )
        rows = result.all()

    if not rows:
        # Check if authors exist at all (no vectors → run seed_postgres.py)
        from sqlalchemy import func
        async with AsyncSessionLocal() as session:
            cr = await session.execute(
                select(func.count()).select_from(Author)
            )
            total = cr.scalar_one_or_none() or 0

        if total == 0:
            logger.warning(
                "DB is empty — no data loaded yet. "
                "Run the Colab pipeline then `scripts/seed_postgres.py`. See README."
            )
        else:
            logger.warning(
                "DB has %d authors but no embedding_vector bytes stored. "
                "Re-run `scripts/seed_postgres.py` to populate vectors.", total
            )
        return

    first_vec = np.frombuffer(rows[0][1], dtype=np.float32)
    dim = len(first_vec)

    logger.info(
        "Rebuilding FAISS index from %d DB vectors (dim=%d) — no ML needed …",
        len(rows), dim,
    )

    ids = np.array([r[0] for r in rows], dtype=np.int64)
    vectors = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])

    base_index = faiss.IndexFlatIP(dim)
    id_index = faiss.IndexIDMap(base_index)
    id_index.add_with_ids(vectors, ids)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(id_index, str(index_path))
    logger.info("FAISS index rebuilt: %d vectors → %s", id_index.ntotal, index_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_all_tables()
    get_embedder()
    await _rebuild_faiss_from_db_vectors()
    matcher = get_matcher()
    if matcher.index is not None:
        logger.info("FAISS vectors: %d", matcher.index.ntotal)
    else:
        p = Path(get_settings().faiss_index_path)
        logger.warning("FAISS index not loaded (missing %s).", p.resolve())
    logger.info("ReviewerMatch API ready.")
    yield


app = FastAPI(
    title="ReviewerMatch API",
    description="Semantic researcher matching from abstracts and research topics.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/api/v1/")


@app.get("/health", tags=["meta"])
async def health_simple():
    return {"status": "ok"}
