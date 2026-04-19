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


async def _auto_build_faiss() -> None:
    """Build FAISS index from DB-stored vectors if the index file is missing but the DB has authors."""
    settings = get_settings()
    index_path = Path(settings.faiss_index_path)
    if index_path.exists():
        return

    from sqlalchemy import func, select
    from app.models.db_models import Author

    async with AsyncSessionLocal() as session:
        count_result = await session.execute(
            select(func.count()).select_from(Author).where(Author.profile_text.isnot(None))
        )
        total = count_result.scalar_one() or 0

    if total == 0:
        logger.warning(
            "FAISS index missing and DB is empty — no data loaded yet. "
            "Run the ingest pipeline: see README."
        )
        return

    logger.info(
        "FAISS index not found at %s but DB has %d authors — building index now …",
        index_path,
        total,
    )
    try:
        from scripts.build_index import build_index as _build
        async with AsyncSessionLocal() as session:
            n = await _build(session)
        logger.info("Auto-built FAISS index: %d vectors written to %s.", n, index_path)
    except Exception as exc:
        logger.error("Auto-build of FAISS index failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_all_tables()
    get_embedder()
    await _auto_build_faiss()
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
