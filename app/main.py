import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import get_settings
from app.database import create_all_tables
from app.api.routes import router
from app.services.embedder import get_embedder
from app.services.matcher import get_matcher

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_all_tables()
    get_embedder()
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
