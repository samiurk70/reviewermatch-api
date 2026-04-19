"""API endpoints for ReviewerMatch."""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.db_models import Author
from app.models.schemas import AuthorDetail, HealthResponse, MatchRequest, MatchResponse
from app.services.embedder import get_embedder
from app.services.matcher import match_authors

router = APIRouter()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    if api_key != get_settings().api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


async def _data_freshness(db: AsyncSession) -> str:
    result = await db.execute(select(func.max(Author.updated_at)))
    latest = result.scalar_one_or_none()
    if latest is None:
        return "no data"
    return latest.strftime("%Y-%m-%d")


@router.get("/")
async def root():
    return {
        "name": "ReviewerMatch API",
        "version": "0.1.0",
        "description": "Semantic researcher discovery from paper abstracts and topics",
        "docs": "/docs",
    }


@router.get("/health", response_model=HealthResponse)
async def health(db: AsyncSession = Depends(get_db)):
    settings = get_settings()

    count_result = await db.execute(select(func.count()).select_from(Author))
    authors_in_db = count_result.scalar_one_or_none() or 0

    freshness_result = await db.execute(select(func.max(Author.updated_at)))
    latest = freshness_result.scalar_one_or_none()
    last_ingestion = latest.strftime("%Y-%m-%d") if latest else None

    index_path = Path(settings.faiss_index_path)
    index_built = index_path.exists()
    faiss_vectors: int | None = None
    if index_built:
        try:
            import faiss

            idx = faiss.read_index(str(index_path.resolve()))
            faiss_vectors = idx.ntotal
        except Exception:
            faiss_vectors = None

    try:
        get_embedder()
        model_loaded = True
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        authors_in_db=authors_in_db,
        index_built=index_built,
        faiss_vectors=faiss_vectors,
        last_ingestion=last_ingestion,
    )


@router.post("/match", response_model=MatchResponse)
async def match(
    body: MatchRequest,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
) -> MatchResponse:
    t0 = time.perf_counter()
    top = body.top_n or get_settings().max_results
    top = min(top, get_settings().max_results)
    authors = await match_authors(body, top, db)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    freshness = await _data_freshness(db)
    q = body.query
    summary = q[:200] + ("…" if len(q) > 200 else "")

    return MatchResponse(
        query_summary=summary,
        total_matched=len(authors),
        results=authors,
        processing_time_ms=round(elapsed_ms, 1),
        data_freshness=freshness,
    )


@router.get("/author/{author_id}", response_model=AuthorDetail)
async def get_author(
    author_id: int,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
) -> AuthorDetail:
    result = await db.execute(select(Author).where(Author.id == author_id))
    author = result.scalar_one_or_none()
    if author is None:
        raise HTTPException(status_code=404, detail=f"Author {author_id} not found")

    return AuthorDetail(
        author_id=author.id,
        openalex_id=author.openalex_id,
        display_name=author.display_name,
        institution_name=author.institution_name,
        h_index=author.h_index,
        works_count=author.works_count,
        profile_text=author.profile_text,
        last_known_activity_year=author.last_known_activity_year,
    )
