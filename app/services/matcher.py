"""Embed query → FAISS retrieval → filters → weighted rank (04_API_ENDPOINTS)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.db_models import Author
from app.models.schemas import AuthorMatch, MatchRequest, WorkSummary
from app.services.embedder import SentenceEmbedder, get_embedder
from app.services.ranker import combined_score

logger = logging.getLogger(__name__)

_CANDIDATE_POOL = 150
_FALLBACK_POOL = 50


def _openalex_profile_url(openalex_id: str) -> str:
    oid = (openalex_id or "").strip()
    if oid.startswith("http"):
        return oid
    return f"https://openalex.org/{oid}"


def _matching_works(author: Author, limit: int = 2) -> list[WorkSummary]:
    works = author.recent_works or []
    if not isinstance(works, list):
        return []
    works_sorted = sorted(works, key=lambda w: (w.get("year") or 0), reverse=True)
    out: list[WorkSummary] = []
    for w in works_sorted[:limit]:
        if not isinstance(w, dict):
            continue
        wid = w.get("id")
        out.append(
            WorkSummary(
                id=str(wid) if wid is not None else "",
                title=w.get("title"),
                year=w.get("year"),
                venue=w.get("venue"),
            )
        )
    return out


def _concepts_list(author: Author) -> list[str]:
    raw = author.top_concepts or []
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for c in raw:
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, dict) and "display_name" in c:
            out.append(str(c["display_name"]))
    return out


def _passes_filters(author: Author, req: MatchRequest) -> bool:
    if req.min_h_index is not None and (author.h_index or 0) < req.min_h_index:
        return False
    if req.min_works is not None and (author.works_count or 0) < req.min_works:
        return False
    if req.countries:
        cc = (author.institution_country or "").strip().upper()
        allowed = {c.strip().upper() for c in req.countries}
        if cc not in allowed:
            return False
    if req.exclude_openalex_ids and author.openalex_id in req.exclude_openalex_ids:
        return False
    if req.require_active_since is not None:
        y = author.last_known_activity_year
        if y is None:
            works = author.recent_works or []
            y = max((w.get("year") or 0) for w in works) if works else 0
        if (y or 0) < req.require_active_since:
            return False
    return True


class AuthorMatcher:
    def __init__(self, embedder: SentenceEmbedder | None = None) -> None:
        self.embedder = embedder or get_embedder()
        self.index = None
        self._load_index()

    def _load_index(self) -> None:
        settings = get_settings()
        path = Path(settings.faiss_index_path)
        if not path.exists():
            logger.warning(
                "FAISS index not found at %s — semantic search disabled.",
                path,
            )
            return
        try:
            import faiss

            self.index = faiss.read_index(str(path))
            logger.info("FAISS index loaded: %d vectors.", self.index.ntotal)
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc)

    def _faiss_search(
        self, query_vec: np.ndarray
    ) -> tuple[list[int], dict[int, float]]:
        assert self.index is not None
        distances, ids = self.index.search(query_vec, _CANDIDATE_POOL)
        raw: dict[int, float] = {}
        order: list[int] = []
        for dist, aid in zip(distances[0], ids[0]):
            if aid == -1:
                continue
            i = int(aid)
            order.append(i)
            raw[i] = max(0.0, min(1.0, float(dist)))
        return order, raw

    async def _db_fallback(
        self, db_session: AsyncSession
    ) -> tuple[list[int], dict[int, float]]:
        result = await db_session.execute(
            select(Author).order_by(func.random()).limit(_FALLBACK_POOL)
        )
        authors = result.scalars().all()
        candidate_ids = [a.id for a in authors]
        scores = {a.id: 0.5 for a in authors}
        return candidate_ids, scores

    async def match(
        self,
        request: MatchRequest,
        db_session: AsyncSession,
    ) -> list[AuthorMatch]:
        top_n = request.top_n
        query_vec = (
            self.embedder.encode_single(request.query).reshape(1, -1).astype(np.float32)
        )

        if self.index is not None:
            candidate_order, semantic_raw = self._faiss_search(query_vec)
        else:
            candidate_order, semantic_raw = await self._db_fallback(db_session)

        if not candidate_order:
            return []

        result = await db_session.execute(
            select(Author).where(Author.id.in_(candidate_order))
        )
        by_id: dict[int, Author] = {a.id: a for a in result.scalars().all()}

        ranked: list[tuple[Author, float, float]] = []
        for aid in candidate_order:
            author = by_id.get(aid)
            if author is None:
                continue
            if not _passes_filters(author, request):
                continue
            sim = semantic_raw.get(aid, 0.5)
            score_01 = combined_score(author, sim)
            ranked.append((author, sim, score_01))

        ranked.sort(key=lambda x: x[2], reverse=True)
        ranked = ranked[:top_n]

        out: list[AuthorMatch] = []
        for author, sim, score_01 in ranked:
            concepts = _concepts_list(author)
            works = _matching_works(author, 2)
            out.append(
                AuthorMatch(
                    openalex_id=author.openalex_id,
                    display_name=author.display_name,
                    orcid=author.orcid,
                    institution_name=author.institution_name,
                    institution_country=author.institution_country,
                    h_index=author.h_index or 0,
                    i10_index=author.i10_index or 0,
                    works_count=author.works_count or 0,
                    cited_by_count=author.cited_by_count or 0,
                    top_concepts=concepts,
                    score=round(score_01 * 100, 1),
                    similarity=round(sim, 4),
                    matching_works=works,
                    profile_url=_openalex_profile_url(author.openalex_id),
                )
            )
        return out


_matcher: AuthorMatcher | None = None


def get_matcher() -> AuthorMatcher:
    global _matcher
    if _matcher is None:
        _matcher = AuthorMatcher()
    return _matcher


async def match_authors(
    request: MatchRequest, top_k: int, db_session: AsyncSession
) -> list[AuthorMatch]:
    req = request.model_copy(update={"top_n": top_k})
    return await get_matcher().match(req, db_session)
