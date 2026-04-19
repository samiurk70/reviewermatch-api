"""
Build a FAISS IndexFlatIP from Author.profile_text embeddings.

Usage:
    python -m scripts.build_index
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import faiss
import numpy as np
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import AsyncSessionLocal, create_all_tables
from app.models.db_models import Author
from app.services.embedder import get_embedder

logger = logging.getLogger(__name__)

BATCH_SIZE = 128


def _text_for_row(display_name: str, profile_text: str | None) -> str:
    body = (profile_text or "")[:800]
    return f"{display_name} {body}".strip()


def _pack_vector(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


async def build_index(db_session: AsyncSession) -> int:
    settings = get_settings()
    index_path = Path(settings.faiss_index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    embedder = get_embedder()
    sample = embedder.encode(["dimension probe"])
    dim = sample.shape[1]

    base_index = faiss.IndexFlatIP(dim)
    id_index = faiss.IndexIDMap(base_index)

    count_result = await db_session.execute(
        select(func.count()).select_from(Author).where(Author.profile_text.isnot(None))
    )
    total = count_result.scalar_one() or 0
    if total == 0:
        logger.warning("No authors with profile_text — index not built.")
        return 0

    logger.info("Indexing %d authors (batch size=%d).", total, BATCH_SIZE)

    indexed = 0
    offset = 0
    while offset < total:
        result = await db_session.execute(
            select(Author.id, Author.display_name, Author.profile_text)
            .where(Author.profile_text.isnot(None))
            .order_by(Author.id)
            .limit(BATCH_SIZE)
            .offset(offset)
        )
        rows = result.all()
        if not rows:
            break

        texts = [_text_for_row(name, prof) for _, name, prof in rows]
        vectors = embedder.encode(texts).astype(np.float32)
        faiss.normalize_L2(vectors)

        ids = np.array([r[0] for r in rows], dtype=np.int64)
        id_index.add_with_ids(vectors, ids)

        for row, vec in zip(rows, vectors):
            aid, _, _ = row
            await db_session.execute(
                update(Author)
                .where(Author.id == aid)
                .values(embedding_vector=_pack_vector(vec))
            )

        indexed += len(rows)
        offset += BATCH_SIZE
        await db_session.commit()
        logger.info("Indexed %d / %d authors.", indexed, total)

    faiss.write_index(id_index, str(index_path))
    logger.info("Wrote FAISS index to %s (%d vectors).", index_path, id_index.ntotal)
    return id_index.ntotal


async def _amain() -> int:
    logging.basicConfig(level=logging.INFO)
    await create_all_tables()
    async with AsyncSessionLocal() as session:
        await build_index(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))
