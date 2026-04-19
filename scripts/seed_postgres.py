"""
One-shot script: seed a remote Postgres (Railway) from a local Colab bundle.

Run this locally after the Colab notebook finishes and you have:
  data/authors_meta.json   — metadata records with faiss_row_id
  data/authors.faiss       — FAISS IndexIDMap with 384-dim float32 vectors

It loads every author row AND stores the embedding vector in the DB so that
Railway can rebuild the FAISS index at startup from Postgres alone — no ML
inference needed on the server.

Usage:
    # Point DATABASE_URL at Railway Postgres (copy from Railway dashboard):
    DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/railway" \\
        python -m scripts.seed_postgres

    # Or use a local bundle path override:
    DATABASE_URL="..." python -m scripts.seed_postgres \\
        --meta data/authors_meta.json --faiss data/authors.faiss

Env:
    DATABASE_URL  — target database (required; Railway Postgres URL)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal, create_all_tables
from app.models.db_models import Author


# ── vector extraction ─────────────────────────────────────────────────────────

def load_vectors_from_faiss(index_path: Path) -> dict[int, bytes]:
    """
    Return {author_id -> packed float32 bytes} for every vector in the index.

    Works with IndexIDMap wrapping IndexFlatIP (the shape built by build_index.py
    and the Colab notebook).
    """
    idx = faiss.read_index(str(index_path))
    # unwrap IndexIDMap → underlying flat index
    if hasattr(idx, "index"):
        flat = idx.index
        id_map: np.ndarray = faiss.vector_to_array(idx.id_map)  # external IDs
    else:
        flat = idx
        id_map = np.arange(idx.ntotal, dtype=np.int64)

    total = flat.ntotal
    if total == 0:
        return {}

    all_vecs = np.empty((total, flat.d), dtype=np.float32)
    for i in range(total):
        flat.reconstruct(i, all_vecs[i])

    return {int(id_map[i]): all_vecs[i].tobytes() for i in range(total)}


# ── db load ───────────────────────────────────────────────────────────────────

def _activity_year(recent_works: list) -> int | None:
    years = [w.get("year") for w in (recent_works or []) if w.get("year")]
    return max(years) if years else None


async def seed(
    meta_path: Path,
    faiss_path: Path,
    session: AsyncSession,
    batch_size: int = 500,
) -> int:
    with meta_path.open(encoding="utf-8") as f:
        records: list[dict] = json.load(f)

    # deduplicate by openalex_id (keep lowest faiss_row_id as load_metadata.py does)
    seen: dict[str, dict] = {}
    for rec in sorted(records, key=lambda r: r["faiss_row_id"]):
        oid = rec["openalex_id"]
        if oid not in seen:
            seen[oid] = rec
    records = list(seen.values())
    print(f"Unique authors after dedup: {len(records)}")

    print(f"Loading vectors from {faiss_path} …")
    vectors = load_vectors_from_faiss(faiss_path)
    print(f"  {len(vectors)} vectors loaded (dim={next(iter(vectors.values())).__len__() // 4 if vectors else '?'})")

    loaded = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        for rec in batch:
            rid = rec["faiss_row_id"]
            recent = rec.get("recent_works") or []
            author = Author(
                id=rid,
                openalex_id=rec["openalex_id"],
                display_name=rec["display_name"],
                orcid=rec.get("orcid"),
                profile_text=rec.get("profile_text"),
                institution_name=rec.get("institution_name"),
                institution_country=rec.get("institution_country"),
                h_index=rec.get("h_index"),
                works_count=rec.get("works_count"),
                cited_by_count=rec.get("cited_by_count"),
                i10_index=rec.get("i10_index"),
                two_year_mean_citedness=rec.get("two_year_mean_citedness"),
                top_concepts=rec.get("top_concepts") or [],
                recent_works=recent,
                last_known_activity_year=rec.get("last_known_activity_year")
                or _activity_year(recent),
                embedding_vector=vectors.get(rid),  # ← key: pre-computed vector
            )
            session.add(author)
        await session.commit()
        loaded += len(batch)
        pct = loaded / len(records) * 100
        missing_vecs = sum(1 for r in batch if r["faiss_row_id"] not in vectors)
        print(f"  {loaded}/{len(records)} ({pct:.0f}%)  missing_vectors_this_batch={missing_vecs}")

    return loaded


# ── entry point ───────────────────────────────────────────────────────────────

async def main(meta_path: Path, faiss_path: Path) -> None:
    await create_all_tables()

    async with AsyncSessionLocal() as session:
        count_result = await session.execute(
            select(func.count()).select_from(Author)
        )
        existing = count_result.scalar_one_or_none() or 0

    if existing:
        answer = input(
            f"\nRemote DB already has {existing} authors. Replace all? [y/N] "
        ).strip().lower()
        if answer != "y":
            raise SystemExit("Aborted.")
        print("Clearing existing authors …")
        async with AsyncSessionLocal() as session:
            await session.execute(delete(Author))
            await session.commit()

    print(f"\nSeeding from:\n  meta  : {meta_path}\n  faiss : {faiss_path}\n")
    async with AsyncSessionLocal() as session:
        n = await seed(meta_path, faiss_path, session)

    # verify
    async with AsyncSessionLocal() as session:
        chk = await session.execute(select(Author).where(Author.id == 1))
        if chk.scalar_one_or_none() is None:
            raise SystemExit("Verification failed: Author id=1 missing after load.")
        vec_result = await session.execute(
            select(func.count()).select_from(Author).where(Author.embedding_vector.isnot(None))
        )
        vec_count = vec_result.scalar_one_or_none() or 0

    print(f"\nDone. {n} authors loaded, {vec_count} with embedding vectors.")
    print("Railway will rebuild the FAISS index from these vectors at next startup.")
    print("No ML inference needed on Railway.")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", default="data/authors_meta.json", type=Path)
    p.add_argument("--faiss", default="data/authors.faiss", type=Path)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    missing = [p for p in (args.meta, args.faiss) if not p.exists()]
    if missing:
        print("Missing files:", *missing)
        print("\nRun the Colab notebook first to generate these files, then run this script.")
        sys.exit(1)
    asyncio.run(main(args.meta, args.faiss))
