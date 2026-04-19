"""
Load data/authors_meta.json into the database with explicit Author.id valuesthat match labels stored in data/authors.faiss (IndexIDMap, 1..N).

Run after unpacking the Colab bundle:

    python -m scripts.load_metadata

Requires .env / DATABASE_URL (SQLite or Postgres).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import AsyncSessionLocal, create_all_tables
from app.models.db_models import Author


def _activity_year(recent_works: list) -> int | None:
    years = [w.get("year") for w in (recent_works or []) if w.get("year")]
    return max(years) if years else None


async def load_from_json(path: Path, session: AsyncSession) -> int:
    with path.open(encoding="utf-8") as f:
        records = json.load(f)

    for rec in sorted(records, key=lambda r: r["faiss_row_id"]):
        aid = rec["faiss_row_id"]
        recent = rec.get("recent_works") or []
        author = Author(
            id=aid,
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
        )
        session.add(author)

    await session.commit()
    return len(records)


async def main() -> None:
    settings = get_settings()
    meta_path = Path("data/authors_meta.json")
    if not meta_path.exists():
        raise SystemExit(f"Missing {meta_path}")

    await create_all_tables()

    async with AsyncSessionLocal() as session:
        await session.execute(delete(Author))
        await session.commit()
        n = await load_from_json(meta_path, session)

    async with AsyncSessionLocal() as session:
        chk = await session.execute(select(Author).where(Author.id == 1))
        if chk.scalar_one_or_none() is None:
            raise SystemExit("Load verification failed: Author id=1 missing")

    db_hint = settings.database_url.split("@")[-1] if "@" in settings.database_url else settings.database_url
    print(f"Loaded {n} authors into {db_hint}")


if __name__ == "__main__":
    asyncio.run(main())
