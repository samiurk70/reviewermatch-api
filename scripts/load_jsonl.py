"""
Load data/authors_raw.jsonl directly into the database.

This is the non-Colab alternative to scripts/load_metadata.py.
It assigns auto-increment IDs; run scripts/build_index.py afterwards
to build the FAISS index using those IDs.

Usage:
    python -m scripts.load_jsonl                        # default: data/authors_raw.jsonl
    python -m scripts.load_jsonl data/authors_raw.jsonl

Env:
    DATABASE_URL — defaults to sqlite+aiosqlite:///data/reviewermatch.db
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal, create_all_tables
from app.models.db_models import Author


def _activity_year(recent_works: list) -> int | None:
    years = [w.get("year") for w in (recent_works or []) if w.get("year")]
    return max(years) if years else None


def _load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  Skipping line {lineno}: {exc}")
                continue
            oid = rec.get("openalex_id")
            if not oid:
                print(f"  Skipping line {lineno}: missing openalex_id")
                continue
            if oid in seen_ids:
                continue
            seen_ids.add(oid)
            records.append(rec)
    return records


async def load(path: Path, session: AsyncSession) -> int:
    records = _load_records(path)
    if not records:
        raise SystemExit(f"No valid records found in {path}")

    print(f"Loading {len(records)} unique authors from {path} …")

    for rec in records:
        recent = rec.get("recent_works") or []
        author = Author(
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


async def main(jsonl_path: Path) -> None:
    await create_all_tables()

    async with AsyncSessionLocal() as session:
        count_result = await session.execute(
            __import__("sqlalchemy", fromlist=["select"])
            .select(__import__("sqlalchemy", fromlist=["func"]).func.count())
            .select_from(Author)
        )
        existing = count_result.scalar_one_or_none() or 0

    if existing:
        answer = input(
            f"Database already has {existing} authors. "
            "Replace all? [y/N] "
        ).strip().lower()
        if answer != "y":
            raise SystemExit("Aborted.")
        async with AsyncSessionLocal() as session:
            await session.execute(delete(Author))
            await session.commit()
        print("  Cleared existing authors.")

    async with AsyncSessionLocal() as session:
        n = await load(jsonl_path, session)

    print(f"Done. {n} authors loaded. Run `python -m scripts.build_index` next.")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/authors_raw.jsonl")
    if not path.exists():
        raise SystemExit(
            f"File not found: {path}\n"
            "Run `python -m data.ingest_openalex` first to generate it."
        )
    asyncio.run(main(path))
