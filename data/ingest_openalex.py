"""
Ingest ML/CS/AI researchers from OpenAlex into JSONL (one record per line).

Discovers authors via recent works in target concepts (OpenAlex does not support
`concepts.id` on the /authors endpoint). Fetches each author's works with abstracts
to build profile_text for embedding (Phase 3 / Colab).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import AsyncIterator

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

OPENALEX_BASE = os.environ.get("OPENALEX_BASE_URL", "https://api.openalex.org").rstrip("/")
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "").strip()

# Concept IDs (OpenAlex / C-prefix form works in works filter)
TARGET_CONCEPTS = [
    "C154945302",  # Artificial Intelligence
    "C119857082",  # Machine Learning
    "C108583219",  # Deep Learning
    "C153180895",  # Pattern Recognition
    "C13280743",  # Natural Language Processing
    "C31972630",  # Computer Vision
]

MIN_WORKS = 3
MIN_RECENT_YEAR = 2022  # author must have a work in or after this year (from abstract set)
# Works search: include papers after this year to discover active authors
WORKS_PUBLISHED_AFTER = MIN_RECENT_YEAR - 2

TARGET_AUTHOR_COUNT = 150_000


def _short_id(openalex_url_or_id: str) -> str:
    return openalex_url_or_id.replace("https://openalex.org/", "")


def _load_written_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                oid = rec.get("openalex_id")
                if oid:
                    ids.add(oid)
            except json.JSONDecodeError:
                continue
    return ids


def _add_api_key(params: dict) -> dict:
    """Inject OPENALEX_API_KEY when set — unlocks higher rate limits."""
    if OPENALEX_API_KEY:
        params = dict(params)
        params["api_key"] = OPENALEX_API_KEY
    return params


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
async def fetch_page(client: httpx.AsyncClient, url: str, params: dict | None = None) -> dict:
    r = await client.get(url, params=_add_api_key(params or {}))
    r.raise_for_status()
    return r.json()


async def iterate_works_by_concept(
    client: httpx.AsyncClient,
    concept_id: str,
    per_page: int = 50,
) -> AsyncIterator[dict]:
    """Recent works in a concept with abstracts (cursor pagination)."""
    cursor: str | None = "*"
    while cursor:
        params = {
            "filter": (
                f"concepts.id:{concept_id},has_abstract:true,"
                f"publication_year:>{WORKS_PUBLISHED_AFTER}"
            ),
            "per-page": per_page,
            "cursor": cursor,
            "select": "id,authorships,publication_year",
        }
        data = await fetch_page(client, f"{OPENALEX_BASE}/works", params)
        for work in data.get("results", []):
            yield work
        cursor = data.get("meta", {}).get("next_cursor")
        if not data.get("results"):
            break


async def fetch_author(client: httpx.AsyncClient, author_openalex_url: str) -> dict:
    aid = _short_id(author_openalex_url)
    params = {
        "select": (
            "id,display_name,orcid,works_count,cited_by_count,summary_stats,"
            "last_known_institutions,x_concepts"
        ),
    }
    return await fetch_page(client, f"{OPENALEX_BASE}/authors/{aid}", params)


async def fetch_author_works(
    client: httpx.AsyncClient,
    author_openalex_url: str,
    max_works: int = 10,
) -> list[dict]:
    params = {
        "filter": f"authorships.author.id:{author_openalex_url},has_abstract:true",
        "per-page": max_works,
        "sort": "publication_year:desc",
        "select": "id,title,abstract_inverted_index,publication_year,primary_location",
    }
    data = await fetch_page(client, f"{OPENALEX_BASE}/works", params)
    return data.get("results", [])


def reconstruct_abstract(inverted_index: dict) -> str:
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idx_list in inverted_index.items():
        for idx in idx_list:
            positions.append((idx, word))
    positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in positions)


def build_author_profile(author: dict, works: list[dict]) -> dict:
    abstracts: list[str] = []
    for w in works:
        text = reconstruct_abstract(w.get("abstract_inverted_index") or {})
        if text and len(text) > 50:
            abstracts.append(text)

    profile_text = " ".join(abstracts[:10])

    if not profile_text.strip():
        profile_text = " ".join(w.get("title", "") for w in works if w.get("title"))

    lki = author.get("last_known_institutions") or []
    first_inst = lki[0] if lki and isinstance(lki[0], dict) else {}

    return {
        "openalex_id": _short_id(author["id"]),
        "display_name": author["display_name"],
        "orcid": author.get("orcid"),
        "works_count": author.get("works_count", 0),
        "cited_by_count": author.get("cited_by_count", 0),
        "h_index": (author.get("summary_stats") or {}).get("h_index", 0),
        "i10_index": (author.get("summary_stats") or {}).get("i10_index", 0),
        "two_year_mean_citedness": (author.get("summary_stats") or {}).get(
            "2yr_mean_citedness", 0.0
        ),
        "institution_name": first_inst.get("display_name"),
        "institution_country": first_inst.get("country_code"),
        "top_concepts": [c["display_name"] for c in (author.get("x_concepts") or [])[:5]],
        "recent_works": [
            {
                "id": _short_id(w["id"]),
                "title": w.get("title"),
                "year": w.get("publication_year"),
                "venue": ((w.get("primary_location") or {}).get("source") or {}).get(
                    "display_name"
                ),
            }
            for w in works
        ],
        "profile_text": profile_text,
    }


async def main(
    email: str,
    output_path: Path,
    target_count: int = TARGET_AUTHOR_COUNT,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids = _load_written_ids(output_path)
    if len(seen_ids) >= target_count:
        print(
            f"Already have {len(seen_ids)} authors in {output_path} "
            f"(target {target_count}). Nothing to do."
        )
        return 0

    written_this_run = 0
    if OPENALEX_API_KEY:
        print(f"Using OpenAlex API key: {OPENALEX_API_KEY[:6]}…")
    headers = {"User-Agent": f"ReviewerMatch (mailto:{email})"}
    limits = httpx.Limits(max_connections=10)
    timeout = httpx.Timeout(30.0)

    async with httpx.AsyncClient(headers=headers, limits=limits, timeout=timeout) as client:
        with output_path.open("a", encoding="utf-8") as f:
            n_already = len(seen_ids)
            pbar = tqdm(total=target_count, initial=n_already, desc="Authors")

            for concept_id in TARGET_CONCEPTS:
                if len(seen_ids) >= target_count:
                    break

                async for work in iterate_works_by_concept(client, concept_id):
                    if len(seen_ids) >= target_count:
                        break

                    for authorship in work.get("authorships") or []:
                        if len(seen_ids) >= target_count:
                            break

                        author_stub = authorship.get("author") or {}
                        aid = author_stub.get("id")
                        if not aid:
                            continue

                        short = _short_id(aid)
                        if short in seen_ids:
                            continue

                        try:
                            author = await fetch_author(client, aid)
                            if (author.get("works_count") or 0) <= MIN_WORKS:
                                continue

                            works_list = await fetch_author_works(client, aid)
                            if not works_list:
                                continue

                            max_year = max((w.get("publication_year") or 0) for w in works_list)
                            if max_year < MIN_RECENT_YEAR:
                                continue

                            first_inst = (author.get("last_known_institutions") or [])
                            if first_inst:
                                if not (first_inst[0] or {}).get("country_code"):
                                    continue
                            else:
                                continue

                            profile = build_author_profile(author, works_list)
                            if len(profile["profile_text"]) < 100:
                                continue

                            f.write(json.dumps(profile, ensure_ascii=False) + "\n")
                            f.flush()
                            seen_ids.add(short)
                            written_this_run += 1
                            pbar.update(1)
                        except Exception as exc:
                            print(f"Skipped {aid}: {exc}")

            pbar.close()

    print(
        f"Done. {len(seen_ids)} total authors in {output_path} "
        f"({written_this_run} new this run, target was {target_count})."
    )
    return written_this_run


def _cli() -> None:
    email = os.environ.get("OPENALEX_EMAIL", "").strip()
    if not email:
        raise SystemExit("Set OPENALEX_EMAIL environment variable (e.g. your@email.com)")

    if not OPENALEX_API_KEY:
        print(
            "Tip: set OPENALEX_API_KEY for higher rate limits "
            "(see https://openalex.org/pricing)."
        )

    out = Path(os.environ.get("OPENALEX_OUTPUT", "data/authors_raw.jsonl"))
    asyncio.run(main(email, out))


if __name__ == "__main__":
    _cli()
