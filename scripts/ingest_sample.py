"""
Smoke-test OpenAlex ingestion (~500 authors by default).

  export OPENALEX_EMAIL=you@example.com
  python -m scripts.ingest_sample

Output: data/authors_raw.jsonl (append + resume-safe).
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from data.ingest_openalex import main as ingest_main

SAMPLE_TARGET = 500


def main() -> None:
    email = os.environ.get("OPENALEX_EMAIL", "").strip()
    if not email:
        raise SystemExit("Set OPENALEX_EMAIL environment variable")

    out = Path("data/authors_raw.jsonl")
    count = int(os.environ.get("INGEST_SAMPLE_SIZE", str(SAMPLE_TARGET)))
    asyncio.run(ingest_main(email, out, target_count=count))
    print(f"Sample complete. Inspect: head -1 {out}")


if __name__ == "__main__":
    main()
