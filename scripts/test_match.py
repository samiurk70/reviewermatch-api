"""Smoke-test POST /api/v1/match (see 04_API_ENDPOINTS.md)."""
from __future__ import annotations

import asyncio
import os

import httpx


async def main() -> None:
    base = os.environ.get("API_BASE", "http://127.0.0.1:8000")
    key = os.environ.get("API_KEY", "changeme")
    async with httpx.AsyncClient(base_url=base, timeout=120.0) as client:
        r = await client.post(
            "/api/v1/match",
            json={
                "query": (
                    "We study spiking neural networks for energy-efficient "
                    "anomaly detection on neuromorphic hardware."
                ),
                "top_n": 10,
            },
            headers={"X-API-Key": key},
        )
        r.raise_for_status()
        data = r.json()
        print(
            f"{data['total_matched']} matches in {data['processing_time_ms']} ms "
            f"(freshness: {data.get('data_freshness')})\n"
        )
        for m in data["results"]:
            inst = m.get("institution_name") or ""
            print(
                f"  {m['score']:5.1f}  sim={m['similarity']:.3f}  "
                f"h={m['h_index']:3d}  {m['display_name'][:40]:40s}  {inst[:40]}"
            )


if __name__ == "__main__":
    asyncio.run(main())
