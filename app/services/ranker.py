"""
Combined score: semantic similarity + h-index + recency + citation velocity.

Weights are heuristic; replace with learned reranker when you have feedback data.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.models.db_models import Author

W_SIMILARITY = 0.60
W_HINDEX = 0.15
W_RECENCY = 0.15
W_VELOCITY = 0.10


def _normalise_h_index(h: int) -> float:
    if h <= 0:
        return 0.0
    return min(1.0, h / 80.0)


def _normalise_recency(author: Author) -> float:
    y = author.last_known_activity_year
    if y is None:
        works = author.recent_works or []
        if not works:
            return 0.0
        y = max((w.get("year") or 0) for w in works)
    if not y:
        return 0.0
    current_year = datetime.now(timezone.utc).year
    gap = current_year - int(y)
    return max(0.0, 1.0 - gap / 5.0)


def _normalise_velocity(author: Author) -> float:
    v = author.two_year_mean_citedness or 0.0
    return min(1.0, float(v) / 10.0)


def combined_score(author: Author, similarity: float) -> float:
    """Return a score in [0, 1]. Multiply by 100 for display."""
    sim = max(0.0, min(1.0, float(similarity)))
    return (
        W_SIMILARITY * sim
        + W_HINDEX * _normalise_h_index(author.h_index or 0)
        + W_RECENCY * _normalise_recency(author)
        + W_VELOCITY * _normalise_velocity(author)
    )
