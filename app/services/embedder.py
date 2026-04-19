"""Sentence-transformer singleton — loaded once at startup, reused per request."""
from __future__ import annotations

import logging
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


class SentenceEmbedder:
    """Wraps a SentenceTransformer and exposes encode / encode_single."""

    def __init__(self) -> None:
        settings = get_settings()
        logger.info("Loading embedding model: %s", settings.embedding_model)
        t0 = time.perf_counter()
        self._model = SentenceTransformer(settings.embedding_model)
        elapsed = time.perf_counter() - t0
        logger.info("Embedding model loaded in %.2fs.", elapsed)

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a batch of texts.

        Returns a float32 matrix of shape (n, 384), L2-normalised so that
        inner-product == cosine similarity.
        """
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode one text. Returns shape (384,)."""
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# Module-level singleton — initialised lazily, replaced at startup
# ---------------------------------------------------------------------------

_embedder: SentenceEmbedder | None = None


def get_embedder() -> SentenceEmbedder:
    """Return the module-level SentenceEmbedder singleton, creating it if needed."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceEmbedder()
    return _embedder
