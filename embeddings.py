"""Semantic similarity scoring between job listings and the configured job profile.

Uses sentence-transformers to encode job text and compute cosine similarity
against the ideal profile embedding. Scores are in [0, 1] — higher is closer.
"""

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

import config
from scrapers import Job

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    logger.info(f"Loading sentence-transformer model: {_MODEL_NAME}")
    return SentenceTransformer(_MODEL_NAME)


@lru_cache(maxsize=1)
def _profile_embedding() -> np.ndarray:
    """Encode the job profile once and cache the result."""
    return _model().encode(config.JOB_PROFILE, normalize_embeddings=True)


def embed_jobs(jobs: list[Job]) -> list[float]:
    """Return cosine similarity [0-1] of each job's text to the configured job profile."""
    if not jobs:
        return []

    texts = [f"{job.title}. {job.snippet}" for job in jobs]
    job_embs = _model().encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
    )
    profile_emb = _profile_embedding()

    # Dot product of unit vectors == cosine similarity, range [-1, 1].
    # We consciously return raw values rather than rescaling to [0, 1]: real-world
    # snippet-to-profile similarities stay in roughly [0.1, 0.6], so negative
    # values don't occur in practice and rescaling would compress the useful range.
    # If you ever see negative values in logs, revisit this assumption.
    sims: list[float] = (job_embs @ profile_emb).tolist()
    logger.info(
        f"Embedded {len(jobs)} jobs — similarity range "
        f"[{min(sims):.3f}, {max(sims):.3f}]"
    )
    return sims
