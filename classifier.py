"""Prototype-based AI-native classifier using sentence embeddings.

Classifies whether a job is genuinely AI-native (focuses on AI tooling/adoption
for dev teams) versus generic full-stack with AI sprinkles (builds software that
happens to use AI APIs).

No labelled training data needed: positive and negative class centroids are
computed from a small set of hand-crafted prototype sentences.
"""

import logging
from functools import lru_cache

import numpy as np

from scrapers import Job

logger = logging.getLogger(__name__)

# Positive class: genuine AI-native / AI-enablement roles
_AI_NATIVE_PROTOTYPES = [
    "AI enablement engineer helping development teams adopt GitHub Copilot and Claude Code",
    "Developer experience engineer focused on AI tooling adoption and measuring developer productivity",
    "AI adoption specialist coaching software engineering teams on integrating AI into their workflows",
    "GenAI champion embedding AI practices and tooling into the software development lifecycle",
    "Platform engineer implementing MCP servers and AI coding assistants for enterprise dev teams",
    "AI way-of-working specialist defining standards for responsible AI tool use across engineering",
    "Internal AI tooling lead helping engineers use Cursor, Copilot, and autonomous coding agents effectively",
]

# Negative class: generic developer roles with superficial AI involvement
_AI_SPRINKLES_PROTOTYPES = [
    "Full-stack developer building React and Node.js web applications with OpenAI API integrations",
    "Software engineer adding AI chatbot features to existing SaaS products via third-party LLM APIs",
    "Backend engineer implementing machine learning model inference endpoints for the product",
    "Python developer building data pipelines and ML model training infrastructure with TensorFlow",
    "Senior developer working on AI-powered product features using pre-built cloud AI services",
    "MLOps engineer managing model deployment pipelines and monitoring for production ML systems",
    "Data scientist building predictive models and analytics dashboards for business intelligence",
]


@lru_cache(maxsize=1)
def _centroids() -> tuple[np.ndarray, np.ndarray]:
    """Compute and cache the prototype centroids for both classes."""
    from embeddings import _model
    model = _model()
    native_embs = model.encode(_AI_NATIVE_PROTOTYPES, normalize_embeddings=True)
    sprinkles_embs = model.encode(_AI_SPRINKLES_PROTOTYPES, normalize_embeddings=True)

    native_centroid = native_embs.mean(axis=0)
    sprinkles_centroid = sprinkles_embs.mean(axis=0)

    # Re-normalise so dot product == cosine similarity when encoding jobs
    native_centroid /= np.linalg.norm(native_centroid)
    sprinkles_centroid /= np.linalg.norm(sprinkles_centroid)
    return native_centroid, sprinkles_centroid


def classify_ai_native(jobs: list[Job]) -> list[float]:
    """Return an AI-native score in [0, 1] for each job.

    Uses temperature-scaled softmax over cosine similarities to the two class
    centroids. Values near 1.0 = likely genuine AI-native; near 0.0 = AI sprinkles.

    Note: this is a confidence-shaped score, NOT a calibrated probability.
    It is not calibrated against labelled data, so treat it as a ranking signal
    rather than a true posterior. The field is named ai_native_score (not prob)
    to make this explicit. Run scripts/evaluate_classifier.py with hand labels
    to measure how useful the score actually is.
    """
    if not jobs:
        return []

    from embeddings import _model
    model = _model()
    native_centroid, sprinkles_centroid = _centroids()

    texts = [f"{job.title}. {job.snippet}" for job in jobs]
    embs = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)

    native_sims = embs @ native_centroid      # shape (N,)
    sprinkles_sims = embs @ sprinkles_centroid  # shape (N,)

    # Temperature-scaled softmax; lower temperature = sharper separation
    temp = 0.15
    native_exp = np.exp(native_sims / temp)
    sprinkles_exp = np.exp(sprinkles_sims / temp)
    scores: list[float] = (native_exp / (native_exp + sprinkles_exp)).tolist()

    n_native = sum(1 for s in scores if s >= 0.5)
    logger.info(f"Classified {len(jobs)} jobs — {n_native} likely AI-native")
    return scores
