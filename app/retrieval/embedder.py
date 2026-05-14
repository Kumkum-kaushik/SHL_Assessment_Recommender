"""
Text embedding using sentence-transformers/all-MiniLM-L6-v2.

Produces L2-normalised embeddings suitable for cosine similarity
via FAISS IndexFlatIP (inner product == cosine sim for unit vectors).
"""

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Singleton-safe text embedder wrapping SentenceTransformer."""

    def __init__(self, model_name: str = MODEL_NAME):
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded. Dimension: %d", self.dimension)

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts.
        Returns float32 array of shape (N, dimension), L2-normalised.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,   # Unit norm → cosine sim via dot product
            show_progress_bar=len(texts) > 50,
            batch_size=32,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        Returns float32 array of shape (1, dimension).
        """
        return self.embed([query])


def build_assessment_text(assessment: dict) -> str:
    """
    Combine assessment fields into a single text chunk for embedding.
    Richer text → better retrieval accuracy.
    """
    parts = [
        assessment.get("name", ""),
        assessment.get("description", ""),
        f"Test type: {assessment.get('test_type', '')}",
        f"Duration: {assessment.get('duration', '')}",
    ]

    skills = assessment.get("skills_measured", [])
    if skills:
        parts.append(f"Measures: {', '.join(skills)}")

    suitable = assessment.get("suitable_for", [])
    if suitable:
        parts.append(f"Suitable for: {', '.join(suitable)}")

    if assessment.get("adaptive"):
        parts.append("Adaptive assessment")
    if assessment.get("remote_testing"):
        parts.append("Remote testing available")

    return " | ".join(filter(None, parts))


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return a cached Embedder instance (loaded once per process)."""
    return Embedder()
