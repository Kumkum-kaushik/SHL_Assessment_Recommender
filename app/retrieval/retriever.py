"""
FAISS-based retrieval for SHL assessments.

Index is built offline by scripts/build_index.py and loaded once at startup.
Uses IndexFlatIP (inner product) with L2-normalised embeddings for cosine similarity.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from app.retrieval.embedder import get_embedder

logger = logging.getLogger(__name__)

VECTORSTORE_DIR = Path(__file__).parent.parent.parent / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "metadata.json"


class FAISSRetriever:
    """Load a pre-built FAISS index and serve similarity queries."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.metadata: list[dict] = []
        self.embedder = get_embedder()
        self._load()

    def _load(self):
        """Load FAISS index and metadata from disk."""
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            logger.error(
                "Vector store not found at %s. Run: python scripts/build_index.py",
                VECTORSTORE_DIR,
            )
            return

        try:
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(METADATA_PATH, encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info(
                "FAISS index loaded: %d vectors, dimension %d",
                self.index.ntotal,
                self.index.d,
            )
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc)
            self.index = None
            self.metadata = []

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.metadata) > 0

    def search(self, query: str, k: int = 8) -> list[dict]:
        """
        Embed the query and return the top-k most similar assessments.
        Returns empty list if the index is not loaded.
        """
        if not self.is_ready:
            logger.warning("Retriever not ready — index missing.")
            return []

        query_vec = self.embedder.embed_query(query)  # shape (1, dim)

        # Ensure float32 and correct shape
        query_vec = query_vec.astype(np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        actual_k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, actual_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            item = self.metadata[idx].copy()
            item["_score"] = float(dist)
            results.append(item)

        return results

    def search_by_names(self, names: list[str], k_each: int = 3) -> list[dict]:
        """Retrieve assessments matching a list of specific names (for comparisons)."""
        seen_names = set()
        results = []
        for name in names:
            hits = self.search(name, k=k_each)
            for hit in hits:
                if hit["name"] not in seen_names:
                    seen_names.add(hit["name"])
                    results.append(hit)
        return results

    def build_retrieval_query(self, messages: list) -> str:
        """
        Construct a rich retrieval query from conversation history.
        Concatenates all user messages to capture the full context.
        """
        user_texts = [
            m.content for m in messages if m.role == "user"
        ]
        return " ".join(user_texts[-3:])  # Use last 3 user turns max


# Module-level singleton — loaded once when the app starts
_retriever: Optional[FAISSRetriever] = None


def get_retriever() -> FAISSRetriever:
    """Return the module-level retriever, initialising it on first call."""
    global _retriever
    if _retriever is None:
        _retriever = FAISSRetriever()
    return _retriever
