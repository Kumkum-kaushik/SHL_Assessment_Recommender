"""
Offline script to scrape the SHL catalog (or load seed data)
and build the FAISS vector index.

Usage:
    python scripts/build_index.py             # Try scraping, fall back to seed
    python scripts/build_index.py --use-seed  # Always use seed data (fast, offline)
    python scripts/build_index.py --scrape    # Force live scraping

Outputs:
    vectorstore/index.faiss    — FAISS index (IndexFlatIP, L2-normalised)
    vectorstore/metadata.json  — Parallel array of assessment metadata
    data/assessments.json      — Final merged assessment list used for indexing
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Make sure the project root is on sys.path when running as a script
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import faiss

from app.retrieval.embedder import Embedder, build_assessment_text
from app.retrieval.scraper import load_or_scrape, load_seed_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = ROOT / "data"
VECTOR_DIR = ROOT / "vectorstore"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SHL FAISS index")
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--use-seed",
        action="store_true",
        help="Use curated seed data only (no network requests)",
    )
    group.add_argument(
        "--scrape",
        action="store_true",
        help="Force live scraping from SHL website",
    )
    return p.parse_args()


def build_index(assessments: list[dict]) -> None:
    """Embed assessments and write FAISS index + metadata to disk."""
    if not assessments:
        logger.error("No assessments to index. Aborting.")
        sys.exit(1)

    # ── Build text representations ──────────────────────────────────────
    logger.info("Building text representations for %d assessments...", len(assessments))
    texts = [build_assessment_text(a) for a in assessments]

    # ── Embed ────────────────────────────────────────────────────────────
    logger.info("Embedding with sentence-transformers/all-MiniLM-L6-v2...")
    embedder = Embedder()
    embeddings = embedder.embed(texts)           # shape (N, 384), float32, L2-normalised
    logger.info("Embedding shape: %s", embeddings.shape)

    # ── FAISS index ───────────────────────────────────────────────────────
    # IndexFlatIP = exact inner-product search
    # With L2-normalised vectors this is equivalent to cosine similarity
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    logger.info("FAISS index has %d vectors", index.ntotal)

    # ── Save index ────────────────────────────────────────────────────────
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    index_path = VECTOR_DIR / "index.faiss"
    faiss.write_index(index, str(index_path))
    logger.info("Index written to %s", index_path)

    # ── Save metadata (parallel array to index) ───────────────────────────
    metadata = [
        {
            "id": i,
            "name": a.get("name", ""),
            "url": a.get("url", ""),
            "test_type": a.get("test_type", "Assessment"),
            "description": a.get("description", ""),
            "skills_measured": a.get("skills_measured", []),
            "duration": a.get("duration", ""),
            "suitable_for": a.get("suitable_for", []),
            "remote_testing": a.get("remote_testing", True),
            "adaptive": a.get("adaptive", False),
        }
        for i, a in enumerate(assessments)
    ]
    meta_path = VECTOR_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Metadata written to %s", meta_path)

    # ── Save processed assessment list ────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    assessments_path = DATA_DIR / "assessments.json"
    with open(assessments_path, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)
    logger.info("Assessment data written to %s", assessments_path)


def main() -> None:
    args = parse_args()

    if args.scrape:
        logger.info("Mode: forced live scraping")
        assessments = load_or_scrape(use_seed=False)
    elif args.use_seed:
        logger.info("Mode: seed data only")
        assessments = load_seed_data()
    else:
        logger.info("Mode: scrape with seed fallback")
        assessments = load_or_scrape(use_seed=False)

    logger.info("Total assessments to index: %d", len(assessments))
    build_index(assessments)
    logger.info("Done! Vector store is ready.")


if __name__ == "__main__":
    main()
