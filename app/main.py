"""
FastAPI application entry point.

Startup sequence:
1. Load environment variables from .env (if present)
2. Verify FAISS index exists; build it from seed data if not
3. Warm up the embedding model
4. Start serving requests
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env file before importing anything that reads env vars
load_dotenv()

from app.routes.chat import router as chat_router
from app.retrieval.retriever import get_retriever

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

VECTORSTORE_DIR = Path("vectorstore")
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"


def _ensure_index():
    """Build the FAISS index from seed data if it does not already exist."""
    if INDEX_PATH.exists():
        logger.info("FAISS index found at %s", INDEX_PATH)
        return

    logger.warning("FAISS index not found — building from seed data now...")
    try:
        # Import here to avoid circular imports at module load time
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "scripts/build_index.py", "--use-seed"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            logger.info("Index built successfully")
        else:
            logger.error("Index build failed:\n%s", result.stderr)
    except Exception as exc:
        logger.error("Could not build index: %s", exc)


# ─────────────────────────────────────────────
# Application lifespan
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    logger.info("Starting SHL Assessment Recommender...")
    _ensure_index()

    # Warm up retriever (loads FAISS index + embedding model into memory)
    retriever = get_retriever()
    if retriever.is_ready:
        logger.info("Retriever ready with %d assessments", len(retriever.metadata))
    else:
        logger.warning("Retriever not ready — check vectorstore directory")

    yield

    logger.info("Shutting down SHL Assessment Recommender")


# ─────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="SHL Assessment Recommender",
        description=(
            "Conversational AI agent that recommends SHL assessments "
            "grounded in the official SHL product catalog."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — allow all origins for the demo; tighten in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(chat_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
