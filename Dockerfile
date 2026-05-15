# ── Stage 1: Build FAISS index offline ─────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what is needed for index building
COPY data/ data/
COPY scripts/ scripts/
COPY app/ app/

# Build the FAISS index from seed data (no network required)
RUN python scripts/build_index.py --use-seed


# ── Stage 2: Production image ───────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY data/ data/
COPY scripts/ scripts/

# Copy the pre-built vector store from builder stage
COPY --from=builder /build/vectorstore/ vectorstore/

# Expose default port (Render overrides via PORT env var)
EXPOSE 8000

# Start the server using the PORT environment variable
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
