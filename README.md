# SHL Assessment Recommender

A production-ready conversational AI agent that recommends assessments from the SHL catalog. Built for the SHL AI Intern take-home assignment.

## Architecture

```
app/
├── main.py                  # FastAPI app factory + startup lifecycle
├── routes/chat.py           # /health and /chat endpoints
├── models/schemas.py        # Pydantic request/response models
├── services/
│   ├── agent.py             # Core agent logic — intent routing, grounding
│   └── llm.py               # Gemini / OpenRouter LLM client
├── retrieval/
│   ├── scraper.py           # BeautifulSoup SHL catalog scraper + seed loader
│   ├── embedder.py          # SentenceTransformers embedding (all-MiniLM-L6-v2)
│   └── retriever.py         # FAISS similarity search
├── prompts/templates.py     # Grounded prompt templates for each intent
└── utils/helpers.py         # JSON extraction, context formatting, fallbacks

data/
└── seed_assessments.json    # 23 curated SHL assessments with URLs

vectorstore/
├── index.faiss              # Built by scripts/build_index.py
└── metadata.json            # Parallel metadata for FAISS index

scripts/
└── build_index.py           # Offline indexing script

tests/
└── test_conversations.py    # Unit + integration tests
```

## How It Works

1. **Scraping**: `scraper.py` crawls the SHL website with BeautifulSoup. On failure it falls back to `data/seed_assessments.json`.
2. **Indexing**: `build_index.py` embeds all assessments with `sentence-transformers/all-MiniLM-L6-v2` and builds a FAISS `IndexFlatIP` (cosine similarity via inner product on L2-normalised vectors).
3. **Intent routing**: `agent.py` uses rule-based pre-filters (no LLM cost) for off-topic/comparison detection, then falls back to LLM classification.
4. **Retrieval**: The user's query is embedded and the top-k most similar assessments are retrieved from FAISS.
5. **Grounding**: The LLM receives only the retrieved catalog context and is strictly instructed not to invent assessments or URLs.
6. **Anti-hallucination**: After LLM generation, `_ground_recommendations()` removes any recommendations whose name/URL don't match the retrieved context.

## Supported Intents

| Intent | Trigger | Response |
|--------|---------|----------|
| CLARIFY | Vague request, no job role/level | Ask 1-2 focused questions, `recommendations: []` |
| RECOMMEND | Job role + level or skills specified | Top 1-10 relevant SHL assessments |
| REFINE | Previous recommendations + new constraint | Updated recommendation list |
| COMPARE | "compare", "vs", "difference between" | Structured side-by-side comparison |
| REFUSE | Off-topic, competitor names, prompt injection | Polite refusal, `recommendations: []` |

## API

### GET /health
```json
{ "status": "ok" }
```

### POST /chat
**Request:**
```json
{
  "messages": [
    { "role": "user", "content": "I need to hire a senior Java developer" }
  ]
}
```

**Response:**
```json
{
  "reply": "For a senior Java developer, I recommend the following SHL assessments...",
  "recommendations": [
    {
      "name": "Coding Pro — Java",
      "url": "https://www.shl.com/solutions/products/assessments/technology-skills/",
      "test_type": "Technical Skills"
    }
  ],
  "end_of_conversation": false
}
```

**Schema rules:**
- `recommendations` is `[]` when clarifying
- `recommendations` contains 1–10 items when recommending
- `end_of_conversation` is `true` only when the task is fully complete
- No markdown in API responses

## Example Conversations

### Vague → Clarify → Recommend
```
User: "I need an assessment"
Agent: "I'd be happy to help! Could you tell me: (1) What role are you hiring for? (2) What seniority level?"
recommendations: []

User: "Mid-level Java developer who will work with stakeholders"
Agent: "Based on your requirements, here are the most relevant SHL assessments..."
recommendations: [Coding Pro Java, Verify Inductive Reasoning, OPQ32]
```

### Refinement
```
User: "Also add a personality assessment"
Agent: "I've updated your recommendations to include personality assessment..."
recommendations: [Coding Pro Java, Verify Inductive Reasoning, OPQ32, RemoteWorkQ]
```

### Comparison
```
User: "What is the difference between OPQ32 and MQ?"
Agent: "OPQ32 measures 32 personality dimensions for predicting workplace behaviour, while MQ focuses on..."
recommendations: [OPQ32, Motivation Questionnaire]
```

### Off-topic Refusal
```
User: "Can you help me write interview questions?"
Agent: "I specialise in SHL assessment recommendations only. I'm unable to help with interview question writing, but I can recommend SHL's Virtual Interview product..."
recommendations: []
```

## Local Development

### Prerequisites
- Python 3.11+
- A Gemini API key (or OpenRouter API key)

### Setup
```bash
# 1. Clone and enter project
cd SHL_Assessment

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 5. Build the FAISS index (uses seed data, no network required)
python scripts/build_index.py --use-seed

# 6. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Building the index with live scraping
```bash
python scripts/build_index.py --scrape
```

### Running tests
```bash
python -m pytest tests/ -v
# or
python -m unittest tests/test_conversations.py -v
```

## Docker

```bash
# Build (includes index building from seed data)
docker build -t shl-recommender .

# Run
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key_here \
  -e LLM_PROVIDER=gemini \
  shl-recommender
```

## Deployment on Render

1. Push this repository to GitHub.
2. Go to [render.com](https://render.com) → New → Web Service → connect your repo.
3. Render will detect `render.yaml` automatically.
4. In the Render dashboard, add your secret environment variable:
   - `GEMINI_API_KEY` = `your_api_key`
5. Deploy. The build command installs dependencies and builds the index; the start command launches the server.

The service binds to Render's dynamic `$PORT` variable automatically.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | `gemini` or `openrouter` |
| `GEMINI_API_KEY` | — | **Required** if provider is gemini |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model name |
| `OPENROUTER_API_KEY` | — | **Required** if provider is openrouter |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | OpenRouter model name |
| `PORT` | `8000` | Server port (set automatically by Render) |

## Design Decisions

**Why FAISS over a managed vector DB?**
Zero external dependencies, no extra API keys needed, fully portable in a Docker image. Sufficient for a catalog of ~20-100 assessments.

**Why stateless API?**
Simpler deployment, no session storage needed, horizontally scalable. The full conversation history is cheap to pass (assessments are short).

**Why rule-based pre-filters before LLM intent classification?**
Catches obvious cases (injection attempts, competitor mentions) with zero LLM cost and zero latency. Reduces API spend and speeds up response time.

**Why seed data over live-only scraping?**
SHL's website may block automated crawlers. Seed data ensures the system works in CI, Docker builds, and offline environments. Scraping augments it when available.
