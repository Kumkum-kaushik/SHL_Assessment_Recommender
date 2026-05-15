"""
LLM client supporting Google Gemini and OpenRouter.

Provider is selected via the LLM_PROVIDER environment variable.
Includes retry logic, timeout handling, and structured JSON extraction.
"""

import logging
import os
import time
from typing import Optional

from app.utils.helpers import extract_json, safe_fallback_response

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _cfg(key: str, default: str = "") -> str:
    """Read env var at call time (not import time) so .env is always loaded first."""
    return os.getenv(key, default)

MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds


# ─────────────────────────────────────────────
# Gemini client
# ─────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    """Call Google Gemini API and return raw text response."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_cfg("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=_cfg("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )
    return response.text


# ─────────────────────────────────────────────
# OpenRouter client
# ─────────────────────────────────────────────

def _call_openrouter(prompt: str) -> str:
    """Call OpenRouter API (OpenAI-compatible) and return raw text response."""
    from openai import OpenAI

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=_cfg("OPENROUTER_API_KEY"),
    )
    response = client.chat.completions.create(
        model=_cfg("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def _call_groq(prompt: str) -> str:
    """Call Groq API (OpenAI-compatible, free tier) and return raw text response."""
    from openai import OpenAI

    client = OpenAI(
        base_url=GROQ_BASE_URL,
        api_key=_cfg("GROQ_API_KEY"),
    )
    response = client.chat.completions.create(
        model=_cfg("GROQ_MODEL", "llama-3.1-8b-instant"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


# ─────────────────────────────────────────────
# Unified interface
# ─────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    """
    Call the configured LLM provider with retry logic.
    Returns raw text. Raises RuntimeError after exhausting retries.
    """
    provider = _cfg("LLM_PROVIDER", "gemini").lower()
    if provider == "gemini":
        caller = _call_gemini
    elif provider == "groq":
        caller = _call_groq
    else:
        caller = _call_openrouter

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug("LLM call attempt %d/%d", attempt, MAX_RETRIES)
            result = caller(prompt)
            logger.debug("LLM response length: %d chars", len(result))
            return result
        except Exception as exc:
            logger.warning("LLM call failed (attempt %d): %s", attempt, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError(f"LLM failed after {MAX_RETRIES} retries")


def call_llm_json(prompt: str) -> dict:
    """
    Call the LLM and parse the response as JSON.
    Falls back to safe_fallback_response on parse failure.
    """
    try:
        raw = call_llm(prompt)
        parsed = extract_json(raw)
        if parsed and "reply" in parsed:
            # Enforce schema constraints
            parsed.setdefault("recommendations", [])
            parsed.setdefault("end_of_conversation", False)
            # Ensure recommendations list items have required fields
            clean_recs = []
            for rec in parsed.get("recommendations", []):
                if isinstance(rec, dict) and rec.get("name") and rec.get("url"):
                    clean_recs.append({
                        "name": rec.get("name", ""),
                        "url": rec.get("url", ""),
                        "test_type": rec.get("test_type", "Assessment"),
                    })
            parsed["recommendations"] = clean_recs[:10]  # Cap at 10
            return parsed
    except Exception as exc:
        logger.error("LLM JSON call failed: %s", exc)

    return safe_fallback_response()


def classify_intent(prompt: str) -> str:
    """
    Call LLM for intent classification.
    Returns one of: CLARIFY, RECOMMEND, REFINE, COMPARE, REFUSE.
    """
    valid_intents = {"CLARIFY", "RECOMMEND", "REFINE", "COMPARE", "REFUSE"}

    try:
        raw = call_llm(prompt)
        intent = raw.strip().upper().split()[0]  # Take first word only
        if intent in valid_intents:
            return intent
        logger.warning("Unexpected intent from LLM: %r — defaulting to CLARIFY", raw)
    except Exception as exc:
        logger.error("Intent classification failed: %s", exc)

    return "CLARIFY"
