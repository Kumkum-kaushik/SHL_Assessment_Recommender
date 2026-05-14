"""
Core agent logic for the SHL Assessment Recommender.

Implements deterministic intent routing:
  vague query          → CLARIFY
  enough info          → RECOMMEND
  modifying prev recs  → REFINE
  compare assessments  → COMPARE
  off-topic / attack   → REFUSE

The API is stateless: full conversation history is passed with every request.
"""

import logging
import re

from app.models.schemas import ChatRequest, ChatResponse, Message, Recommendation
from app.prompts.templates import (
    CLARIFICATION_PROMPT,
    COMPARISON_PROMPT,
    INTENT_CLASSIFICATION_PROMPT,
    RECOMMENDATION_PROMPT,
    REFINEMENT_PROMPT,
    REFUSAL_PROMPT,
)
from app.retrieval.retriever import get_retriever
from app.services.llm import call_llm_json, classify_intent
from app.utils.helpers import format_conversation_history, format_retrieved_context, safe_fallback_response

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Pre-filter patterns (no LLM cost)
# ─────────────────────────────────────────────

_OFF_TOPIC_PATTERNS = [
    r"\b(legal|lawsuit|lawyer|court|sue|attorney|litigation)\b",
    r"\b(medical|diagnos|prescri|treatment|doctor|clinical)\b",
    r"\b(hogan|korn ferry|criteria|hirevue|pymetrics|mercer mettl|assessment\.com)\b",
    r"\b(write me|draft|template|script|email|cover letter|resume|cv)\b",
    r"\b(interview questions?|onboarding|background check|reference check)\b",
    r"\bsalary|compensation|benefits|payroll\b",
    r"ignore\s+.{0,20}\binstructions?\b",
    r"pretend (you are|to be|you're)",
    r"new (system )?prompt",
    r"jailbreak|dan mode|act as",
    r"forget\s+.{0,20}\b(instructions?|rules?|guidelines?)\b",
]

_COMPARE_PATTERNS = [
    r"\b(compare|comparison|versus|vs\.?|difference between|which is better|distinguish)\b",
]

_OFF_TOPIC_RE = re.compile("|".join(_OFF_TOPIC_PATTERNS), re.IGNORECASE)
_COMPARE_RE = re.compile("|".join(_COMPARE_PATTERNS), re.IGNORECASE)


def _is_off_topic(text: str) -> bool:
    return bool(_OFF_TOPIC_RE.search(text))


def _looks_like_comparison(text: str) -> bool:
    return bool(_COMPARE_RE.search(text))


def _has_prior_recommendations(messages: list[Message]) -> bool:
    """Check if the assistant has already made recommendations in this conversation."""
    for msg in messages:
        if msg.role == "assistant" and (
            "recommendation" in msg.content.lower()
            or "assess" in msg.content.lower()
        ):
            return True
    return False


def _has_enough_context(messages: list[Message]) -> bool:
    """
    Heuristic: does the conversation contain enough info to attempt retrieval?
    Looks for job-role-level signals.
    """
    all_user_text = " ".join(m.content for m in messages if m.role == "user").lower()
    has_role = any(
        k in all_user_text
        for k in [
            "developer", "engineer", "analyst", "manager", "director", "lead",
            "sales", "marketing", "hr", "human resources", "finance", "accounting",
            "data", "designer", "customer", "operations", "warehouse", "driver",
            "nursing", "teacher", "graduate", "intern", "executive", "c-suite",
            "java", "python", "software", "front-end", "backend", "full-stack",
            "junior", "senior", "mid-level", "entry", "hire", "hiring",
            "recruit", "assess", "personality", "cognitive", "technical",
        ]
    )
    return has_role


# ─────────────────────────────────────────────
# Intent detection
# ─────────────────────────────────────────────

def detect_intent(messages: list[Message]) -> str:
    """
    Two-stage intent detection:
    1. Fast rule-based pre-filters (no LLM cost)
    2. LLM classification for ambiguous cases
    """
    last_user = next(
        (m.content for m in reversed(messages) if m.role == "user"), ""
    )

    # Stage 1: rule-based pre-filters
    if _is_off_topic(last_user):
        logger.info("Intent (rule): REFUSE — off-topic detected")
        return "REFUSE"

    if _looks_like_comparison(last_user):
        logger.info("Intent (rule): COMPARE")
        return "COMPARE"

    if _has_prior_recommendations(messages) and not _has_enough_context(messages):
        # User is continuing conversation without adding new job-role info → refine
        logger.info("Intent (rule): REFINE — modifying prior recommendations")
        return "REFINE"

    if not _has_enough_context(messages):
        logger.info("Intent (rule): CLARIFY — insufficient context")
        return "CLARIFY"

    # Stage 2: LLM classification for nuanced cases
    history = format_conversation_history(messages[:-1])  # Exclude last message
    prompt = INTENT_CLASSIFICATION_PROMPT.format(
        history=history or "(start of conversation)",
        last_message=last_user,
    )
    intent = classify_intent(prompt)
    logger.info("Intent (LLM): %s", intent)
    return intent


# ─────────────────────────────────────────────
# Action handlers
# ─────────────────────────────────────────────

def _handle_clarify(messages: list[Message]) -> dict:
    history = format_conversation_history(messages)
    prompt = CLARIFICATION_PROMPT.format(history=history)
    return call_llm_json(prompt)


def _handle_recommend(messages: list[Message]) -> dict:
    retriever = get_retriever()
    query = retriever.build_retrieval_query(messages)
    retrieved = retriever.search(query, k=8)
    context = format_retrieved_context(retrieved)
    history = format_conversation_history(messages)
    prompt = RECOMMENDATION_PROMPT.format(context=context, history=history)
    result = call_llm_json(prompt)
    # Ground check: ensure returned URLs exist in retrieved context
    result["recommendations"] = _ground_recommendations(
        result.get("recommendations", []), retrieved
    )
    return result


def _handle_refine(messages: list[Message]) -> dict:
    retriever = get_retriever()
    query = retriever.build_retrieval_query(messages)
    retrieved = retriever.search(query, k=10)
    context = format_retrieved_context(retrieved)
    history = format_conversation_history(messages)
    prompt = REFINEMENT_PROMPT.format(context=context, history=history)
    result = call_llm_json(prompt)
    result["recommendations"] = _ground_recommendations(
        result.get("recommendations", []), retrieved
    )
    return result


def _handle_compare(messages: list[Message]) -> dict:
    last_user = next(
        (m.content for m in reversed(messages) if m.role == "user"), ""
    )
    retriever = get_retriever()
    # Search for assessments mentioned by name + general query
    retrieved = retriever.search(last_user, k=6)
    context = format_retrieved_context(retrieved)
    history = format_conversation_history(messages)
    prompt = COMPARISON_PROMPT.format(context=context, history=history)
    result = call_llm_json(prompt)
    result["recommendations"] = _ground_recommendations(
        result.get("recommendations", []), retrieved
    )
    return result


def _handle_refuse(messages: list[Message]) -> dict:
    last_user = next(
        (m.content for m in reversed(messages) if m.role == "user"), ""
    )
    history = format_conversation_history(messages[:-1])
    prompt = REFUSAL_PROMPT.format(
        history=history or "(start of conversation)",
        last_message=last_user,
    )
    result = call_llm_json(prompt)
    result["recommendations"] = []  # Always empty for refusals
    return result


# ─────────────────────────────────────────────
# Grounding: prevent hallucinated recommendations
# ─────────────────────────────────────────────

def _ground_recommendations(
    recs: list[dict], retrieved: list[dict]
) -> list[dict]:
    """
    Remove any recommendations whose URL does not appear in retrieved context.
    This is the anti-hallucination safeguard — the LLM can only recommend
    assessments that were actually retrieved from the catalog.
    """
    if not retrieved:
        return []

    # Build a set of valid (name, url) pairs from retrieval results
    valid_urls = {r["url"].rstrip("/").lower() for r in retrieved}
    valid_names = {r["name"].lower() for r in retrieved}

    grounded = []
    for rec in recs:
        rec_url = rec.get("url", "").rstrip("/").lower()
        rec_name = rec.get("name", "").lower()

        # Accept if URL matches OR name matches (LLM might slightly rephrase the URL)
        if rec_url in valid_urls or any(
            vn in rec_name or rec_name in vn for vn in valid_names
        ):
            # Canonicalize: use the retrieved metadata's URL and name
            matched = next(
                (r for r in retrieved if r["url"].rstrip("/").lower() == rec_url
                 or r["name"].lower() in rec_name or rec_name in r["name"].lower()),
                None,
            )
            if matched:
                grounded.append({
                    "name": matched["name"],
                    "url": matched["url"],
                    "test_type": matched.get("test_type", rec.get("test_type", "Assessment")),
                })
            else:
                grounded.append(rec)

    return grounded[:10]


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def process_chat(request: ChatRequest) -> ChatResponse:
    """
    Main agent function. Receives full conversation history, returns structured response.
    """
    messages = request.messages

    if not messages:
        return ChatResponse(
            reply="Hello! I am your SHL Assessment Recommender. Tell me about the role you are hiring for and I will suggest the most relevant SHL assessments.",
            recommendations=[],
            end_of_conversation=False,
        )

    try:
        intent = detect_intent(messages)

        handler_map = {
            "CLARIFY": _handle_clarify,
            "RECOMMEND": _handle_recommend,
            "REFINE": _handle_refine,
            "COMPARE": _handle_compare,
            "REFUSE": _handle_refuse,
        }

        handler = handler_map.get(intent, _handle_clarify)
        result = handler(messages)

    except Exception as exc:
        logger.error("Agent error: %s", exc, exc_info=True)
        result = safe_fallback_response()

    return ChatResponse(
        reply=result.get("reply", ""),
        recommendations=[
            Recommendation(**r) for r in result.get("recommendations", [])
        ],
        end_of_conversation=result.get("end_of_conversation", False),
    )
