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

_SATISFIED_PATTERNS = re.compile(
    r"\b(perfect|great|thanks|thank you|that works|that'?s (what|it|all|correct|perfect)|"
    r"looks good|approved|confirmed|finalized|locking|that covers|all good|sounds good|"
    r"got it|done|exactly|just what|no more|nothing else|that'?s enough|we'?re good)\b",
    re.IGNORECASE,
)

_OFF_TOPIC_PATTERNS = [
    # Legal ADVICE requests (not hiring for legal roles)
    r"\b(lawsuit|lawyer|court|sue|attorney|litigation)\b",
    r"\blegal (advice|counsel|opinion|requirement|obligation|compliance question)\b",
    # Medical ADVICE (not hiring for medical/healthcare roles)
    r"\b(diagnos|prescri|prescription|medical advice|treatment plan|clinical trial result)\b",
    # Competitor tools
    r"\b(hogan|korn ferry|criteria|hirevue|pymetrics|mercer mettl|assessment\.com)\b",
    # Writing tasks (not assessment selection)
    r"\b(write me|draft me|draft a|generate a template|write an email|cover letter|resume|cv)\b",
    # Non-assessment HR tasks
    r"\b(interview questions?|onboarding plan|background check|reference check)\b",
    r"\b(salary|compensation|benefits|payroll)\b",
    # Prompt injection
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
    """Check if the assistant has already made CONCRETE recommendations (URLs present)."""
    for msg in messages:
        if msg.role == "assistant" and "shl.com" in msg.content.lower():
            return True
    return False


def _count_clarification_turns(messages: list[Message]) -> int:
    """Count assistant turns that asked questions but gave no recommendations."""
    count = 0
    for msg in messages:
        if msg.role == "assistant" and "?" in msg.content and "shl.com" not in msg.content.lower():
            count += 1
    return count


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
            "contact center", "contact centre", "call center", "call centre",
            "plant operator", "manufacturing", "industrial", "safety",
            "leadership", "cxo", "ceo", "cfo", "cto", "trainee", "agent",
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
    user_turn_count = sum(1 for m in messages if m.role == "user")
    total_message_count = len(messages)

    # Stage 1: rule-based pre-filters
    if _is_off_topic(last_user):
        logger.info("Intent (rule): REFUSE — off-topic detected")
        return "REFUSE"

    if _looks_like_comparison(last_user):
        logger.info("Intent (rule): COMPARE")
        return "COMPARE"

    # Hard turn cap: force RECOMMEND when approaching the 8-turn limit.
    # Triggers at user turn 6 (one turn early) OR total messages >= 11
    # so we always return a recommendation before the conversation is cut off.
    if total_message_count >= 7:
        logger.info(
            "Intent (rule): RECOMMEND — approaching turn cap (total_messages=%d)",
            total_message_count,
        )
        return "RECOMMEND"

    # If agent already gave concrete recs and user didn't add new role context → refine
    if _has_prior_recommendations(messages) and not _has_enough_context(messages):
        logger.info("Intent (rule): REFINE — modifying prior recommendations")
        return "REFINE"

    # After 2 clarification rounds, stop asking and try to recommend with what we have
    if _count_clarification_turns(messages) >= 2:
        logger.info("Intent (rule): RECOMMEND — 2 clarifications done, attempting recommendation")
        return "RECOMMEND"

    if not _has_enough_context(messages):
        logger.info("Intent (rule): CLARIFY — insufficient context")
        return "CLARIFY"

    # Enough context on first user message → recommend directly, no LLM needed.
    # Prevents LLM from over-clarifying when role + skill signals are already present.
    if _count_clarification_turns(messages) == 0 and not _has_prior_recommendations(messages):
        logger.info("Intent (rule): RECOMMEND — sufficient context on first pass")
        return "RECOMMEND"

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


# Technology keywords that warrant dedicated sub-queries because they get
# diluted when mixed with other terms in a long query.
_TECH_KEYWORDS = re.compile(
    r"\b(java|python|sql|aws|docker|spring|angular|react|node|kubernetes|"
    r"linux|networking|rest|restful|javascript|typescript|css|html|"
    r"excel|word|office|powerpoint|outlook|sharepoint|"
    r"medical|hipaa|nursing|healthcare|"
    r"sales|contact.?center|customer.?service|"
    r"global skills|safety|dependability|warehouse)\b",
    re.IGNORECASE,
)

# Domain-level signals → supplementary queries that pull niche but relevant items
_DOMAIN_SUPPLEMENTS: list[tuple[re.Pattern, list[tuple[str, int]]]] = [
    # Healthcare / medical admin roles → medical terminology, DSI
    (re.compile(r"\b(healthcare|medical|hospital|patient|clinical|nursing|hipaa)\b", re.IGNORECASE),
     [("medical terminology healthcare knowledge HIPAA", 3),
      ("dependability safety instrument DSI trust reliability", 2)]),
    # Sales / reskilling / talent audit → global skills assessment
    (re.compile(r"\b(reskill|re.skill|talent audit|restructur|sales organization|upskill)\b", re.IGNORECASE),
     [("global skills assessment development report talent", 3)]),
    # Software engineers (any language) → live coding interview
    (re.compile(r"\b(engineer|developer|software|coding|programmer)\b", re.IGNORECASE),
     [("smart interview live coding technical programming", 2)]),
    # Graduate / entry-level → situational judgement scenarios
    (re.compile(r"\b(graduate|trainee|entry.?level|intern|fresh)\b", re.IGNORECASE),
     [("graduate scenarios situational judgment entry level", 2)]),
    # Safety-critical / warehouse / industrial plant → dependability + manufacturing bundle
    (re.compile(r"\b(safety|warehouse|manufacturing|dependab|front.?line|plant|chemical|industrial|operator)\b", re.IGNORECASE),
     [("dependability safety instrument DSI workplace health safety", 3),
      ("manufacturing industrial safety dependability 8.0 plant operator", 3)]),
    # Office / admin roles → Microsoft Office assessments (knowledge + 365 simulations)
    (re.compile(r"\b(admin|administrative|office|secretary|records|clerical)\b", re.IGNORECASE),
     [("Microsoft Word Excel Office 365 simulation administrative skills", 4),
      ("MS Excel MS Word knowledge administrative assistant", 3)]),
    # Leadership / executive / senior management selection
    (re.compile(r"\b(leadership|executive|cxo|ceo|cfo|cto|vp\b|director|senior management|c.suite|c.level)\b", re.IGNORECASE),
     [("OPQ leadership report executive senior selection benchmark", 3),
      ("OPQ universal competency report personality behavior UCR", 3)]),
    # Contact center / customer service roles
    (re.compile(r"\b(contact.?cent(er|re)|call.?cent(er|re)|customer.?service|inbound|outbound|call agent)\b", re.IGNORECASE),
     [("SVAR spoken English contact center call simulation screen", 3),
      ("entry level customer service retail contact center personality", 3),
      ("customer service phone simulation biodata situational judgment", 2)]),
    # Networking / systems / infrastructure / low-level languages
    (re.compile(r"\b(network(ing)?|infrastructure|systems? programming|rust|embedded|linux|low.?level)\b", re.IGNORECASE),
     [("networking implementation linux programming general systems", 3)]),
    # Finance / accounting / analyst
    (re.compile(r"\b(finance|financial|accounting|accountant|analyst|investment|banking|fintech)\b", re.IGNORECASE),
     [("financial accounting statistics numerical reasoning analyst", 3),
      ("SHL Verify Interactive numerical reasoning finance graduate analyst", 3),
      ("basic statistics math quantitative knowledge test", 2)]),
]


def _multi_query_retrieve(retriever, query: str, k_main: int = 15) -> list[dict]:
    """
    Multi-query retrieval to ensure broad coverage.

    1. Primary: full conversation query (role/context)
    2. Supplementary standard: OPQ32r + Verify Interactive G+ (universally relevant)
    3. Supplementary per-tech: each technology keyword gets its own search
       to avoid dilution in multi-skill queries (Java+Spring+SQL+AWS+Docker)
    4. Supplementary per-domain: domain-pattern signals pull niche items
       that require agent inference (e.g. healthcare → medical terminology)
    """
    results = retriever.search(query, k=k_main)
    seen_urls = {r["url"] for r in results}

    def _add(sup_query: str, k: int) -> None:
        for r in retriever.search(sup_query, k=k):
            if r["url"] not in seen_urls:
                results.append(r)
                seen_urls.add(r["url"])

    # Always-relevant assessments (score poorly on job-specific queries)
    _add("OPQ32r occupational personality questionnaire workplace behavior selection", 3)
    _add("SHL Verify Interactive G+ adaptive cognitive ability general reasoning", 3)

    # Per-technology sub-queries
    seen_tech: set[str] = set()
    for tech in _TECH_KEYWORDS.findall(query):
        t = tech.lower()
        if t not in seen_tech:
            seen_tech.add(t)
            _add(f"{tech} knowledge assessment skills test", 3)

    # Domain-level supplementary queries
    for pattern, supplements in _DOMAIN_SUPPLEMENTS:
        if pattern.search(query):
            for sup_query, k in supplements:
                _add(sup_query, k)

    return results


def _handle_recommend(messages: list[Message]) -> dict:
    retriever = get_retriever()
    query = retriever.build_retrieval_query(messages)
    retrieved = _multi_query_retrieve(retriever, query, k_main=12)
    context = format_retrieved_context(retrieved)
    history = format_conversation_history(messages)
    prompt = RECOMMENDATION_PROMPT.format(context=context, history=history)
    result = call_llm_json(prompt)
    result["recommendations"] = _ground_recommendations(
        result.get("recommendations", []), retrieved
    )
    return result


def _handle_refine(messages: list[Message]) -> dict:
    retriever = get_retriever()
    query = retriever.build_retrieval_query(messages)
    retrieved = _multi_query_retrieve(retriever, query, k_main=10)
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
    # Search by full last message + each word individually to catch named assessments
    # k=5 per search ensures acronyms like GSA/OPQ find their full catalog entries
    seen_urls: set[str] = set()
    retrieved: list[dict] = []
    for q in [last_user] + last_user.split():
        if len(q) < 3:
            continue
        for r in retriever.search(q, k=5):
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                retrieved.append(r)
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
    Remove any recommendations whose URL/name doesn't match the retrieved context.
    Canonicalizes names and URLs back to exact catalog values to prevent drift.
    """
    if not retrieved:
        return []

    # Index retrieved items by normalised URL and normalised name
    by_url = {r["url"].rstrip("/").lower(): r for r in retrieved}
    by_name = {r["name"].lower(): r for r in retrieved}

    grounded = []
    for rec in recs:
        rec_url = rec.get("url", "").rstrip("/").lower()
        rec_name = rec.get("name", "").lower()

        matched = None

        # Exact URL match first (most reliable)
        if rec_url in by_url:
            matched = by_url[rec_url]
        else:
            # Exact name match
            if rec_name in by_name:
                matched = by_name[rec_name]
            else:
                # Substring name match (LLM may shorten/lengthen names slightly)
                for catalog_name, item in by_name.items():
                    if catalog_name in rec_name or rec_name in catalog_name:
                        matched = item
                        break

        if matched:
            grounded.append({
                "name": matched["name"],
                "url": matched["url"],
                "test_type": matched.get("test_type", rec.get("test_type", "Assessment")),
            })
        else:
            logger.debug("Grounding dropped hallucinated rec: %s (%s)", rec.get("name"), rec.get("url"))

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
        intent = "CLARIFY"

    recs = [Recommendation(**r) for r in result.get("recommendations", [])]

    # end_of_conversation: true only when the user signals they are satisfied
    # AND we have already given them a shortlist. Never true on the first
    # recommendation turn — the user must confirm before we close.
    last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
    user_is_satisfied = bool(_SATISFIED_PATTERNS.search(last_user))
    end_conv = (
        user_is_satisfied and _has_prior_recommendations(messages)
    ) or result.get("end_of_conversation", False)

    return ChatResponse(
        reply=result.get("reply", ""),
        recommendations=recs,
        end_of_conversation=end_conv,
    )
