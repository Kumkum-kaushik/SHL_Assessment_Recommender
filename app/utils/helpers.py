"""
Utility functions shared across the application.
"""

import json
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_json(text: str) -> Optional[dict]:
    """
    Robustly extract a JSON object from LLM output.
    Handles markdown code blocks, trailing text, and minor formatting issues.
    """
    text = text.strip()

    # Try direct JSON parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Grab the first {...} block in the text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Could not extract JSON from LLM response: %s", text[:200])
    return None


def format_conversation_history(messages: list) -> str:
    """Format message list into a readable history string for prompts."""
    lines = []
    for msg in messages:
        role = msg.role.upper()
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def format_retrieved_context(assessments: list[dict]) -> str:
    """Format retrieved assessments into a grounding context block."""
    if not assessments:
        return "No relevant assessments found in the SHL catalog."

    sections = []
    for i, a in enumerate(assessments, 1):
        skills = ", ".join(a.get("skills_measured", []))
        suitable = ", ".join(a.get("suitable_for", []))
        section = (
            f"[{i}] {a['name']}\n"
            f"    URL: {a['url']}\n"
            f"    Type: {a.get('test_type', 'N/A')}\n"
            f"    Duration: {a.get('duration', 'N/A')}\n"
            f"    Measures: {skills or 'N/A'}\n"
            f"    Suitable for: {suitable or 'N/A'}\n"
            f"    Description: {a.get('description', '')[:300]}"
        )
        sections.append(section)
    return "\n\n".join(sections)


def safe_fallback_response(reason: str = "") -> dict:
    """Return a safe fallback when the LLM fails or returns unparseable output."""
    return {
        "reply": (
            "I encountered an issue generating a response. "
            "Please try rephrasing your request. I can help you find the right SHL assessments "
            "for your hiring needs."
        ),
        "recommendations": [],
        "end_of_conversation": False,
    }
