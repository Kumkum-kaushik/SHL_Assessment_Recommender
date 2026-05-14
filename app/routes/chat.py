"""
API route handlers for /health and /chat endpoints.
"""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.models.schemas import ChatRequest, ChatResponse
from app.services.agent import process_chat

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Simple liveness probe. Returns 200 OK when the service is up."""
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Stateless conversational endpoint.

    Each request must include the FULL conversation history in `messages`.
    The agent classifies intent, retrieves relevant SHL assessments from
    the catalog, and returns a grounded, structured response.

    - recommendations is [] when still clarifying.
    - recommendations contains 1-10 items when recommending.
    - end_of_conversation is true only when the task is fully complete.
    """
    if not request.messages:
        raise HTTPException(status_code=422, detail="messages array must not be empty")

    # Validate roles
    valid_roles = {"user", "assistant"}
    for msg in request.messages:
        if msg.role not in valid_roles:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid role '{msg.role}'. Must be 'user' or 'assistant'.",
            )

    # Ensure the last message is from the user
    if request.messages[-1].role != "user":
        raise HTTPException(
            status_code=422,
            detail="The last message in the history must have role 'user'.",
        )

    try:
        response = process_chat(request)
        return response
    except Exception as exc:
        logger.error("Unhandled error in /chat: %s", exc, exc_info=True)
        # Return a safe degraded response rather than a 500 error
        return ChatResponse(
            reply=(
                "I am having trouble processing your request right now. "
                "Please try again in a moment."
            ),
            recommendations=[],
            end_of_conversation=False,
        )
