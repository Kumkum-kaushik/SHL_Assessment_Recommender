"""
Pydantic models for API request/response validation.
Strictly follows the assignment schema.
"""

from pydantic import BaseModel, Field
from typing import List


class Message(BaseModel):
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="The message content")


class ChatRequest(BaseModel):
    messages: List[Message] = Field(
        ...,
        description="Full conversation history. Each POST includes the complete history.",
        min_length=1,
    )


class Recommendation(BaseModel):
    name: str = Field(..., description="Assessment name from SHL catalog")
    url: str = Field(..., description="Direct URL to the SHL catalog page")
    test_type: str = Field(..., description="Assessment type (e.g. Cognitive, Personality, Technical Skills)")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent's natural-language response")
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="SHL assessments recommended. Empty list when clarifying.",
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the task is fully complete and no further action is needed.",
    )
