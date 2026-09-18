from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    language: str | None = None

    level: Literal[
        "beginner",
        "intermediate",
        "advanced",
    ] | None = None

    response_length: Literal[
        "short",
        "medium",
        "long",
    ] | None = None


class ChatRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=10000,
    )

    model: str = Field(
        default="auto",
        max_length=100,
    )

    user_id: str = Field(
        default_factory=lambda: f"anonymous-{uuid4()}",
        min_length=1,
        max_length=255,
        description="Optional caller identity. Generated when omitted.",
    )

    conversation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        max_length=255,
        description="Optional conversation identity. Generated when omitted.",
    )

    preferences: UserPreferences | None = None


class ChatResponse(BaseModel):
    request_id: str
    status: str
    original_prompt: str
    optimized_prompt: str
    prompt_dna: dict
    selected_model: str
    selected_provider: str | None = None
    response: str
    response_evaluation: dict | None = None
    improvement_applied: bool = False
    message: str | None = None
    processed_prompt: str | None = None