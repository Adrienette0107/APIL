from typing import Any, Literal
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
        description="User prompt to process.",
    )

    model: str = Field(
        default="auto",
        max_length=100,
        description="Model selection. Use 'auto' or provider:model.",
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
    prompt_dna: dict[str, Any]

    processed_prompt: str | None = None

    selected_provider: str | None = None
    selected_model: str

    response: str

    raw_provider_response: str | None = None
    raw_response: str | None = None
    final_response: str | None = None

    response_evaluation: dict[str, Any] | None = None
    final_quality_gate: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None

    improvement_applied: bool = False
    improvement_attempted: bool = False
    improvement_attempts: int = 0
    improvement_needed: bool = False
    improvement_error: str | None = None
    improvement_failure_reason: str | None = None

    provider_call_count: int = 1
    provider_attempts: int = 1

    timing: dict[str, Any] | None = None

    preferences: dict[str, Any] | None = None
    analysis: dict[str, Any] | None = None

    message: str | None = None