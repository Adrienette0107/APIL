from typing import Literal

from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    language: str = "English"

    level: Literal[
        "beginner",
        "intermediate",
        "advanced",
    ] = "beginner"

    response_length: Literal[
        "short",
        "medium",
        "long",
    ] = "medium"


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
        ...,
        min_length=1,
        max_length=255,
    )

    conversation_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
    )

    preferences: UserPreferences | None = None


class ChatResponse(BaseModel):
    request_id: str
    status: str
    original_prompt: str
    selected_model: str
    message: str