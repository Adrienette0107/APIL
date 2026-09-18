from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PromptDNA(BaseModel):
    original_prompt: str
    intent: str | None = None
    task: str | None = None
    subject: str | None = None
    topic: str | None = None
    domain: str | None = None
    context: str | None = None
    audience: str | None = None
    expertise_level: str | None = None
    language: str | None = None
    tone: str | None = None
    desired_depth: str | None = None
    desired_length: str | None = None
    output_format: str | None = None
    constraints: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    code_requirements: list[str] = Field(default_factory=list)
    ambiguity: str | None = None
    missing_information: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    requested_examples: bool = False
    special_instructions: list[str] = Field(default_factory=list)
    user_preferences: dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"
