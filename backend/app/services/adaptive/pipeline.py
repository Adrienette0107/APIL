
from __future__ import annotations

from typing import Any

from backend.app.services.adaptive.analyzer import (
    analyze_prompt,
)
from backend.app.services.adaptive.optimizer import (
    build_adaptive_instruction,
)
from backend.app.services.adaptive.response import (
    adapt_response,
)


def build_adaptive_messages(
    prompt: str,
    history: list[dict[str, Any]] | None = None,
    preferences: dict[str, Any] | None = None,
    analysis: Any = None,
):
    """
    Build the canonical provider message list.

    IMPORTANT:
        PromptOptimizer has already produced the canonical
        optimized prompt before this function is called.

    This function therefore MUST NOT optimize/rewrite the prompt
        a second time.

    Responsibilities:
        1. Analyze the already optimized prompt if needed.
        2. Build the adaptive system instruction.
        3. Preserve conversation history.
        4. Insert the optimized prompt exactly once.
    """

    if history is None:
        history = []

    if preferences is None:
        preferences = {}

    # --------------------------------------------------------------
    # 1. Analyze the canonical optimized prompt
    # --------------------------------------------------------------

    analysis = (
        analysis
        if analysis is not None
        else analyze_prompt(prompt)
    )

    # --------------------------------------------------------------
    # 2. Build adaptive system instruction
    # --------------------------------------------------------------

    instruction = build_adaptive_instruction(
        analysis=analysis,
        preferences=preferences,
    )

    # --------------------------------------------------------------
    # 3. Build canonical message list
    # --------------------------------------------------------------

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": instruction,
        }
    ]

    # --------------------------------------------------------------
    # 4. Add previous conversation history
    # --------------------------------------------------------------

    for message in history:

        if not isinstance(
            message,
            dict,
        ):
            continue

        role = message.get(
            "role"
        )

        content = message.get(
            "content"
        )

        if not role or content is None:
            continue

        messages.append(
            {
                "role": str(role),
                "content": str(content),
            }
        )

    # --------------------------------------------------------------
    # 5. Add optimized user prompt exactly once
    # --------------------------------------------------------------

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    return messages, analysis


def process_response(
    response: str,
    preferences: dict[str, Any] | None = None,
) -> str:
    """
    Run the provider response through APIL's response
    adaptation/sanitization layer.

    This function does not call the provider again.
    """

    if preferences is None:
        preferences = {}

    return adapt_response(
        response=response,
        preferences=preferences,
    )

