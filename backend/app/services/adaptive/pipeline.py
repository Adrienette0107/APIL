from backend.app.services.adaptive.analyzer import analyze_prompt
from backend.app.services.adaptive.optimizer import (
    build_adaptive_instruction,
    build_optimized_prompt,
)
from backend.app.services.adaptive.response import adapt_response


def build_adaptive_messages(
    prompt: str,
    history: list[dict],
    preferences: dict | None = None,
):
    analysis = analyze_prompt(prompt)

    instruction = build_adaptive_instruction(
        analysis=analysis,
        preferences=preferences,
    )

    optimized_prompt = build_optimized_prompt(
        prompt=prompt,
        analysis=analysis,
        preferences=preferences,
        has_context=bool(history),
    )

    messages = [
        {
            "role": "system",
            "content": instruction,
        }
    ]

    messages.extend(history)

    if (
        history
        and history[-1].get("role") == "user"
        and history[-1].get("content") == prompt
    ):
        messages[-1] = {
            "role": "user",
            "content": optimized_prompt,
        }
    else:
        messages.append(
            {
                "role": "user",
                "content": optimized_prompt,
            }
        )

    return messages, analysis


def process_response(
    response: str,
    preferences: dict | None = None,
):
    return adapt_response(
        response=response,
        preferences=preferences,
    )
