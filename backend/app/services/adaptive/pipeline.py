from backend.app.services.adaptive.analyzer import analyze_prompt
from backend.app.services.adaptive.optimizer import build_adaptive_instruction
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

    messages = [
        {
            "role": "system",
            "content": instruction,
        }
    ]

    messages.extend(history)

    if not history or history[-1].get("content") != prompt:
        messages.append(
            {
                "role": "user",
                "content": prompt,
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
