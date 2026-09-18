from __future__ import annotations

import re
from typing import Any

from backend.app.core.errors import ProviderExecutionError
from backend.app.services.ai.provider_router import provider_router


def _strip_thinking(content: str) -> str:
    cleaned = content or ""
    patterns = [
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
        r"<analysis>.*?</analysis>",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0].strip()
    return cleaned


async def improve_response(
    original_prompt: str,
    optimized_prompt: str,
    response: str,
    evaluation: dict[str, Any],
    prompt_dna: dict[str, Any] | None = None,
    provider_name: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    if not evaluation.get("improvement_needed"):
        return {
            "response": response,
            "improvement_applied": False,
            "evaluation": evaluation,
        }

    provider_name = provider_name or "ollama"
    model_name = model_name or "qwen3:4b"

    try:
        provider = provider_router.get_provider(provider_name)
    except ValueError:
        return {
            "response": response,
            "improvement_applied": False,
            "evaluation": evaluation,
            "error": "Unsupported provider for improvement.",
        }

    dna = prompt_dna or {}
    issues = evaluation.get("issues") or []
    missing_requirements = evaluation.get("missing_requirements") or []

    improvement_instructions = [
        "Preserve the correct substance of the answer.",
        "Fix only the identified issues and missing requirements.",
        "Keep the original intent and requested format intact.",
        "Honor all explicit constraints from the user's request.",
        "Do not expose internal reasoning or chain-of-thought.",
        "Return only the improved final answer.",
    ]

    if issues:
        improvement_instructions.append("Problem areas to fix: " + "; ".join(issues))
    if missing_requirements:
        improvement_instructions.append("Missing requirements: " + "; ".join(missing_requirements))

    messages = [
        {
            "role": "system",
            "content": (
                "You are improving an answer generated for a user request. "
                "Fix the documented problems while preserving valid content. "
                "Follow the original prompt constraints exactly and do not reveal internal reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                "Original user request:\n"
                f"{original_prompt}\n\n"
                "Optimized request:\n"
                f"{optimized_prompt}\n\n"
                "Current answer:\n"
                f"{response}\n\n"
                "Evaluation findings:\n"
                + "\n".join(f"- {item}" for item in (issues or ["No issues specified."]))
                + "\n"
                + "\n".join(f"- {item}" for item in (missing_requirements or ["None."]))
                + "\n\n"
                "Improvement instructions:\n"
                + "\n".join(f"- {item}" for item in improvement_instructions)
            ),
        },
    ]

    try:
        improved = await provider.generate(messages=messages, model=model_name)
    except ProviderExecutionError:
        return {
            "response": response,
            "improvement_applied": False,
            "evaluation": evaluation,
            "error": "Improvement generation failed.",
        }

    cleaned = _strip_thinking(improved)
    if not cleaned:
        return {
            "response": response,
            "improvement_applied": False,
            "evaluation": evaluation,
            "error": "Improved response was empty.",
        }

    return {
        "response": cleaned,
        "improvement_applied": True,
        "evaluation": evaluation,
    }
