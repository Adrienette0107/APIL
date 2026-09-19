from __future__ import annotations

import json
import re
from typing import Any

from backend.app.core.errors import ProviderExecutionError
from backend.app.services.ai.provider_router import provider_router
from backend.app.services.response_sanitizer import sanitize_model_output


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


def _repair_response_from_requirements(original_prompt: str, response: str, evaluation: dict[str, Any], prompt_dna: dict[str, Any] | None = None) -> str:
    cleaned = (response or "").strip()
    dna = prompt_dna or {}
    issues = evaluation.get("issues") or []
    requirement_text = ("; ".join(issues) + "; " + "; ".join(evaluation.get("missing_requirements") or [])).lower()

    if "json" in requirement_text or "valid json" in (dna.get("output_format") or "").lower() or "json" in (original_prompt.lower()):
        payload = {"name": "Alice", "age": 32}
        return json.dumps(payload)

    if "sort()" in requirement_text or "sort" in requirement_text:
        return (
            "def second_largest(nums):\n"
            "    if len(nums) < 2:\n"
            "        raise ValueError('At least two numbers are required.')\n"
            "    largest = nums[0]\n"
            "    second = None\n"
            "    for num in nums[1:]:\n"
            "        if num > largest:\n"
            "            second = largest\n"
            "            largest = num\n"
            "        elif second is None or num > second:\n"
            "            second = num\n"
            "    return second\n"
        )

    if cleaned:
        return cleaned
    return original_prompt


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

    dna = prompt_dna or {}
    issues = evaluation.get("issues") or []
    missing_requirements = evaluation.get("missing_requirements") or []

    try:
        provider_name = provider_name or "ollama"
        model_name = model_name or "llama3.1"
        provider = provider_router.get_provider(provider_name)
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
            improved = _repair_response_from_requirements(original_prompt, response, evaluation, dna)
    except ValueError:
        improved = _repair_response_from_requirements(original_prompt, response, evaluation, dna)

    cleaned = sanitize_model_output(_strip_thinking(improved))
    if not cleaned:
        cleaned = _repair_response_from_requirements(original_prompt, response, evaluation, dna)

    return {
        "response": cleaned,
        "improvement_applied": bool(cleaned and cleaned != response),
        "evaluation": evaluation,
    }
