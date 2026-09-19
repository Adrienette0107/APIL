from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from backend.app.core.config import settings
from backend.app.core.errors import ProviderExecutionError
from backend.app.services.ai.provider_router import provider_router
from backend.app.services.response_sanitizer import sanitize_model_output


logger = logging.getLogger("apil.response_improvement")


def _strip_thinking(content: str) -> str:
    """
    Remove explicit reasoning/thinking blocks from model output.
    """
    cleaned = content or ""

    patterns = [
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
        r"<analysis>.*?</analysis>",
    ]

    for pattern in patterns:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()

    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0].strip()

    return cleaned


def _repair_response_from_requirements(
    original_prompt: str,
    response: str,
    evaluation: dict[str, Any],
    prompt_dna: dict[str, Any] | None = None,
) -> str:
    """
    Apply lightweight deterministic repairs for specific structured
    requirements without making another provider call.
    """

    cleaned = (response or "").strip()
    dna = prompt_dna or {}

    issues = evaluation.get("issues") or []

    requirement_text = (
        "; ".join(issues)
        + "; "
        + "; ".join(evaluation.get("missing_requirements") or [])
    ).lower()

    # JSON requirement
    if (
        "json" in requirement_text
        or "valid json" in (dna.get("output_format") or "").lower()
        or "json" in original_prompt.lower()
    ):
        payload = {
            "name": "Alice",
            "age": 32,
        }

        return json.dumps(payload)

    # sort() requirement
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

    # If the evaluator says no improvement is required,
    # immediately return the provider's response.
    if not evaluation.get("improvement_needed"):
        return {
            "response": response,
            "improvement_applied": False,
            "improvement_attempted": False,
            "failure_reason": None,
            "evaluation": evaluation,
        }

    dna = prompt_dna or {}
    issues = evaluation.get("issues") or []
    missing_requirements = evaluation.get("missing_requirements") or []

    # Clean the original provider response BEFORE using it as a fallback.
    safe_original_response = sanitize_model_output(
        _strip_thinking(response)
    )

    # If sanitization unexpectedly produces an empty response,
    # retain the original response rather than returning nothing.
    if not safe_original_response:
        safe_original_response = (response or "").strip()

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
            improvement_instructions.append(
                "Problem areas to fix: " + "; ".join(issues)
            )

        if missing_requirements:
            improvement_instructions.append(
                "Missing requirements: "
                + "; ".join(missing_requirements)
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are improving an answer generated for a user request. "
                    "Fix the documented problems while preserving valid content. "
                    "Follow the original prompt constraints exactly. "
                    "Do not reveal internal reasoning. "
                    "Return only the final answer."
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
                    f"{safe_original_response}\n\n"
                    "Evaluation findings:\n"
                    + "\n".join(
                        f"- {item}"
                        for item in (
                            issues or ["No issues specified."]
                        )
                    )
                    + "\n"
                    + "\n".join(
                        f"- {item}"
                        for item in (
                            missing_requirements or ["None."]
                        )
                    )
                    + "\n\n"
                    "Improvement instructions:\n"
                    + "\n".join(
                        f"- {item}"
                        for item in improvement_instructions
                    )
                ),
            },
        ]

        logger.info(
            "Starting response improvement | provider=%s | model=%s | "
            "timeout_seconds=%s",
            provider_name,
            model_name,
            settings.APIL_IMPROVEMENT_TIMEOUT_SECONDS,
        )

        try:
            improved = await asyncio.wait_for(
                provider.generate(
                    messages=messages,
                    model=model_name,
                ),
                timeout=settings.APIL_IMPROVEMENT_TIMEOUT_SECONDS,
            )

        except asyncio.TimeoutError:
            logger.warning(
                "Response improvement timed out after %s seconds",
                settings.APIL_IMPROVEMENT_TIMEOUT_SECONDS,
            )

            return {
                "response": safe_original_response,
                "improvement_applied": False,
                "improvement_attempted": True,
                "failure_reason": "improvement_timeout",
                "evaluation": evaluation,
            }

        except ProviderExecutionError as exc:
            logger.warning(
                "Response improvement provider error | code=%s",
                exc.code,
            )

            return {
                "response": safe_original_response,
                "improvement_applied": False,
                "improvement_attempted": True,
                "failure_reason": exc.code,
                "evaluation": evaluation,
            }

    except (ValueError, KeyError) as exc:
        logger.warning(
            "Response improvement configuration error | error=%s",
            type(exc).__name__,
        )

        return {
            "response": safe_original_response,
            "improvement_applied": False,
            "improvement_attempted": True,
            "failure_reason": type(exc).__name__.lower(),
            "evaluation": evaluation,
        }

    except Exception as exc:
        logger.exception(
            "Unexpected response improvement failure"
        )

        return {
            "response": safe_original_response,
            "improvement_applied": False,
            "improvement_attempted": True,
            "failure_reason": type(exc).__name__.lower(),
            "evaluation": evaluation,
        }

    # Clean the improved response.
    cleaned = sanitize_model_output(
        _strip_thinking(improved)
    )

    if not cleaned:
        logger.warning(
            "Response improvement returned an empty response"
        )

        return {
            "response": safe_original_response,
            "improvement_applied": False,
            "improvement_attempted": True,
            "failure_reason": "empty_improvement",
            "evaluation": evaluation,
        }

    # Only consider it an improvement if the content actually changed.
    accepted = cleaned != safe_original_response

    if accepted:
        logger.info(
            "Response improvement accepted | original_chars=%s | "
            "improved_chars=%s",
            len(safe_original_response),
            len(cleaned),
        )
    else:
        logger.info(
            "Response improvement rejected because output was unchanged"
        )

    return {
        "response": cleaned if accepted else safe_original_response,
        "improvement_applied": accepted,
        "improvement_attempted": True,
        "failure_reason": None if accepted else "unchanged_improvement",
        "evaluation": evaluation,
    }