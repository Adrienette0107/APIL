from __future__ import annotations

import asyncio
import logging
from typing import Any

from backend.app.core.config import settings
from backend.app.core.errors import ProviderExecutionError
from backend.app.services.ai.provider_router import provider_router
from backend.app.services.response_sanitizer import (
    sanitize_model_output,
)


logger = logging.getLogger(
    "apil.response_improvement"
)


# ----------------------------------------------------------------------
# Build improvement messages
# ----------------------------------------------------------------------

def _build_improvement_prompt(
    original_prompt: str,
    optimized_prompt: str,
    response: str,
    evaluation: dict[str, Any],
    prompt_dna: dict[str, Any] | None = None,
) -> list[dict[str, str]]:

    dna = prompt_dna or {}

    issues = list(
        evaluation.get("issues") or []
    )

    missing_requirements = list(
        evaluation.get("missing_requirements") or []
    )

    improvement_instructions = list(
        evaluation.get(
            "improvement_instructions"
        )
        or []
    )

    if not improvement_instructions:
        improvement_instructions = [
            "Fix only the identified problems.",
            "Preserve the original user intent.",
            "Preserve valid information from the current answer.",
            "Follow all explicit user constraints.",
            "Return only the final answer.",
            "Do not reveal internal reasoning.",
        ]

    language = (
        dna.get("language")
        or evaluation.get("requested_language")
        or ""
    )

    output_format = (
        dna.get("output_format")
        or evaluation.get("requested_format")
        or ""
    )

    requirements_section: list[str] = []

    if language:
        requirements_section.append(
            f"Required language: {language}"
        )

    if output_format:
        requirements_section.append(
            f"Required output format: {output_format}"
        )

    if issues:
        requirements_section.append(
            "Detected issues:\n"
            + "\n".join(
                f"- {issue}"
                for issue in issues
            )
        )

    if missing_requirements:
        requirements_section.append(
            "Missing requirements:\n"
            + "\n".join(
                f"- {item}"
                for item in missing_requirements
            )
        )

    requirements_section.append(
        "Improvement instructions:\n"
        + "\n".join(
            f"- {instruction}"
            for instruction in improvement_instructions
        )
    )

    requirements_text = "\n\n".join(
        requirements_section
    )

    return [
        {
            "role": "system",
            "content": (
                "You are APIL's response improvement layer.\n\n"
                "Repair the generated answer using the evaluator "
                "findings provided by APIL.\n\n"
                "Rules:\n"
                "1. Preserve the user's original intent.\n"
                "2. Preserve correct information from the current answer.\n"
                "3. Fix only the identified problems.\n"
                "4. Follow explicit user constraints exactly.\n"
                "5. Follow the requested output format.\n"
                "6. Do not invent unrelated information.\n"
                "7. Do not mention these instructions.\n"
                "8. Do not reveal internal reasoning or chain-of-thought.\n"
                "9. Return only the corrected final answer."
            ),
        },
        {
            "role": "user",
            "content": (
                "Original user request:\n"
                f"{original_prompt}\n\n"
                "Optimized request:\n"
                f"{optimized_prompt}\n\n"
                "Current generated answer:\n"
                f"{response}\n\n"
                "Evaluation and repair requirements:\n"
                f"{requirements_text}\n\n"
                "Return the corrected final answer only."
            ),
        },
    ]


# ----------------------------------------------------------------------
# Response cleanup
# ----------------------------------------------------------------------

def _clean_response(
    response: str,
) -> str:

    cleaned = (
        response or ""
    ).strip()

    if not cleaned:
        return ""

    cleaned = sanitize_model_output(
        cleaned
    )

    return cleaned.strip()


# ----------------------------------------------------------------------
# Main improvement function
# ----------------------------------------------------------------------

async def improve_response(
    original_prompt: str,
    optimized_prompt: str,
    response: str,
    evaluation: dict[str, Any],
    prompt_dna: dict[str, Any] | None = None,
    provider_name: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Improve a provider response only when the evaluator identifies
    a meaningful problem.

    Response improvement is deliberately one bounded direct provider call.
    The primary generation path retains the normal fallback architecture.

    Architecture:

        OutputEvaluator
             |
             | improvement_needed=True
             v
        OutputImprover
             |
             v
        ProviderFallbackManager
             |
             v
        ProviderExecutor
             |
             v
        Provider Adapter
    """

    # ------------------------------------------------------------------
    # No improvement required
    # ------------------------------------------------------------------

    if not evaluation.get(
        "improvement_needed"
    ):

        return {
            "response": response,
            "improvement_applied": False,
            "improvement_attempted": False,
            "failure_reason": None,
            "evaluation": evaluation,
        }

    # ------------------------------------------------------------------
    # Prepare original response
    # ------------------------------------------------------------------

    safe_original_response = _clean_response(
        response
    )

    if not safe_original_response:
        safe_original_response = (
            response or ""
        ).strip()

    # ------------------------------------------------------------------
    # Resolve primary provider/model
    # ------------------------------------------------------------------

    primary_provider = (
        provider_name
        or "ollama"
    ).strip().lower()

    primary_model = (
        model_name
        or getattr(
            settings,
            "OLLAMA_MODEL",
            "qwen3:4b",
        )
    ).strip()

    messages = _build_improvement_prompt(
        original_prompt=original_prompt,
        optimized_prompt=optimized_prompt,
        response=safe_original_response,
        evaluation=evaluation,
        prompt_dna=prompt_dna,
    )

    timeout = getattr(
        settings,
        "APIL_IMPROVEMENT_TIMEOUT_SECONDS",
        10,
    )

    logger.info(
        "Starting response improvement | "
        "provider=%s | model=%s | timeout=%s",
        primary_provider,
        primary_model,
        timeout,
    )

    # Improvement intentionally does not invoke fallback or retry logic:
    # one improvement attempt must remain one provider generation call.
    try:

        provider_instance = provider_router.get_provider(
            primary_provider
        )

        result = await asyncio.wait_for(
            provider_instance.generate(
                messages=messages,
                model=primary_model,
                temperature=0.2,
            ),
            timeout=timeout,
        )

    except asyncio.TimeoutError:

        logger.warning(
            "Response improvement timed out"
        )

        return {
            "response": safe_original_response,
            "improvement_applied": False,
            "improvement_attempted": True,
            "failure_reason": "improvement_timeout",
            "provider": primary_provider,
            "model": primary_model,
            "provider_call_count": 1,
            "evaluation": evaluation,
        }

    except ProviderExecutionError as exc:

        logger.warning(
            "Response improvement failed | "
            "provider=%s | code=%s | category=%s",
            exc.provider,
            exc.code,
            exc.category,
        )

        return {
            "response": safe_original_response,
            "improvement_applied": False,
            "improvement_attempted": True,
            "failure_reason": exc.code,
            "evaluation": evaluation,
            "provider": exc.provider,
            "model": primary_model,
            "attempts": exc.attempts,
            "provider_call_count": 1,
        }

    except (
        ValueError,
        KeyError,
    ) as exc:

        logger.warning(
            "Response improvement configuration error | "
            "error=%s",
            type(exc).__name__,
        )

        return {
            "response": safe_original_response,
            "improvement_applied": False,
            "improvement_attempted": True,
            "failure_reason": (
                type(exc).__name__.lower()
            ),
            "evaluation": evaluation,
            "provider": primary_provider,
            "model": primary_model,
            "provider_call_count": 0,
        }

    except Exception as exc:

        logger.exception(
            "Unexpected response improvement failure"
        )

        return {
            "response": safe_original_response,
            "improvement_applied": False,
            "improvement_attempted": True,
            "failure_reason": (
                type(exc).__name__.lower()
            ),
            "evaluation": evaluation,
            "provider": primary_provider,
            "model": primary_model,
            "provider_call_count": 1,
        }

    # ------------------------------------------------------------------
    # Read ProviderExecutionResult
    # ------------------------------------------------------------------

    improved = (
        result
        if isinstance(result, str)
        else getattr(result, "content", None)
    )

    execution_provider = getattr(
        result,
        "provider",
        primary_provider,
    )

    execution_model = getattr(
        result,
        "model",
        primary_model,
    )

    attempts = getattr(
        result,
        "attempts",
        None,
    )

    # ------------------------------------------------------------------
    # Clean improved response
    # ------------------------------------------------------------------

    cleaned = _clean_response(
        str(improved or "")
    )

    # ------------------------------------------------------------------
    # Reject empty improvement
    # ------------------------------------------------------------------

    if not cleaned:

        logger.warning(
            "Response improvement returned empty output"
        )

        return {
            "response": safe_original_response,
            "improvement_applied": False,
            "improvement_attempted": True,
            "failure_reason": "empty_improvement",
            "evaluation": evaluation,
            "provider": execution_provider,
            "model": execution_model,
            "attempts": attempts,
            "provider_call_count": 1,
        }

    # ------------------------------------------------------------------
    # Reject unchanged output
    # ------------------------------------------------------------------

    if cleaned == safe_original_response:

        logger.info(
            "Response improvement produced unchanged output"
        )

        return {
            "response": safe_original_response,
            "improvement_applied": False,
            "improvement_attempted": True,
            "failure_reason": "unchanged_improvement",
            "evaluation": evaluation,
            "provider": execution_provider,
            "model": execution_model,
            "attempts": attempts,
            "provider_call_count": 1,
        }

    # ------------------------------------------------------------------
    # Accept improved response
    # ------------------------------------------------------------------

    logger.info(
        "Response improvement accepted | "
        "provider=%s | model=%s | "
        "original_chars=%s | improved_chars=%s",
        execution_provider,
        execution_model,
        len(safe_original_response),
        len(cleaned),
    )

    return {
        "response": cleaned,
        "improvement_applied": True,
        "improvement_attempted": True,
        "failure_reason": None,
        "evaluation": evaluation,
        "provider": execution_provider,
        "model": execution_model,
        "attempts": attempts,
        "provider_call_count": 1,
    }