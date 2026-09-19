from __future__ import annotations

import re


def _remove_thinking_blocks(
    response: str,
) -> str:
    """
    Remove model reasoning/thinking blocks from the user-visible
    response.

    This is especially important for Qwen/Ollama models that may
    return <think>...</think> content.
    """

    if not response:
        return ""

    # Complete thinking block.
    response = re.sub(
        r"<think>.*?</think>",
        "",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Defensive cleanup for an unmatched opening tag.
    response = re.sub(
        r"<think>.*$",
        "",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )

    response = response.replace(
        "</think>",
        "",
    )

    return response.strip()


def _remove_accidental_code_fences(
    response: str,
) -> str:
    """
    Remove unnecessary surrounding markdown code fences.

    We do NOT remove legitimate code blocks in the middle of
    an answer.
    """

    cleaned = response.strip()

    if (
        cleaned.startswith("```")
        and cleaned.endswith("```")
    ):

        lines = cleaned.splitlines()

        if len(lines) >= 2:

            first_line = lines[0].strip()

            if first_line.startswith("```"):
                lines = lines[1:]

            if (
                lines
                and lines[-1].strip() == "```"
            ):
                lines = lines[:-1]

            cleaned = "\n".join(
                lines
            ).strip()

    return cleaned


def _normalize_whitespace(
    response: str,
) -> str:
    """
    Remove excessive blank lines while preserving normal
    paragraph structure.
    """

    response = response.replace(
        "\r\n",
        "\n",
    )

    response = re.sub(
        r"\n{4,}",
        "\n\n\n",
        response,
    )

    return response.strip()


def _limit_response_length(
    response: str,
    response_length: str,
) -> str:
    """
    Apply a conservative deterministic length policy.

    We intentionally do not aggressively truncate model output,
    because truncating code, JSON, tables, or explanations can
    make the result invalid.

    The main length control should happen through the provider's
    generation parameters/prompt instructions.
    """

    # No destructive truncation.
    return response.strip()


def adapt_response(
    response: str,
    preferences: dict | None = None,
) -> str:
    """
    Canonical deterministic output cleanup layer.

    This is APIL's first output-optimization stage.

    It:
        - removes hidden thinking/reasoning blocks
        - removes accidental outer code fences
        - normalizes whitespace
        - preserves useful content
        - avoids destructive truncation

    It does NOT call a GenAI provider.
    """

    if not isinstance(
        response,
        str,
    ):
        response = str(
            response
        )

    preferences = (
        preferences or {}
    )

    response_length = str(
        preferences.get(
            "response_length",
            "medium",
        )
    ).strip().lower()

    cleaned = response.strip()

    if not cleaned:
        return ""

    # --------------------------------------------------------------
    # 1. Remove hidden reasoning
    # --------------------------------------------------------------

    cleaned = _remove_thinking_blocks(
        cleaned
    )

    # --------------------------------------------------------------
    # 2. Remove unnecessary outer code fence
    # --------------------------------------------------------------

    cleaned = _remove_accidental_code_fences(
        cleaned
    )

    # --------------------------------------------------------------
    # 3. Normalize whitespace
    # --------------------------------------------------------------

    cleaned = _normalize_whitespace(
        cleaned
    )

    # --------------------------------------------------------------
    # 4. Apply response-length policy
    # --------------------------------------------------------------

    cleaned = _limit_response_length(
        cleaned,
        response_length,
    )

    return cleaned

