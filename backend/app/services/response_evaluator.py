from __future__ import annotations

import json
import re
from typing import Any


def _looks_like_json(text: str) -> bool:
    stripped = (text or "").strip()

    if not stripped:
        return False

    if not (
        stripped.startswith("{")
        or stripped.startswith("[")
    ):
        return False

    try:
        json.loads(stripped)
        return True
    except json.JSONDecodeError:
        return False


def _count_bullets(text: str) -> int:
    if not text:
        return 0

    pattern = re.compile(
        r"(?m)^\s*(?:[-*•]|\d+[.)])\s+"
    )

    return len(pattern.findall(text))


def _extract_required_format(
    dna: dict[str, Any],
) -> str | None:
    explicit = str(
        dna.get("output_format") or ""
    ).strip().lower()

    if explicit:
        return explicit

    original = str(
        dna.get("original_prompt") or ""
    ).lower()

    if "json" in original:
        return "json"

    if "bullet" in original:
        return "bullet points"

    if "table" in original:
        return "table"

    if (
        "step by step" in original
        or "steps" in original
    ):
        return "steps"

    if (
        "write code" in original
        or "provide code" in original
        or "code" in original
    ):
        return "code"

    return None


def _has_format_violation(
    response: str,
    expected_format: str | None,
) -> bool:
    if not expected_format:
        return False

    text = (response or "").strip()
    expected = expected_format.lower().strip()

    if expected == "json":
        return not _looks_like_json(text)

    if expected in {
        "bullet",
        "bullets",
        "bullet points",
    }:
        return _count_bullets(text) == 0

    if expected == "table":
        lines = text.splitlines()

        has_pipe = any(
            "|" in line
            for line in lines
        )

        has_separator = any(
            re.search(
                r"\|\s*:?-{3,}:?\s*(?:\||$)",
                line,
            )
            for line in lines
        )

        return not (
            has_pipe
            and has_separator
        )

    if expected in {
        "steps",
        "step",
        "step-by-step",
    }:
        return not bool(
            re.search(
                r"(?im)^\s*(?:step\s+\d+|\d+[.)])\s+",
                text,
            )
        )

    if expected == "code":
        return not (
            "```" in text
            or re.search(
                r"(?m)^\s*(?:def |class |import |from )",
                text,
            )
        )

    return False


def _has_reasoning_leakage(
    response: str,
) -> bool:
    """
    Detect internal answer-planning/reasoning that leaked into the
    final provider response.

    Normal explanatory phrases such as "First, ..." are allowed.
    Meta-commentary about constructing the answer is rejected.
    """

    text = (response or "").strip()

    if not text:
        return False

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return False

    strong_patterns = [
        r"^wait[,!.]?\s+",
        r"^oh[,!.]?\s+",
        r"^hmm[,!.]?\s+",
        r"^okay[,!.]?\s+",
        r"^alright[,!.]?\s+",

        r"\bthe user wants\b",
        r"\bthe user asked\b",
        r"\bthe user said\b",
        r"\bthe user needs\b",
        r"\bthe user requested\b",

        r"\bi need to make\b",
        r"\bi need to\b",
        r"\bi should\b",
        r"\bi will\b",
        r"\bi need\b",

        r"\bmaybe start with\b",
        r"\bmaybe use\b",
        r"\bmaybe explain\b",
        r"\bmaybe compare\b",
        r"\bmaybe say\b",
        r"\bso maybe\b",

        r"\blet me\b",
        r"\blet's\b",
        r"\bhow should i answer\b",
        r"\bhow should i explain\b",

        r"\bneed to make it\b",
        r"\bmake it super simple\b",
        r"\bstructure the answer\b",
        r"\bstructure it\b",
        r"\bplan the answer\b",
        r"\bthink about\b",
        r"\bfigure out\b",
    ]

    matches = 0

    for line in lines:
        if any(
            re.search(
                pattern,
                line,
                re.IGNORECASE,
            )
            for pattern in strong_patterns
        ):
            matches += 1

    # One unmistakable planning phrase is enough.
    if matches >= 1:
        return True

    return False
def _has_planning_structure(
    response: str,
) -> bool:
    """
    Detect answers that primarily describe how to construct
    an answer instead of actually answering the user.
    """

    text = (response or "").strip()

    if not text:
        return True

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            text,
        )
        if paragraph.strip()
    ]

    if len(paragraphs) < 2:
        return False

    planning_patterns = [
        r"\bthe user wants\b",
        r"\bthe user asks\b",
        r"\blet me structure\b",
        r"\blet me plan\b",
        r"\bfinal idea\b",
        r"\bcheck length\b",
        r"\bi need to\b",
        r"\bi should\b",
        r"\bi will\b",
        r"\bmaybe say\b",
        r"\balternative example\b",
        r"\bhow should i answer\b",
    ]

    planning_paragraphs = 0

    for paragraph in paragraphs:
        if any(
            re.search(
                pattern,
                paragraph,
                re.IGNORECASE,
            )
            for pattern in planning_patterns
        ):
            planning_paragraphs += 1

    return planning_paragraphs >= 2


def _has_final_answer_validity_issue(
    response: str,
) -> bool:
    return (
        _has_reasoning_leakage(response)
        or _has_planning_structure(response)
        or _has_unfinished_output(response)
    )

def _has_thinking_tags(
    response: str,
) -> bool:
    return bool(
        re.search(
            r"<think\b|</think>",
            response or "",
            re.IGNORECASE,
        )
    )


def _has_provider_error_leakage(
    response: str,
) -> bool:
    text = (response or "").lower()

    error_markers = (
        "providerexecutionerror",
        "traceback (most recent call last)",
        "internal server error",
        "connection refused",
        "api key is invalid",
        "authentication failed",
        "rate limit exceeded",
    )

    return any(
        marker in text
        for marker in error_markers
    )


def _has_unfinished_output(
    response: str,
) -> bool:
    text = (response or "").strip()

    if not text:
        return True

    if text.endswith(("...", "…")):
        return True

    if re.search(
        r"\bto be continued\b",
        text,
        re.IGNORECASE,
    ):
        return True

    if text.count("```") % 2 != 0:
        return True

    incomplete_patterns = [
        r"\bso$",
        r"\bbut$",
        r"\band$",
        r"\bor$",
        r"\bbecause$",
        r"\bmaybe$",
        r"\bperhaps$",
        r"\bsuch as$",
        r"\bfor example:$",
        r"\bfor instance:$",
        r"\bso maybe$",
        r"\bmaybe a$",
        r"\bmaybe an$",
        r"\bmaybe the$",
        r"\blet me$",
        r"\bi need to$",
        r"\bwe need to$",
        r"\bthe user wants$",
        r"\bthe user said$",
    ]

    return any(
        re.search(
            pattern,
            text,
            re.IGNORECASE,
        )
        for pattern in incomplete_patterns
    )
def _has_repetition(
    response: str,
) -> bool:
    sentences = [
        sentence.strip().lower()
        for sentence in re.split(
            r"[.!?]\s+",
            response or "",
        )
        if len(sentence.strip()) > 20
    ]

    if len(sentences) < 4:
        return False

    unique_sentences = set(sentences)

    return (
        len(unique_sentences) / len(sentences)
        < 0.7
    )


def _get_explicit_constraints(
    dna: dict[str, Any],
) -> list[str]:
    constraints: list[str] = []

    for key in (
        "constraints",
        "special_instructions",
    ):
        value = dna.get(key)

        if isinstance(value, list):
            constraints.extend(
                str(item)
                for item in value
                if str(item).strip()
            )

        elif value:
            constraints.append(str(value))

    return constraints


def _build_improvement_instructions(
    issues: list[str],
    missing_requirements: list[str],
) -> list[str]:
    instructions = [
        "Return only the final answer to the original user request.",
        "Preserve the user's original intent.",
        "Preserve all explicit constraints.",
    ]

    if any(
        "reasoning" in issue.lower()
        or "planning" in issue.lower()
        for issue in issues
    ):
        instructions.extend(
            [
                "Remove all internal planning, reasoning, or "
                "answer-construction commentary.",
                "Do not mention the user, your reasoning process, "
                "or how you constructed the answer.",
            ]
        )

    if any(
        "json" in issue.lower()
        for issue in issues
    ):
        instructions.append(
            "Return valid JSON only, with no surrounding commentary."
        )

    if any(
        "bullet" in issue.lower()
        for issue in issues
    ):
        instructions.append(
            "Use the requested bullet-point format."
        )

    if any(
        "table" in issue.lower()
        for issue in issues
    ):
        instructions.append(
            "Return the requested information as a table."
        )

    if any(
        "step" in issue.lower()
        for issue in issues
    ):
        instructions.append(
            "Present the answer as clear numbered steps."
        )

    if any(
        "length" in issue.lower()
        for issue in issues
    ):
        instructions.append(
            "Adjust the answer to the requested response length."
        )

    instructions.extend(
        missing_requirements
    )

    return list(
        dict.fromkeys(instructions)
    )


def evaluate_response(
    original_prompt: str,
    optimized_prompt: str,
    response: str,
    prompt_dna: dict[str, Any] | None = None,
    preferences: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Evaluate the provider response against the user's request.

    This evaluator is deterministic and provider-independent.
    It does not generate or rewrite the response.
    """

    dna = prompt_dna or {}
    preferences = preferences or {}

    cleaned = (response or "").strip()

    if not cleaned:
        return {
            "passed": False,
            "overall_quality": "poor",
            "relevance": "poor",
            "completeness": "poor",
            "instruction_following": "poor",
            "format_compliance": "poor",
            "constraint_compliance": "poor",
            "clarity": "poor",
            "audience_fit": "poor",
            "length_fit": "poor",
            "final_answer_validity": "poor",
            "issues": ["Empty response."],
            "missing_requirements": [],
            "improvement_needed": True,
            "improvement_instructions": [
                "Return a valid answer for the original user request."
            ],
        }

    response_lower = cleaned.lower()

    issues: list[str] = []
    missing_requirements: list[str] = []

    if len(re.findall(r"\b\w+\b", cleaned)) < 2:
        issues.append(
            "The response is too short to satisfy the request."
        )

    # ---------------------------------------------------------
    # Final-answer validity
    # ---------------------------------------------------------

    final_answer_validity_issue = (
        _has_final_answer_validity_issue(
            cleaned
        )
    )

    if final_answer_validity_issue:
        issues.append(
            "The response is not a valid final answer. "
            "It contains internal reasoning, planning, "
            "answer-construction commentary, or unfinished content."
        )

    # ---------------------------------------------------------
    # Thinking leakage
    # ---------------------------------------------------------

    if _has_thinking_tags(cleaned):
        issues.append(
            "The response contains model thinking tags."
        )

    # ---------------------------------------------------------
    # Provider error leakage
    # ---------------------------------------------------------

    if _has_provider_error_leakage(cleaned):
        issues.append(
            "The response contains provider or infrastructure "
            "error information."
        )

    # ---------------------------------------------------------
    # Repetition
    # ---------------------------------------------------------

    if _has_repetition(cleaned):
        issues.append(
            "The response contains excessive sentence repetition."
        )

    # ---------------------------------------------------------
    # Incomplete output
    # ---------------------------------------------------------

    if _has_unfinished_output(cleaned):
        issues.append(
            "The response appears incomplete or unfinished."
        )

    # ---------------------------------------------------------
    # Explicit constraints
    # ---------------------------------------------------------

    explicit_constraints = _get_explicit_constraints(
        dna
    )

    for constraint in explicit_constraints:
        constraint_lower = constraint.lower()

        if any(
            token in constraint_lower
            for token in (
                "do not use sort",
                "without using sort",
                "avoid sort",
                "do not use sorted",
            )
        ):
            if (
                "sort(" in response_lower
                or "sorted(" in response_lower
            ):
                issues.append(
                    "The response violates the explicit "
                    "constraint against using sort()."
                )

        if any(
            token in constraint_lower
            for token in (
                "valid json",
                "return valid json",
                "json output",
            )
        ):
            if not _looks_like_json(cleaned):
                issues.append(
                    "The response does not match the requested "
                    "JSON output format."
                )

        if any(
            token in constraint_lower
            for token in (
                "keep the response brief",
                "briefly",
                "concise",
                "one sentence",
            )
        ):
            word_count = len(
                re.findall(
                    r"\b\w+\b",
                    cleaned,
                )
            )

            if word_count > 35:
                issues.append(
                    "The response is longer than the requested "
                    "brief length."
                )

        if any(
            token in constraint_lower
            for token in (
                "five bullet points",
                "exactly five bullet points",
            )
        ):
            if _count_bullets(cleaned) != 5:
                issues.append(
                    "The response does not contain the requested "
                    "five bullet points."
                )

    # ---------------------------------------------------------
    # Output format
    # ---------------------------------------------------------

    expected_format = _extract_required_format(
        dna
    )

    if _has_format_violation(
        cleaned,
        expected_format,
    ):
        issues.append(
            f"The response does not match the requested "
            f"{expected_format} format."
        )

    # ---------------------------------------------------------
    # Language
    # ---------------------------------------------------------

    required_language = str(
        dna.get("language")
        or preferences.get("language")
        or ""
    ).strip().lower()

    # Language classification is intentionally not performed
    # using simple keyword matching because that is unreliable.

    # ---------------------------------------------------------
    # Response length
    # ---------------------------------------------------------

    length_pref = (
        preferences.get("response_length")
        or dna.get("desired_length")
    )

    word_count = len(
        re.findall(
            r"\b\w+\b",
            cleaned,
        )
    )

    if (
        length_pref == "short"
        and word_count > 180
    ):
        issues.append(
            "The response is longer than the requested "
            "short length."
        )

    elif (
        length_pref == "long"
        and word_count < 80
    ):
        issues.append(
            "The response is shorter than the requested "
            "detailed length."
        )

    # ---------------------------------------------------------
    # Requested examples
    # ---------------------------------------------------------

    if dna.get("requested_examples"):
        example_markers = (
            "example",
            "for example",
            "e.g.",
            "illustration",
            "imagine",
        )

        if not any(
            marker in response_lower
            for marker in example_markers
        ):
            missing_requirements.append(
                "An example was requested but is missing."
            )

    # ---------------------------------------------------------
    # Missing information
    # ---------------------------------------------------------

    missing_information = dna.get(
        "missing_information"
    )

    if isinstance(
        missing_information,
        list,
    ):
        missing_requirements.extend(
            str(item)
            for item in missing_information
            if str(item).strip()
        )

    # ---------------------------------------------------------
    # Quality
    # ---------------------------------------------------------

    if not issues and not missing_requirements:
        overall_quality = "good"
        relevance = "good"
        completeness = "good"
        instruction_following = "good"
        format_compliance = "good"
        constraint_compliance = "good"
        clarity = "good"
        audience_fit = "good"
        length_fit = "good"
        final_answer_validity = "good"
        improvement_needed = False
        improvement_instructions: list[str] = []

    else:
        overall_quality = "needs_improvement"

        relevance = "good"

        completeness = (
            "good"
            if not missing_requirements
            else "partial"
        )

        instruction_following = "partial"

        format_compliance = (
            "partial"
            if any(
                keyword in issue.lower()
                for issue in issues
                for keyword in (
                    "format",
                    "json",
                    "bullet",
                    "table",
                )
            )
            else "good"
        )

        constraint_compliance = (
            "partial"
            if issues
            else "good"
        )

        clarity = (
            "partial"
            if final_answer_validity_issue
            else "good"
        )

        audience_fit = "good"

        length_fit = (
            "partial"
            if any(
                "length" in issue.lower()
                for issue in issues
            )
            else "good"
        )

        final_answer_validity = (
            "poor"
            if final_answer_validity_issue
            else "good"
        )

        improvement_needed = True

        improvement_instructions = (
            _build_improvement_instructions(
                issues,
                missing_requirements,
            )
        )

    passed = (
        not issues
        and not missing_requirements
        and final_answer_validity == "good"
    )

    return {
        "passed": passed,
        "overall_quality": overall_quality,
        "relevance": relevance,
        "completeness": completeness,
        "instruction_following": instruction_following,
        "format_compliance": format_compliance,
        "constraint_compliance": constraint_compliance,
        "clarity": clarity,
        "audience_fit": audience_fit,
        "length_fit": length_fit,
        "final_answer_validity": final_answer_validity,
        "requested_language": (
            required_language or None
        ),
        "requested_format": expected_format,
        "word_count": word_count,
        "issues": issues,
        "missing_requirements": missing_requirements,
        "improvement_needed": improvement_needed,
        "improvement_instructions": improvement_instructions,
    }

