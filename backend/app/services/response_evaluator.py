from __future__ import annotations

import json
import re
from typing import Any


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return False

    if stripped.startswith("{") or stripped.startswith("["):
        try:
            json.loads(stripped)
            return True
        except json.JSONDecodeError:
            return False

    return False


def _count_bullets(text: str) -> int:
    return len(
        re.findall(
            r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+",
            text,
        )
    )


def _extract_required_format(dna: dict[str, Any]) -> str | None:
    explicit = (dna.get("output_format") or "").lower()

    if explicit:
        return explicit

    original = (dna.get("original_prompt") or "").lower()

    if "json" in original:
        return "json"

    if "bullet" in original:
        return "bullet points"

    if "table" in original:
        return "table"

    if "step by step" in original or "steps" in original:
        return "steps"

    if "code" in original:
        return "code"

    return None


def _has_format_violation(
    response: str,
    expected_format: str | None,
) -> bool:
    if not expected_format:
        return False

    text = (response or "").strip()
    expected = expected_format.lower()

    if expected == "json":
        return not _looks_like_json(text)

    if expected == "bullet points":
        return (
            _count_bullets(text) == 0
            and "- " not in text
            and "* " not in text
        )

    if expected == "table":
        return "|" not in text and "column" not in text.lower()

    if expected == "steps":
        return not re.search(
            r"\b(step|first|second|next|finally)\b",
            text.lower(),
        )

    if expected == "code":
        return not any(
            token in text.lower()
            for token in (
                "def ",
                "class ",
                "function ",
                "import ",
                "return ",
                "```",
            )
        )

    return False


def _has_reasoning_leakage(response: str) -> bool:
    """
    Detect model planning/reasoning that has leaked into the final answer.

    This is intentionally structural and limited. It is not intended to
    classify the user's topic or act as a large keyword classifier.
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

    meta_line = re.compile(
        r"^(?:"
        r"okay\b|"
        r"alright\b|"
        r"hmm\b|"
        r"let me\b|"
        r"i need to\b|"
        r"i should\b|"
        r"i will\b|"
        r"i'll\b|"
        r"the user\b|"
        r"we need to\b|"
        r"to answer\b|"
        r"first,?\s+let(?:'s| us)\b|"
        r"thinking about\b|"
        r"considering\b|"
        r"final idea\b|"
        r"let me structure\b|"
        r"check length\b|"
        r"alternative example\b"
        r")",
        re.IGNORECASE,
    )

    meta_lines = sum(
        bool(meta_line.match(line))
        for line in lines
    )

    if meta_lines >= 2:
        return True

    if len(lines) >= 4 and meta_lines / len(lines) >= 0.5:
        return True

    return False


def _has_planning_structure(response: str) -> bool:
    """
    Detect a response that is predominantly describing how to construct
    an answer instead of actually answering the user.
    """

    text = (response or "").strip()

    if not text:
        return True

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    if len(paragraphs) < 2:
        return False

    planning_patterns = [
        r"\bthe user wants\b",
        r"\bthe user asks\b",
        r"\blet me structure\b",
        r"\bfinal idea\b",
        r"\bcheck length\b",
        r"\bi need to\b",
        r"\bi should\b",
        r"\bi will\b",
        r"\blet me\b",
        r"\bmaybe say\b",
        r"\balternative example\b",
    ]

    planning_paragraphs = 0

    for paragraph in paragraphs:
        if any(
            re.search(pattern, paragraph, re.IGNORECASE)
            for pattern in planning_patterns
        ):
            planning_paragraphs += 1

    return planning_paragraphs >= 2


def _has_final_answer_validity_issue(response: str) -> bool:
    """
    Provider-independent final-answer validity check.

    Returns True when the response appears to contain model planning,
    reasoning, or answer-construction content instead of a clean answer.
    """

    return (
        _has_reasoning_leakage(response)
        or _has_planning_structure(response)
    )


def evaluate_response(
    original_prompt: str,
    optimized_prompt: str,
    response: str,
    prompt_dna: dict[str, Any] | None = None,
    preferences: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the provider response against the actual user request."""

    dna = prompt_dna or {}
    preferences = preferences or {}

    cleaned = (response or "").strip()

    issues: list[str] = []
    missing_requirements: list[str] = []

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

    # ---------------------------------------------------------
    # Final-answer validity
    # ---------------------------------------------------------

    final_answer_validity_issue = _has_final_answer_validity_issue(
        cleaned
    )

    if final_answer_validity_issue:
        issues.append(
            "The response contains reasoning or planning content "
            "instead of only the final answer."
        )

    # ---------------------------------------------------------
    # Explicit constraints
    # ---------------------------------------------------------

    explicit_constraints = (
        list(dna.get("constraints") or [])
        + list(dna.get("special_instructions") or [])
    )

    for constraint in explicit_constraints:
        constraint_lower = str(constraint).lower()

        if any(
            token in constraint_lower
            for token in (
                "do not use sort",
                "without using sort",
                "avoid sort",
                "do not use sorted",
            )
        ):
            if "sort(" in response_lower or "sorted(" in response_lower:
                issues.append(
                    "The response violates the explicit constraint "
                    "against using sort()."
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
                    "The response does not match the requested JSON "
                    "output format."
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
                re.findall(r"\b\w+\b", cleaned)
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
                    "The response does not have the requested "
                    "five bullet points."
                )

    # ---------------------------------------------------------
    # Output format
    # ---------------------------------------------------------

    expected_format = _extract_required_format(dna)

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

    required_language = (
        dna.get("language")
        or preferences.get("language")
        or ""
    ).strip()

    if required_language:
        required_language_lower = required_language.lower()

        if required_language_lower in {
            "spanish",
            "french",
            "german",
            "italian",
            "portuguese",
            "japanese",
            "korean",
            "chinese",
            "hindi",
        }:
            # Keep this as a lightweight signal only.
            # The evaluator should not assume a language merely
            # because the language name appears in the answer.
            pass

        elif (
            required_language_lower == "english"
            and "in english" in (original_prompt or "").lower()
        ):
            pass

    # ---------------------------------------------------------
    # Response length
    # ---------------------------------------------------------

    length_pref = (
        preferences.get("response_length")
        or dna.get("desired_length")
    )

    word_count = len(cleaned.split())

    if length_pref == "short" and word_count > 180:
        issues.append(
            "The response is longer than the requested short length."
        )

    elif length_pref == "long" and word_count < 80:
        issues.append(
            "The response is shorter than the requested detailed length."
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
            "like",
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

    if dna.get("missing_information"):
        missing_requirements.extend(
            str(item)
            for item in dna["missing_information"]
        )

    # ---------------------------------------------------------
    # Quality calculation
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

        relevance = (
            "good"
            if not any(
                "not relevant" in issue.lower()
                for issue in issues
            )
            else "poor"
        )

        completeness = (
            "good"
            if not missing_requirements
            else "partial"
        )

        instruction_following = (
            "partial"
            if issues or missing_requirements
            else "good"
        )

        format_compliance = (
            "partial"
            if any(
                "format" in issue.lower()
                for issue in issues
            )
            else "good"
        )

        constraint_compliance = (
            "partial"
            if issues or missing_requirements
            else "good"
        )

        clarity = (
            "partial"
            if final_answer_validity_issue
            else "good"
        )

        audience_fit = "good"

        length_fit = (
            "good"
            if not any(
                "length" in issue.lower()
                for issue in issues
            )
            else "partial"
        )

        final_answer_validity = (
            "poor"
            if final_answer_validity_issue
            else "good"
        )

        improvement_needed = True

        improvement_instructions = [
            "Fix only the identified issues and missing requirements.",
            "Preserve the original user intent and explicit constraints.",
            "Return only the final answer, not planning or reasoning.",
            "Do not describe how the answer is being constructed.",
            "Do not add unrelated requirements or unsupported content.",
        ]

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
        "issues": issues,
        "missing_requirements": missing_requirements,
        "improvement_needed": improvement_needed,
        "improvement_instructions": improvement_instructions,
    }
