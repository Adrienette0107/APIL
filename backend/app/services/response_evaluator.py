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
    return len(re.findall(r"(^|\n)\s*(?:[-*•]|\d+[.)])\s+", text))


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


def _has_format_violation(response: str, expected_format: str | None) -> bool:
    if not expected_format:
        return False
    text = (response or "").strip()
    expected = expected_format.lower()
    if expected == "json":
        return not _looks_like_json(text)
    if expected == "bullet points":
        return _count_bullets(text) == 0 and "- " not in text and "* " not in text
    if expected == "table":
        return "|" not in text and "column" not in text.lower()
    if expected == "steps":
        return not re.search(r"\b(step|first|second|next|finally)\b", text.lower())
    if expected == "code":
        return not any(token in text.lower() for token in ("def ", "class ", "function ", "import ", "return ", "```"))
    return False


def evaluate_response(
    original_prompt: str,
    optimized_prompt: str,
    response: str,
    prompt_dna: dict[str, Any] | None = None,
    preferences: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the provider response against the actual user request and applicable requirements."""

    dna = prompt_dna or {}
    preferences = preferences or {}
    cleaned = (response or "").strip()
    issues: list[str] = []
    missing_requirements: list[str] = []

    if not cleaned:
        issues.append("Empty response.")
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
            "issues": issues,
            "missing_requirements": missing_requirements,
            "improvement_needed": True,
            "improvement_instructions": ["Return a valid answer for the original user request."],
        }

    response_lower = cleaned.lower()

    explicit_constraints = list(dna.get("constraints") or []) + list(dna.get("special_instructions") or [])
    for constraint in explicit_constraints:
        constraint_lower = constraint.lower()
        if any(token in constraint_lower for token in ("do not use sort", "without using sort", "avoid sort", "do not use sorted")):
            if "sort(" in response_lower or "sorted(" in response_lower:
                issues.append("The response violates the explicit constraint against using sort().")
        if any(token in constraint_lower for token in ("valid json", "return valid json", "json output")):
            if not _looks_like_json(cleaned):
                issues.append("The response does not match the requested JSON output format.")
        if any(token in constraint_lower for token in ("keep the response brief", "briefly", "concise", "one sentence")):
            word_count = len(re.findall(r"\b\w+\b", cleaned))
            if word_count > 35:
                issues.append("The response is longer than the requested brief length.")
        if any(token in constraint_lower for token in ("five bullet points", "exactly five bullet points")):
            if _count_bullets(cleaned) != 5:
                issues.append("The response does not have the requested five bullet points.")

    expected_format = _extract_required_format(dna)
    if _has_format_violation(cleaned, expected_format):
        issues.append(f"The response does not match the requested {expected_format} format.")

    required_language = (dna.get("language") or preferences.get("language") or "").strip()
    if required_language:
        required_language_lower = required_language.lower()
        if required_language_lower in {"spanish", "french", "german", "italian", "portuguese", "japanese", "korean", "chinese", "hindi"}:
            if required_language_lower not in response_lower:
                missing_requirements.append(f"The response should be in {required_language}.")
        elif required_language_lower == "english" and "in english" in (original_prompt or "").lower():
            if "english" not in response_lower and "en" not in response_lower:
                missing_requirements.append("The response should be in English.")

    length_pref = preferences.get("response_length") or dna.get("desired_length")
    if length_pref == "short" and len(cleaned.split()) > 180:
        issues.append("The response is longer than the requested short length.")
    elif length_pref == "long" and len(cleaned.split()) < 80:
        issues.append("The response is shorter than the requested detailed length.")

    if dna.get("requested_examples") and not any(marker in response_lower for marker in ("example", "for example", "e.g.", "illustration")):
        missing_requirements.append("An example was requested but is missing.")

    if dna.get("missing_information"):
        missing_requirements.extend(str(item) for item in dna["missing_information"])

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
        improvement_needed = False
        improvement_instructions = []
    else:
        overall_quality = "needs_improvement"
        relevance = "good" if not any("not relevant" in issue.lower() for issue in issues) else "poor"
        completeness = "good" if not missing_requirements else "partial"
        instruction_following = "partial" if issues or missing_requirements else "good"
        format_compliance = "partial" if any("format" in issue.lower() for issue in issues) else "good"
        constraint_compliance = "partial" if issues or missing_requirements else "good"
        clarity = "good" if not any("unclear" in issue.lower() for issue in issues) else "partial"
        audience_fit = "good"
        length_fit = "good" if not any("length" in issue.lower() for issue in issues) else "partial"
        improvement_needed = True
        improvement_instructions = [
            "Fix only the identified issues and missing requirements.",
            "Preserve the original user intent and explicit constraints.",
            "Do not add unrelated requirements or unsupported content.",
        ]

    return {
        "passed": not issues and not missing_requirements,
        "overall_quality": overall_quality,
        "relevance": relevance,
        "completeness": completeness,
        "instruction_following": instruction_following,
        "format_compliance": format_compliance,
        "constraint_compliance": constraint_compliance,
        "clarity": clarity,
        "audience_fit": audience_fit,
        "length_fit": length_fit,
        "issues": issues,
        "missing_requirements": missing_requirements,
        "improvement_needed": improvement_needed,
        "improvement_instructions": improvement_instructions,
    }
