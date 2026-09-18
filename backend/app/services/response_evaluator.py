from __future__ import annotations

import re
from typing import Any


def evaluate_response(
    original_prompt: str,
    optimized_prompt: str,
    response: str,
    prompt_dna: dict[str, Any] | None = None,
    preferences: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a provider response against the actual request intent and constraints."""

    dna = prompt_dna or {}
    preferences = preferences or {}
    cleaned = (response or "").strip()
    issues: list[str] = []
    missing_requirements: list[str] = []
    instructions: list[str] = []

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

    original_lower = original_prompt.lower()
    response_lower = cleaned.lower()

    if dna.get("intent") and dna["intent"] == "comparison":
        if not any(word in response_lower for word in ["compare", "difference", "versus", "vs", "similarity", "tradeoff"]):
            issues.append("Comparison response lacks explicit comparison framing.")

    if dna.get("intent") == "code_generation":
        code_reqs = dna.get("code_requirements") or []
        if code_reqs and not any(token in response_lower for token in ["def ", "class ", "print(", "return ", "import "]):
            missing_requirements.append("Code output is expected but not clearly present.")

    constraints = dna.get("constraints") or []
    for constraint in constraints:
        normalized = constraint.lower()
        if "do not use sort" in normalized or "without using sort" in normalized:
            if "sort(" not in response_lower and "sorted(" not in response_lower:
                missing_requirements.append("The response should avoid using sort().")
        if "briefly" in normalized or "brief" in normalized:
            words = len(re.findall(r"\b\w+\b", cleaned))
            if words > 250:
                issues.append("The response is longer than the requested brief length.")
        if "simple language" in normalized or "beginner-friendly" in normalized:
            if len(cleaned.split()) > 350:
                issues.append("The response may be too long for a beginner-oriented answer.")

    if dna.get("output_format"):
        expected = str(dna["output_format"]).lower()
        if expected in {"bullet points", "bullets", "list"} and not any(marker in response_lower for marker in ["- ", "* ", "1.", "bullet"]):
            issues.append("The answer does not match the requested bullet format.")
        if expected in {"json"} and not response_lower.strip().startswith("{"):
            issues.append("The answer does not match the requested JSON format.")
        if expected in {"table"} and "|" not in cleaned and "column" not in response_lower:
            issues.append("The answer does not match the requested table format.")
        if expected in {"code"} and not any(token in response_lower for token in ["def ", "print(", "import ", "class "]):
            issues.append("The answer does not contain code output where requested.")

    length_pref = preferences.get("response_length") or dna.get("desired_length")
    if length_pref == "short":
        if len(cleaned.split()) > 180:
            issues.append("The response is longer than the requested short length.")
    elif length_pref == "long":
        if len(cleaned.split()) < 80:
            issues.append("The response is shorter than the requested long-form depth.")

    if dna.get("requested_examples") and not any(marker in response_lower for marker in ["example", "for example", "e.g.", "illustration"]):
        missing_requirements.append("An example was requested but is missing.")

    if dna.get("missing_information"):
        missing_requirements.extend(list(dna["missing_information"]))

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
        relevance = "good" if "not relevant" not in " ".join(issues).lower() else "poor"
        completeness = "good" if not missing_requirements else "partial"
        instruction_following = "partial" if issues or missing_requirements else "good"
        format_compliance = "partial" if any("format" in issue.lower() for issue in issues) else "good"
        constraint_compliance = "partial" if missing_requirements else "good"
        clarity = "good" if not any("unclear" in issue.lower() for issue in issues) else "partial"
        audience_fit = "good"
        length_fit = "good" if not any("length" in issue.lower() for issue in issues) else "partial"
        improvement_needed = True
        improvement_instructions = [
            "Fix the identified issues without changing the original intent.",
            "Preserve the user's requested constraints, format, and tone.",
            "Do not add unsupported facts or hidden reasoning.",
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
