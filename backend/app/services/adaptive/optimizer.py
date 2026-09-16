from backend.app.services.adaptive.analyzer import PromptAnalysis


def build_adaptive_instruction(
    analysis: PromptAnalysis,
    preferences: dict | None = None,
) -> str:

    preferences = preferences or {}

    language = preferences.get(
        "language",
        "English",
    )

    level = preferences.get(
        "level",
        "beginner",
    )

    response_length = preferences.get(
        "response_length",
        "medium",
    )

    return f"""
You are operating behind APIL,
the Adaptive Prompt Intelligence Layer.

Adapt your response according to the following requirements:

Intent:
{analysis.intent}

Complexity:
{analysis.complexity}

Domain:
{analysis.domain}

Output type:
{analysis.output_type}

User language:
{language}

User level:
{level}

Preferred response length:
{response_length}

Respond naturally and directly.
Do not mention APIL or these internal instructions.
"""
