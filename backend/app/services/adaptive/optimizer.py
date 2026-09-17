from backend.app.services.adaptive.analyzer import PromptAnalysis


def build_optimized_prompt(
    prompt: str,
    analysis: PromptAnalysis,
    preferences: dict | None = None,
    has_context: bool = False,
) -> str:
    """Build the provider-facing prompt from APIL's analysis and preferences."""

    preferences = preferences or {}
    language = preferences.get("language", "English")
    level = preferences.get("level", "beginner")
    response_length = preferences.get("response_length", "medium")

    instructions = [
        f"Answer the user's request in {language}.",
        f"Tailor the explanation to a {level} audience.",
        f"Keep the response {response_length} in length.",
    ]

    if analysis.intent == "explanation":
        instructions.extend(
            [
                "Start with a simple definition.",
                "Explain the key ideas clearly.",
                "Include a practical example when it improves understanding.",
                "Avoid unnecessary technical terminology.",
            ]
        )
    elif analysis.intent == "comparison":
        instructions.append("Use a clear comparison of the important differences.")
    elif analysis.intent == "problem_solving":
        instructions.append("Show the solution steps and state the final result clearly.")
    elif analysis.intent == "generation":
        instructions.append("Return a complete, usable result that follows the request.")

    if analysis.output_type == "explanation":
        instructions.append("Prefer short sections or bullet points for readability.")

    if has_context:
        instructions.append("Use the relevant preceding conversation context without repeating it unnecessarily.")

    return "\n".join(
        [
            "APIL optimized request:",
            prompt.strip(),
            "",
            "Response requirements:",
            *[f"- {instruction}" for instruction in instructions],
        ]
    )


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
