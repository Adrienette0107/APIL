from dataclasses import dataclass


@dataclass
class PromptAnalysis:
    intent: str
    complexity: str
    domain: str
    output_type: str


def analyze_prompt(prompt: str) -> PromptAnalysis:

    prompt_lower = prompt.lower()

    if any(
        word in prompt_lower
        for word in ["explain", "what is", "define", "meaning"]
    ):
        intent = "explanation"

    elif any(
        word in prompt_lower
        for word in ["write", "create", "generate", "make"]
    ):
        intent = "generation"

    elif any(
        word in prompt_lower
        for word in ["compare", "difference", "versus", "vs"]
    ):
        intent = "comparison"

    elif any(
        word in prompt_lower
        for word in ["solve", "calculate", "find"]
    ):
        intent = "problem_solving"

    else:
        intent = "general"

    word_count = len(prompt.split())

    if word_count < 15:
        complexity = "low"
    elif word_count < 50:
        complexity = "medium"
    else:
        complexity = "high"

    if any(
        word in prompt_lower
        for word in ["python", "program", "code", "api", "software"]
    ):
        domain = "technology"

    elif any(
        word in prompt_lower
        for word in ["math", "calculate", "equation"]
    ):
        domain = "mathematics"

    else:
        domain = "general"

    output_type = intent

    return PromptAnalysis(
        intent=intent,
        complexity=complexity,
        domain=domain,
        output_type=output_type,
    )
