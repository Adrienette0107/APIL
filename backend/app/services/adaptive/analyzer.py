from dataclasses import asdict, dataclass
import re
from typing import Any


@dataclass
class PromptAnalysis:
    intent: str
    complexity: str
    domain: str
    output_type: str


@dataclass(frozen=True)
class PromptDNA:
    original_prompt: str
    intent: str | None = None
    task: str | None = None
    subject: str | None = None
    topic: str | None = None
    domain: str | None = None
    context: str | None = None
    audience: str | None = None
    expertise_level: str | None = None
    language: str | None = None
    tone: str | None = None
    desired_depth: str | None = None
    desired_length: str | None = None
    response_length: str | None = None
    output_format: str | None = None
    constraints: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    code_requirements: tuple[str, ...] = ()
    ambiguity: str | None = None
    missing_information: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    requested_examples: bool = False
    special_instructions: tuple[str, ...] = ()
    user_preferences: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _infer_language(prompt_lower: str) -> str | None:
    language_patterns = {
        "Spanish": ("en español", "in spanish", "spanish"),
        "French": ("en français", "in french", "french"),
        "German": ("in german", "german"),
        "Italian": ("in italian", "italian"),
        "Portuguese": ("in portuguese", "portuguese"),
        "Japanese": ("in japanese", "japanese"),
        "Korean": ("in korean", "korean"),
        "Chinese": ("in chinese", "chinese"),
        "Hindi": ("in hindi", "hindi"),
        "English": ("in english", "english"),
    }
    for name, patterns in language_patterns.items():
        if any(pattern in prompt_lower for pattern in patterns):
            return name
    return None


def _infer_context(prompt_lower: str) -> str | None:
    if any(phrase in prompt_lower for phrase in ("for a college project", "college project")):
        return "college project"
    if any(phrase in prompt_lower for phrase in ("for my boss", "for work", "professional context")):
        return "work context"
    if any(phrase in prompt_lower for phrase in ("for a 10-year-old", "10-year-old", "beginner audience")):
        return "beginner audience"
    return None


def _extract_subject(prompt: str, task: str | None) -> str | None:
    text = prompt.strip()
    if not text:
        return None
    if task:
        patterns = [
            rf"^\s*{re.escape(task)}\s+",
            rf"^\s*{re.escape(task)}\s*\([^\)]*\)\s+",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.I)
    text = re.sub(r"\s+(in|using|with|for|to)\s+.*$", "", text, flags=re.I)
    text = text.strip(" .;:-")
    return text or None


def _detect_entities(prompt: str) -> tuple[str, ...]:
    found: list[str] = []
    matches = re.findall(r"\b[A-Z][A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)?\b", prompt)
    for item in matches:
        if item.lower() not in {"i", "apil"}:
            found.append(item)
    for token in ("PostgreSQL", "MongoDB", "Python", "JavaScript", "SQL", "FastAPI"):
        if token.lower() in prompt.lower():
            found.append(token)
    seen: set[str] = set()
    unique: list[str] = []
    for item in found:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return tuple(unique[:10])


def _collect_constraints(prompt_lower: str) -> list[str]:
    constraints: list[str] = []

    if any(phrase in prompt_lower for phrase in ("without using sort", "do not use sort", "without sort()", "avoid sort")):
        constraints.append("Do not use sort().")
    if any(phrase in prompt_lower for phrase in ("briefly", "in short", "short answer", "concise")):
        constraints.append("Keep the response brief.")
    if any(phrase in prompt_lower for phrase in ("simple", "plain english", "beginner-friendly", "for a beginner")):
        constraints.append("Use simple language appropriate for the intended audience.")
    if any(phrase in prompt_lower for phrase in ("complete runnable code", "runnable code", "must run")):
        constraints.append("Provide complete runnable code.")
    if any(phrase in prompt_lower for phrase in ("valid json", "json output", "as json")):
        constraints.append("Return valid JSON.")
    if any(phrase in prompt_lower for phrase in ("five bullet points", "in five bullet points")):
        constraints.append("Give exactly five bullet points.")
    if any(phrase in prompt_lower for phrase in ("single sentence", "one sentence", "one-sentence")):
        constraints.append("Return a single sentence.")
    if any(phrase in prompt_lower for phrase in ("in spanish", "spanish")):
        constraints.append("Respond in Spanish.")

    for phrase in re.findall(r"(?:do not|don't|never|without|must|should|required)\s+[^.?!]+", prompt_lower):
        cleaned = phrase.strip()
        if cleaned and len(cleaned) < 140:
            constraints.append(cleaned.capitalize())

    return list(dict.fromkeys(constraints))


def extract_prompt_dna(
    prompt: str,
    preferences: dict[str, Any] | None = None,
) -> PromptDNA:
    """Extract usable prompt requirements without relying on fixed categories or a giant classifier."""

    original = prompt.strip()
    if not original:
        raise ValueError("Prompt cannot be empty.")

    lower = original.lower()
    explicit_preferences = preferences or {}

    intent = "general"
    code_indicators = ("write code", "create code", "build a program", "implement", "function", "class", "script", "program")
    if any(match in lower for match in code_indicators):
        intent = "code_generation"
    elif any(match in lower for match in ("compare", "versus", "vs", "difference between", "tradeoff")):
        intent = "comparison"
    elif any(match in lower for match in ("summarize", "summary", "brief overview")):
        intent = "summarization"
    elif any(match in lower for match in ("translate", "translation")):
        intent = "translation"
    elif any(match in lower for match in ("explain", "what is", "define", "how does", "why")):
        intent = "explanation"
    elif any(match in lower for match in ("solve", "calculate", "find", "debug", "fix", "optimize")):
        intent = "problem_solving"

    task = None
    for verb in (
        "build",
        "write",
        "create",
        "generate",
        "compare",
        "summarize",
        "translate",
        "explain",
        "describe",
        "solve",
        "debug",
        "design",
        "plan",
        "analyze",
        "find",
    ):
        if re.search(rf"\b{re.escape(verb)}\b", lower):
            task = verb
            break

    subject = _extract_subject(original, task)
    if not subject:
        subject = original

    audience = explicit_preferences.get("level")
    if audience is None:
        for label in ("beginner", "intermediate", "advanced"):
            if label in lower:
                audience = label
                break
        if audience is None and re.search(r"\b\d+\s*-?year-old\b|\bchild\b|\bteen\b", lower):
            audience = "beginner"

    language = explicit_preferences.get("language") or _infer_language(lower)
    tone = None
    for candidate in ("formal", "casual", "friendly", "professional", "neutral"):
        if candidate in lower:
            tone = candidate
            break

    desired_depth = explicit_preferences.get("desired_depth")
    if desired_depth is None:
        if any(phrase in lower for phrase in ("simple", "easy", "plain english", "beginner-friendly")):
            desired_depth = "simple"
        elif any(phrase in lower for phrase in ("step by step", "detailed", "thoroughly", "in depth", "deeply")):
            desired_depth = "detailed"

    desired_length = explicit_preferences.get("response_length")
    if desired_length is None:
        if any(phrase in lower for phrase in ("briefly", "short answer", "in short", "concise", "one sentence")):
            desired_length = "short"
        elif any(phrase in lower for phrase in ("detailed", "comprehensive", "long-form", "in depth")):
            desired_length = "long"

    output_format = None
    if any(phrase in lower for phrase in ("single sentence", "one sentence", "one-sentence")):
        output_format = "single sentence"
    if output_format is None and "example" in lower and intent == "explanation":
        output_format = "explanation + example"
    for format_name, markers in (
        ("json", ("json", "valid json", "as json")),
        ("bullet points", ("bullet points", "bullets", "bullet list", "list of items")),
        ("table", ("table", "comparison table")),
        ("steps", ("step by step", "steps", "procedure", "workflow")),
        ("code", ("code", "program", "script", "runnable code")),
        ("summary", ("summary", "summarize")),
        ("essay", ("essay", "paragraph", "prose")),
    ):
        if output_format is None and any(marker in lower for marker in markers):
            output_format = format_name
            break

    constraints = _collect_constraints(lower)
    requirements = list(constraints)
    if output_format:
        requirements.append(f"Use {output_format} format.")
    if desired_length:
        requirements.append(f"Keep the response {desired_length} in length.")
    if audience:
        requirements.append(f"Tailor the response for a {audience} audience.")
    if language:
        requirements.append(f"Respond in {language}.")

    code_requirements: list[str] = []
    if any(token in lower for token in ("python", "javascript", "code", "program", "script", "function")):
        code_requirements.append("Provide code that is directly usable and complete.")
        if any(token in lower for token in ("without using sort", "no external libraries", "without libraries")):
            code_requirements.append("Respect the explicit coding constraints in the prompt.")
    if "runnable" in lower:
        code_requirements.append("Ensure the code is runnable as written.")

    requested_examples = any(phrase in lower for phrase in ("example", "examples", "for example", "e.g."))
    missing_information: list[str] = []
    assumptions: list[str] = []
    ambiguity = "low"
    if (not task or not subject) and len(original.split()) < 4:
        ambiguity = "medium"
        missing_information.append("The task is underspecified.")
    if "something useful" in lower or ("something" in lower and "build" in lower):
        ambiguity = "medium"
        missing_information.append("The exact deliverable and success criteria are not specified.")
    if not output_format and not constraints and len(original.split()) >= 6 and not "?" in original:
        ambiguity = "medium"
        missing_information.append("The preferred output shape is not explicit.")

    special_instructions = []
    for phrase in re.findall(r"(?:do not|without|must|should|required)\s+[^.?!]+", lower):
        special_instructions.append(phrase.strip())

    return PromptDNA(
        original_prompt=original,
        intent=intent,
        task=task,
        subject=subject,
        topic=subject,
        domain="general",
        context=_infer_context(lower),
        audience=audience,
        expertise_level=audience,
        language=language,
        tone=tone,
        desired_depth=desired_depth,
        desired_length=desired_length,
        response_length=desired_length,
        output_format=output_format,
        constraints=tuple(dict.fromkeys(constraints)),
        requirements=tuple(dict.fromkeys(requirements)),
        entities=tuple(_detect_entities(original)),
        code_requirements=tuple(dict.fromkeys(code_requirements)),
        ambiguity=ambiguity,
        missing_information=tuple(dict.fromkeys(missing_information)),
        assumptions=tuple(dict.fromkeys(assumptions)),
        requested_examples=requested_examples,
        special_instructions=tuple(dict.fromkeys(special_instructions)),
        user_preferences=explicit_preferences or None,
    )


def analyze_prompt(prompt: str) -> PromptAnalysis:
    prompt_lower = prompt.lower()
    word_count = len(prompt.split())

    intent = "general"
    for candidate, matches in (
        ("code_generation", ("write code", "python", "javascript", "script", "function", "class", "program")),
        ("comparison", ("compare", "versus", "vs", "difference between", "tradeoff")),
        ("summarization", ("summarize", "summary", "brief overview")),
        ("translation", ("translate", "translation")),
        ("explanation", ("explain", "what is", "define", "how does", "why")),
        ("problem_solving", ("solve", "calculate", "find", "debug", "fix", "optimize")),
    ):
        if any(match in prompt_lower for match in matches):
            intent = candidate
            break

    if word_count < 15:
        complexity = "low"
    elif word_count < 50:
        complexity = "medium"
    else:
        complexity = "high"

    if any(word in prompt_lower for word in ("python", "javascript", "typescript", "sql", "database", "api", "software", "server", "frontend")):
        domain = "technology"
    elif any(word in prompt_lower for word in ("math", "calculate", "equation", "algebra", "statistics", "probability")):
        domain = "mathematics"
    elif any(word in prompt_lower for word in ("biology", "chemistry", "physics", "medicine", "health")):
        domain = "science"
    elif any(word in prompt_lower for word in ("history", "politics", "economics", "literature", "philosophy")):
        domain = "humanities"
    else:
        domain = "general"

    return PromptAnalysis(
        intent=intent,
        complexity=complexity,
        domain=domain,
        output_type=intent,
    )
