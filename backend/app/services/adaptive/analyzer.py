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
        "spanish": ("en español", "in spanish", "spanish"),
        "french": ("en français", "in french", "french"),
        "german": ("in german", "german"),
        "italian": ("in italian", "italian"),
        "portuguese": ("in portuguese", "portuguese"),
        "japanese": ("in japanese", "japanese"),
        "korean": ("in korean", "korean"),
        "chinese": ("in chinese", "chinese"),
        "hindi": ("in hindi", "hindi"),
    }
    for name, patterns in language_patterns.items():
        if any(pattern in prompt_lower for pattern in patterns):
            return name.title()
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


def extract_prompt_dna(
    prompt: str,
    preferences: dict[str, Any] | None = None,
) -> PromptDNA:
    """Extract usable prompt signals without relying on a massive keyword catalog."""

    original = prompt.strip()
    if not original:
        raise ValueError("Prompt cannot be empty.")

    analysis = analyze_prompt(original)
    lower = original.lower()
    explicit_preferences = preferences or {}

    intent = analysis.intent
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
        "classify",
        "analyze",
        "find",
    ):
        if re.search(rf"\b{re.escape(verb)}\b", lower):
            task = verb
            break

    if task is None and intent == "code_generation":
        task = "build"

    subject = _extract_subject(original, task)
    if not subject:
        subject = original

    context = None
    if "for a college project" in lower or "college project" in lower:
        context = "college project"
    elif "for my boss" in lower or "for work" in lower:
        context = "work context"
    elif "for a 10-year-old" in lower or "10-year-old" in lower:
        context = "beginner audience"

    audience = explicit_preferences.get("level") or None
    if audience is None:
        for label in ("beginner", "intermediate", "advanced"):
            if label in lower:
                audience = label
                break
        if audience is None and re.search(r"\b\d+\s*-?year-old\b|\bchild\b|\bteen\b", lower):
            audience = "beginner"

    expertise_level = audience
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
        elif any(phrase in lower for phrase in ("step by step", "detailed", "thoroughly", "in depth")):
            desired_depth = "detailed"

    desired_length = explicit_preferences.get("response_length")
    if desired_length is None:
        if any(phrase in lower for phrase in ("briefly", "short answer", "in short", "concise")):
            desired_length = "short"
        elif any(phrase in lower for phrase in ("detailed", "comprehensive", "long-form", "in depth")):
            desired_length = "long"

    output_format = None
    if "example" in lower and intent == "explanation":
        output_format = "explanation + example"
    for format_name, marker in (
        ("bullet points", ("bullet points", "bullets", "bullet list", "five bullet points")),
        ("steps", ("step by step", "steps", "procedure")),
        ("table", ("table", "comparison table")),
        ("json", ("json", "json output")),
        ("code", ("code", "program", "script", "runnable code")),
        ("essay", ("essay", "paragraph", "write an essay")),
        ("summary", ("summary", "summarize")),
    ):
        if output_format is None and any(item in lower for item in marker):
            output_format = format_name
            break

    constraints: list[str] = []
    if "without using sort" in lower or "do not use sort" in lower or "without sort()" in lower:
        constraints.append("Do not use sort().")
    if "briefly" in lower or "brief" in lower:
        constraints.append("Keep the response brief.")
    if "simple" in lower or "plain english" in lower or "beginner-friendly" in lower:
        constraints.append("Use simple language appropriate for a beginner.")
    if "complete runnable code" in lower or "runnable code" in lower:
        constraints.append("Provide complete runnable code.")
    if "do not" in lower:
        for phrase in re.findall(r"do not\s+[^.?!]+", lower):
            constraints.append(f"Do not {phrase.replace('do not ', '').strip()}.")
    if "without" in lower:
        for phrase in re.findall(r"without\s+[^.?!]+", lower):
            constraints.append(f"Without {phrase.replace('without ', '').strip()}.")
    if "in five bullet points" in lower:
        constraints.append("Give exactly five bullet points.")
    if "in spanish" in lower:
        constraints.append("Respond in Spanish.")

    requirements = list(constraints)
    if output_format:
        requirements.append(f"Use {output_format} format.")
    if desired_length:
        requirements.append(f"Keep the response {desired_length} in length.")
    if audience:
        requirements.append(f"Tailor the response for a {audience} audience.")

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
    if not task or not subject or len(original.split()) < 4:
        ambiguity = "medium"
        missing_information.append("The task is underspecified.")
    if intent == "code_generation" and ("something useful" in lower or "something" in lower):
        ambiguity = "medium"
        missing_information.append("The exact deliverable and success criteria are not specified.")
    if not output_format and intent in {"comparison", "summarization", "translation"}:
        ambiguity = "medium"
        missing_information.append("The preferred output shape is not explicit.")
    if intent == "general" and len(original.split()) > 3:
        ambiguity = "medium"
        missing_information.append("The user request may need clarification to confirm the exact task.")

    special_instructions = []
    for phrase in re.findall(r"(?:do not|without|must|should|required)\s+[^.?!]+", lower):
        special_instructions.append(phrase.strip())

    return PromptDNA(
        original_prompt=original,
        intent=intent,
        task=task,
        subject=subject,
        topic=subject,
        domain=analysis.domain,
        context=context,
        audience=audience,
        expertise_level=expertise_level,
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

    if any(word in prompt_lower for word in ["write", "create", "generate", "draft", "compose", "build", "implement"]) or (
        any(word in prompt_lower for word in ["python", "javascript", "code", "program", "function", "class"]) and any(word in prompt_lower for word in ["build", "write", "create", "generate"])):
        intent = "code_generation"
    elif any(word in prompt_lower for word in ["compare", "difference", "versus", "vs", "tradeoff", "pros and cons"]):
        intent = "comparison"
    elif any(word in prompt_lower for word in ["summarize", "summary", "brief overview", "condense"]):
        intent = "summarization"
    elif any(word in prompt_lower for word in ["translate", "translation"]):
        intent = "translation"
    elif any(word in prompt_lower for word in ["explain", "what is", "define", "meaning", "how does", "why"]):
        intent = "explanation"
    elif any(word in prompt_lower for word in ["solve", "calculate", "find", "debug", "fix", "optimize"]):
        intent = "problem_solving"
    else:
        intent = "general"

    if word_count < 15:
        complexity = "low"
    elif word_count < 50:
        complexity = "medium"
    else:
        complexity = "high"

    if any(word in prompt_lower for word in ["python", "javascript", "typescript", "sql", "database", "api", "software", "program", "code", "server", "frontend"]):
        domain = "technology"
    elif any(word in prompt_lower for word in ["math", "calculate", "equation", "algebra", "statistics", "probability"]):
        domain = "mathematics"
    elif any(word in prompt_lower for word in ["biology", "chemistry", "physics", "medicine", "health"]):
        domain = "science"
    elif any(word in prompt_lower for word in ["history", "politics", "economics", "literature", "philosophy"]):
        domain = "humanities"
    else:
        domain = "general"

    output_type = intent
    return PromptAnalysis(
        intent=intent,
        complexity=complexity,
        domain=domain,
        output_type=output_type,
    )
