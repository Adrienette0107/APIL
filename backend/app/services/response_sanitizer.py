import re

from collections.abc import Mapping
from typing import Any, Final


REASONING_TAGS: Final[tuple[str, ...]] = (
    "think",
    "thinking",
    "analysis",
    "reasoning",
    "chain-of-thought",
)


# Explicit reasoning/planning lines.
# These are intentionally targeted so normal explanations such as
# "First, machine learning..." are not automatically removed.
_REASONING_LINE = re.compile(
    r"^\s*(?:"
    r"okay|"
    r"alright|"
    r"hmm|"
    r"let me|"
    r"i need to|"
    r"i should|"
    r"i will|"
    r"i'll|"
    r"the user(?: wants| asks| is asking| needs| requested)|"
    r"we need to|"
    r"to answer|"
    r"thinking about|"
    r"considering"
    r")\b[\s,:-]*",
    re.IGNORECASE,
)


# Planning statements that may appear in the middle of an otherwise
# valid answer.
_PLANNING_SENTENCE = re.compile(
    r"(?is)"
    r"(?:"
    r"\bI should (?:start|begin|explain|mention|define)\b[^.!?]*[.!?]"
    r"|"
    r"\bI need to (?:keep|make|explain|define|mention|start|begin)\b[^.!?]*[.!?]"
    r"|"
    r"\bI need to\b[^.!?]*[.!?]"
    r"|"
    r"\bMaybe (?:I should|we should|compare|use|explain|start|say)\b[^.!?]*[.!?]"
    r"|"
    r"\bLet me (?:explain|start|define|break|structure)\b[^.!?]*[.!?]"
    r"|"
    r"\bThe user (?:wants|asked|needs|requested)\b[^.!?]*[.!?]"
    r"|"
    r"\bThe answer should\b[^.!?]*[.!?]"
    r")"
)


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)

    return getattr(value, name, None)


def _extract_content(value: Any) -> str:
    """
    Extract only the provider's user-facing content.

    A separate thinking/reasoning field is intentionally ignored.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    message = _field(value, "message")

    if message is not None:
        content = _field(message, "content")

        if isinstance(content, str):
            return content

    for name in ("content", "output", "response", "text"):
        content = _field(value, name)

        if isinstance(content, str):
            return content

    return ""


def _remove_explicit_reasoning_blocks(text: str) -> str:
    """
    Remove explicit provider reasoning formats such as:

    <think>...</think>
    <analysis>...</analysis>
    [begin think] ... [end think]
    <|begin_of_thought|> ... <|end_of_thought|>
    ```thinking
    ...
    ```
    """

    for tag in REASONING_TAGS:
        # Complete tagged block.
        text = re.sub(
            rf"(?is)<{tag}\b[^>]*>.*?</{tag}>",
            " ",
            text,
        )

        # Unclosed reasoning block.
        text = re.sub(
            rf"(?is)<{tag}\b[^>]*>.*$",
            " ",
            text,
        )

        # Remaining opening/closing tags.
        text = re.sub(
            rf"(?is)</?{tag}\b[^>]*>",
            " ",
            text,
        )

    # [begin think] / [end think]
    text = re.sub(
        r"(?is)\[(?:begin|end)\s+"
        r"(?:think|thinking|analysis|reasoning)\]",
        " ",
        text,
    )

    # <|begin_of_thought|> ... <|end_of_thought|>
    text = re.sub(
        r"(?is)"
        r"<\|begin_of_(?:thought|thinking|analysis|reasoning)\|>"
        r".*?"
        r"<\|end_of_(?:thought|thinking|analysis|reasoning)\|>",
        " ",
        text,
    )

    # Remaining special tokens.
    text = re.sub(
        r"(?is)"
        r"<\|(?:begin|end)_of_(?:thought|thinking|analysis|reasoning)\|>",
        " ",
        text,
    )

    # Markdown reasoning blocks.
    text = re.sub(
        r"(?is)"
        r"```(?:think|thinking|analysis|reasoning)\s*.*?```",
        " ",
        text,
    )

    return text


def _strip_unmarked_reasoning_preamble(text: str) -> str:
    """
    Remove a clearly identifiable planning preamble.

    We only remove it when at least two planning lines are detected.
    This prevents legitimate answers beginning with words such as
    "First" from being deleted.
    """

    lines = text.splitlines()

    leading_meta = 0
    leading_end = 0

    for index, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            leading_end = index + 1
            continue

        if _REASONING_LINE.match(stripped):
            leading_meta += 1
            leading_end = index + 1
            continue

        if re.match(
            r"^\s*(?:another idea|wait|maybe|the answer should)\b",
            stripped,
            re.IGNORECASE,
        ):
            leading_meta += 1
            leading_end = index + 1
            continue

        break

    if leading_meta >= 2:
        remaining = lines[leading_end:]
        candidate = "\n".join(remaining).strip()

        if candidate:
            return candidate

    return text


def _remove_planning_sentences(text: str) -> str:
    """
    Remove obvious answer-construction commentary from an otherwise
    valid response.

    This is deliberately conservative.
    """

    cleaned = _PLANNING_SENTENCE.sub(" ", text)

    return cleaned


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    return text.strip()


def sanitize_model_output(
    content: Any,
    provider: str | None = None,
) -> str:
    """
    Convert provider output into clean user-facing text.

    Pipeline:

        provider output
            ↓
        extract content
            ↓
        remove explicit reasoning blocks
            ↓
        remove obvious planning commentary
            ↓
        normalize whitespace
            ↓
        return final text

    Normal explanatory content is preserved.
    """

    text = _extract_content(content)

    if not text:
        return ""

    # 1. Remove explicit reasoning/thinking formats.
    text = _remove_explicit_reasoning_blocks(text)

    # 2. Remove clearly meta-level leading planning.
    text = _strip_unmarked_reasoning_preamble(text)

    # 3. Remove obvious planning sentences appearing inside the answer.
    text = _remove_planning_sentences(text)

    # 4. Normalize whitespace.
    text = _normalize_whitespace(text)

    return text
