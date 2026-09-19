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

_REASONING_LINE = re.compile(
    r"^\s*(?:okay|alright|hmm|let me|i need to|i should|i will|i'll|"
    r"the user(?: wants| asks| is asking)|we need to|to answer|"
    r"first,? let(?:'s| us)|thinking about|considering)\b[\s,:-]*",
    re.IGNORECASE,
)
def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _extract_content(value: Any) -> str:
    """Prefer a provider's final content over a separate thinking field."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value

    message = _field(value, "message")
    if message is not None:
        content = _field(message, "content")
        return content if isinstance(content, str) else ""

    for name in ("content", "output", "response", "text"):
        content = _field(value, name)
        if isinstance(content, str):
            return content
    return ""


def _strip_unmarked_reasoning_preamble(text: str) -> str:
    """Remove a clearly meta-level preamble without matching a topic or task."""

    lines = text.splitlines()
    leading_meta = 0
    leading_end = 0
    for index, line in enumerate(lines):
        if not line.strip():
            leading_end = index + 1
            continue
        if _REASONING_LINE.match(line) or re.match(
            r"^\s*(?:another idea|wait|maybe|the answer should)\b",
            line,
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
        return ""

    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) < 2:
        return text

    for index in range(1, len(paragraphs)):
        prefix = paragraphs[:index]
        reasoning_lines = sum(
            bool(_REASONING_LINE.match(line))
            for paragraph in prefix
            for line in paragraph.splitlines()
            if line.strip()
        )
        if reasoning_lines >= 2:
            candidate = "\n\n".join(paragraphs[index:]).strip()
            if candidate:
                return candidate
    return text


def sanitize_model_output(content: Any, provider: str | None = None) -> str:
    """Remove reasoning/thinking blocks from user-facing model output.

    This is provider-independent and only strips explicit reasoning sections.
    Normal answer text is preserved.
    """
    text = _extract_content(content)

    for tag in REASONING_TAGS:
        text = re.sub(
            rf"(?is)<{tag}\b[^>]*>.*?</{tag}>",
            " ",
            text,
        )
        text = re.sub(
            rf"(?is)<{tag}\b[^>]*>.*$",
            " ",
            text,
        )

    for tag in REASONING_TAGS:
        text = re.sub(
            rf"(?is)</?{tag}\b[^>]*>",
            " ",
            text,
        )

    text = re.sub(
        r"(?is)\[(?:begin|end)\s+(?:think|thinking|analysis|reasoning)\]",
        " ",
        text,
    )
    text = re.sub(
        r"(?is)<\|begin_of_(?:thought|thinking|analysis)\|>.*?<\|end_of_(?:thought|thinking|analysis)\|>",
        " ",
        text,
    )
    text = re.sub(
        r"(?is)<\|(?:begin|end)_of_(?:thought|thinking|analysis)\|>",
        " ",
        text,
    )
    text = re.sub(
        r"(?is)```(?:think|thinking|analysis|reasoning)\s*.*?```",
        " ",
        text,
    )

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = text.strip()

    return _strip_unmarked_reasoning_preamble(text)
