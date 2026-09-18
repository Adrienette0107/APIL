import re
from typing import Final

REASONING_TAGS: Final[tuple[str, ...]] = (
    "think",
    "thinking",
    "analysis",
    "reasoning",
    "chain-of-thought",
)


def sanitize_model_output(content: str | None, provider: str | None = None) -> str:
    """Remove reasoning/thinking blocks from user-facing model output.

    This is provider-independent and only strips explicit reasoning sections.
    Normal answer text is preserved.
    """
    if content is None:
        return ""

    text = str(content)

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

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = text.strip()

    return text
