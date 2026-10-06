"""Make chatbot text speakable: drop markdown, links and emoji before synthesis.

LLM replies are written for screens: ``**bold**``, ``# headings``, ``- bullets``,
``[links](https://...)`` and emoji. A TTS model reads those symbols aloud or stumbles
over them. ``normalize_for_speech`` is applied to each chunk *after* it has been cut
(``app/core/chunker.py`` keeps markdown closers on a chunk), so it always sees whole
text, even when the reply arrives token by token.

It only removes and simplifies; numbers, currency and abbreviations are left to the
TTS model's own text front end, which handles them.
"""

from __future__ import annotations

import re
import unicodedata

_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]*)\)")  # [text](url) -> text
_URL = re.compile(r"(?:https?://|www\.)[^\s)>\]]+", re.IGNORECASE)
_LINE_MARKER = re.compile(r"^\s*(?:#{1,6}\s+|>\s*|[-*+•]\s+|\d{1,3}[.)]\s+)")
_EMPHASIS = re.compile(r"(\*{1,3}|_{2,3}|~~|`+)")
# _italic_ wrapping a word, but not the underscore inside snake_case.
_UNDERSCORE_ITALIC = re.compile(r"(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)")
_SPACES = re.compile(r"\s+")


def normalize_for_speech(text: str) -> str:
    """Speakable version of one chunk; may be empty (e.g. a chunk that was only emoji)."""
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub("link", text)
    text = _LINE_MARKER.sub("", text)
    text = _EMPHASIS.sub("", text)
    text = _UNDERSCORE_ITALIC.sub(r"\1", text)
    text = "".join(ch for ch in text if not _is_emoji(ch))
    return _SPACES.sub(" ", text).strip()


def _is_emoji(ch: str) -> bool:
    code = ord(ch)
    if code in (0x200D, 0xFE0F, 0x20E3):  # joiners/variation selectors used by emoji
        return True
    if 0x1F000 <= code <= 0x1FAFF or 0x2600 <= code <= 0x27BF:
        return True
    return unicodedata.category(ch) == "So"
