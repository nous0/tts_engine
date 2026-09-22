"""Guess a speaker's gender from their name, to cast podcast voices.

This is a small built-in lookup, not a classifier. It knows common English and
Vietnamese given names, plus a few titles and Vietnamese middle names, and returns
``None`` for anything it isn't sure about (``Host``, ``Speaker 1``, unisex names).
Callers treat ``None`` as "unknown" and fall back to balancing voices, and users can
always override a guess with an explicit gender or voice.

Matching ignores case and diacritics, so ``Hương``, ``huong`` and ``HUONG`` are the
same name. Vietnamese names that differ only by diacritics and have different
genders (``Dung``/``Dũng``, ``Hoa``/``Hòa``) are deliberately left out.
"""

from __future__ import annotations

import re
import unicodedata

from .providers.base import Gender

_FEMALE_NAMES = frozenset(
    """
    alice amanda amelia amy anna anne ava barbara bella betty carol caroline charlotte
    chloe claire deborah diana dorothy elizabeth ella emily emma eva evelyn grace hannah
    helen isabella jane jennifer jessica joan julia karen kate katherine laura linda
    lisa lucy margaret maria mary megan mia michelle nancy natalie nicole olivia patricia
    rachel rebecca rose ruth samantha sandra sarah sophia sophie susan victoria zoe
    lan mai huong hang trang thao nga ngan hanh thu yen nhung oanh uyen diem nhi vy
    tram loan lien cuc hue thuy tuyet ly quynh thoa trinh
    """.split()
)

_MALE_NAMES = frozenset(
    """
    adam alexander andrew anthony ben benjamin bob brian charles christopher
    daniel david edward eric ethan frank gary george henry jack jacob james jason john
    jonathan joseph joshua kevin larry liam lucas mark matthew michael nathan noah
    oliver patrick paul peter richard robert ryan samuel scott steven thomas tim
    timothy tom tony william
    nam hung tuan long duc quang huy trung kien phong son thang vinh khoa dat cuong
    hoang tai thinh loc phuc tung khang
    """.split()
)

# Titles/honorifics that mark gender when they lead a name ("Mr Smith", "Chị Lan").
_FEMALE_TITLES = frozenset({"mrs", "ms", "miss", "madam", "lady", "ba", "chi", "co"})
_MALE_TITLES = frozenset({"mr", "sir", "ong", "chu"})
# Titles that say nothing about gender; skipped so the name after them is used.
_NEUTRAL_TITLES = frozenset({"dr", "prof", "professor"})

# Vietnamese middle names that carry gender: "Nguyễn Thị Lan", "Trần Văn Nam".
_FEMALE_MIDDLE = frozenset({"thi"})
_MALE_MIDDLE = frozenset({"van"})

_TOKEN_SPLIT = re.compile(r"[\s_\-.]+")


def normalize(name: str) -> list[str]:
    """Lowercase, strip diacritics (incl. Vietnamese đ), and split into tokens."""
    text = name.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [t for t in _TOKEN_SPLIT.split(text.lower()) if t]


def guess_gender(name: str) -> Gender | None:
    """Best-effort gender for a speaker name, or ``None`` when unsure."""
    tokens = normalize(name)
    while len(tokens) > 1 and tokens[0] in _NEUTRAL_TITLES:
        tokens = tokens[1:]
    if not tokens:
        return None

    if len(tokens) >= 2:
        if tokens[0] in _FEMALE_TITLES:
            return "female"
        if tokens[0] in _MALE_TITLES:
            return "male"
        middle = set(tokens[1:-1])
        if middle & _FEMALE_MIDDLE:
            return "female"
        if middle & _MALE_MIDDLE:
            return "male"

    # English puts the given name first, Vietnamese puts it last; check both, and
    # only answer when they don't disagree.
    found = {_lookup(tokens[0]), _lookup(tokens[-1])} - {None}
    return found.pop() if len(found) == 1 else None


def _lookup(token: str) -> Gender | None:
    if token in _FEMALE_NAMES:
        return "female"
    if token in _MALE_NAMES:
        return "male"
    return None
