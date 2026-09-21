"""Podcast assembly: parse a multi-speaker script and render it to one file.

A script is a list of ``{speaker, text}`` turns. It can arrive as JSON (a list, or
an object with a ``turns`` key) or as a plain transcript, which is what people
actually paste::

    Alice: Welcome to the show.
    Bob: Glad to be here.

Rendering synthesizes each turn as raw PCM, stitches the turns with a configurable
pause, normalizes the whole mix so no speaker sticks out, and encodes once at the
end. Synthesis is injected as a callable, so this module stays independent of the
engine and is trivial to test.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import audio
from .providers.base import SynthOpts

# "Alice: text" — a speaker label is a short leading token before the colon.
_SPEAKER_LINE = re.compile(r"^([^:\n]{1,60}?)\s*:\s*(.+)$")

# Used when a plain-text script has no speaker labels at all.
DEFAULT_SPEAKER = "Speaker"

DEFAULT_PAUSE_MS = 600

# Called with (turns_done, turns_total) after each turn is synthesized.
ProgressCallback = Callable[[int, int], None]

# (text, voice, provider, opts) -> encoded audio bytes.
SynthesizeFn = Callable[[str, str, str | None, SynthOpts], Awaitable[bytes]]


class ScriptError(ValueError):
    """Raised when a script cannot be parsed or a turn has no usable voice."""


@dataclass(slots=True)
class Turn:
    """One line of dialogue, optionally overriding the speaker's defaults."""

    speaker: str
    text: str
    voice: str | None = None
    provider: str | None = None
    instructions: str | None = None


@dataclass(slots=True)
class PodcastSpec:
    """Everything needed to render an episode."""

    turns: list[Turn]
    voices: dict[str, str] = field(default_factory=dict)
    provider: str | None = None
    format: str = "wav"
    pause_ms: int = DEFAULT_PAUSE_MS
    normalize: bool = True
    speed: float = 1.0
    instructions: str | None = None


def parse_script(raw: Any) -> list[Turn]:
    """Build turns from JSON, a plain-text transcript, or already-structured data."""
    if isinstance(raw, str):
        return _parse_string(raw)
    if isinstance(raw, dict):
        return _parse_sequence(_turns_field(raw))
    if isinstance(raw, Sequence):
        return _parse_sequence(raw)
    raise ScriptError(f"Cannot read a script from {type(raw).__name__}.")


def speakers(turns: Sequence[Turn]) -> list[str]:
    """Distinct speaker names, in first-appearance order."""
    ordered: dict[str, None] = {}
    for turn in turns:
        ordered.setdefault(turn.speaker, None)
    return list(ordered)


def assign_voices(
    turns: Sequence[Turn],
    voices: dict[str, str] | None,
    catalog: Sequence[str],
) -> dict[str, str]:
    """Complete the speaker -> voice map, giving unassigned speakers distinct voices.

    Explicit assignments always win. Remaining speakers take unused voices from
    ``catalog`` so a two-host show sounds like two people with no configuration;
    once the catalog runs out, voices are reused in order.
    """
    assigned = dict(voices or {})
    missing = [name for name in speakers(turns) if name not in assigned]
    if not missing:
        return assigned
    if not catalog:
        raise ScriptError(
            "No voices available to assign; set 'voices' explicitly or choose a "
            "provider that lists voices."
        )

    pool = [v for v in catalog if v not in set(assigned.values())] or list(catalog)
    for index, name in enumerate(missing):
        assigned[name] = pool[index % len(pool)]
    return assigned


async def render(
    spec: PodcastSpec,
    synthesize: SynthesizeFn,
    on_progress: ProgressCallback | None = None,
) -> bytes:
    """Synthesize every turn and stitch them into a single encoded file.

    Turns are rendered one at a time: for long-form audio, predictable provider
    load matters more than wall-clock speed, and sequential calls keep the
    output deterministic.
    """
    if not spec.turns:
        raise ScriptError("Script contains no turns.")

    total = len(spec.turns)
    segments: list[bytes] = []
    for index, turn in enumerate(spec.turns, start=1):
        voice = turn.voice or spec.voices.get(turn.speaker)
        if not voice:
            raise ScriptError(
                f"No voice assigned for speaker '{turn.speaker}'; "
                f"pass one in 'voices'."
            )
        opts = SynthOpts(
            format="pcm",
            speed=spec.speed,
            instructions=turn.instructions or spec.instructions,
        )
        raw = await synthesize(turn.text, voice, turn.provider or spec.provider, opts)
        segments.append(audio.to_pcm(raw))
        if on_progress is not None:
            on_progress(index, total)

    mixed = audio.concat(segments, pause_ms=spec.pause_ms)
    if spec.normalize:
        mixed = audio.normalize(mixed)
    return await audio.encode(mixed, spec.format)


def _turns_field(data: dict) -> Any:
    for key in ("turns", "script", "dialogue"):
        if key in data:
            return data[key]
    raise ScriptError("Script object must have a 'turns' list.")


def _parse_string(raw: str) -> list[Turn]:
    text = raw.strip()
    if not text:
        raise ScriptError("Script is empty.")
    if text[0] in "[{":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ScriptError(f"Script looks like JSON but is not valid JSON: {exc}") from exc
        return parse_script(data)
    return _parse_transcript(text)


def _parse_transcript(text: str) -> list[Turn]:
    """Parse ``Speaker: line`` lines; unlabelled lines continue the previous turn."""
    turns: list[Turn] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _SPEAKER_LINE.match(stripped)
        if match:
            turns.append(Turn(speaker=match.group(1).strip(), text=match.group(2).strip()))
        elif turns:
            turns[-1].text = f"{turns[-1].text} {stripped}"
        else:
            # A script with no labels at all is a single-voice narration.
            turns.append(Turn(speaker=DEFAULT_SPEAKER, text=stripped))
    if not turns:
        raise ScriptError("Script contains no turns.")
    return turns


def _parse_sequence(data: Any) -> list[Turn]:
    if not isinstance(data, Sequence) or isinstance(data, str):
        raise ScriptError("Script must be a list of {speaker, text} objects.")
    turns = [_parse_turn(index, raw) for index, raw in enumerate(data)]
    if not turns:
        raise ScriptError("Script contains no turns.")
    return turns


def _parse_turn(index: int, raw: Any) -> Turn:
    if isinstance(raw, Turn):
        if not raw.text.strip():
            raise ScriptError(f"Turn {index} has no text.")
        return raw
    if not isinstance(raw, dict):
        raise ScriptError(f"Turn {index} must be an object, got {type(raw).__name__}.")

    text = str(raw.get("text") or raw.get("line") or "").strip()
    if not text:
        raise ScriptError(f"Turn {index} has no text.")
    speaker = str(raw.get("speaker") or raw.get("name") or DEFAULT_SPEAKER).strip()
    return Turn(
        speaker=speaker or DEFAULT_SPEAKER,
        text=text,
        voice=raw.get("voice"),
        provider=raw.get("provider"),
        instructions=raw.get("instructions"),
    )
