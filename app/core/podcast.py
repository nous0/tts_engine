"""Podcast assembly: parse a multi-speaker script and render it to one file.

A script is a list of ``{speaker, text}`` turns. It can arrive as JSON (a list, or
an object with a ``turns`` key) or as a plain transcript, which is what people
actually paste::

    Alice: Welcome to the show.
    Bob: Glad to be here.

Rendering synthesizes each turn as raw PCM, stitches the turns with a configurable
pause, brings the whole mix to one loudness (EBU R128 via ffmpeg, or peak level) so no
speaker sticks out, and encodes once at the end. Synthesis is injected as a callable,
so this module stays independent of the engine and is trivial to test.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from . import audio
from .names import guess_gender
from .providers.base import Gender, SynthOpts, Voice
from .textnorm import normalize_for_speech

# "Alice: text" — a speaker label is a short leading token before the colon.
_SPEAKER_LINE = re.compile(r"^([^:\n]{1,60}?)\s*:\s*(.+)$")

# Used when a plain-text script has no speaker labels at all.
DEFAULT_SPEAKER = "Speaker"

DEFAULT_PAUSE_MS = 600

# How the stitched episode's level is set: EBU R128 loudness (needs ffmpeg), peak level
# (pure Python), or left alone.
Loudness = Literal["lufs", "peak", "off"]

# Called with (turns_done, turns_total) after each turn is synthesized. It may be a
# coroutine function (the server writes progress to the job store), and is awaited then.
ProgressCallback = Callable[[int, int], Awaitable[None] | None]

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
    loudness: Loudness = "peak"
    target_lufs: float = audio.DEFAULT_TARGET_LUFS
    speed: float = 1.0
    instructions: str | None = None
    title: str | None = None
    # Compressed output only (mp3 etc.): resample and bitrate; wav/pcm stay 24 kHz.
    sample_rate: int | None = None
    bitrate: str | None = None


@dataclass(slots=True)
class RenderResult:
    """A rendered episode: the encoded file plus what is known about it."""

    audio: bytes
    duration_s: float
    loudness: dict[str, Any] | None = None


def pick_loudness(requested: Loudness | None, normalize: bool, has_ffmpeg: bool) -> Loudness:
    """The loudness mode to use: the requested one, else LUFS when ffmpeg can do it.

    ``normalize=False`` (the older on/off switch) means "off" unless a mode is requested.
    """
    if requested is not None:
        return requested
    if not normalize:
        return "off"
    return "lufs" if has_ffmpeg else "peak"


def parse_script(raw: Any) -> list[Turn]:
    """Build turns from JSON, a plain-text transcript, or already-structured data."""
    if isinstance(raw, str):
        return _parse_string(raw)
    if isinstance(raw, dict):
        return _parse_sequence(_turns_field(raw))
    if isinstance(raw, Sequence):
        return _parse_sequence(raw)
    raise ScriptError(f"Cannot read a script from {type(raw).__name__}.")


def normalize_turns(turns: Sequence[Turn]) -> list[Turn]:
    """Make each turn speakable: strip markdown, turn links into "link", drop emoji.

    Meant for LLM-written scripts. A turn left with nothing to say is an error, not a
    silent gap, so the caller sees which line was unusable.
    """
    result: list[Turn] = []
    for index, turn in enumerate(turns):
        text = normalize_for_speech(turn.text)
        if not text:
            raise ScriptError(f"Turn {index} has nothing left to say after normalizing.")
        result.append(replace(turn, text=text))
    return result


def speakers(turns: Sequence[Turn]) -> list[str]:
    """Distinct speaker names, in first-appearance order."""
    ordered: dict[str, None] = {}
    for turn in turns:
        ordered.setdefault(turn.speaker, None)
    return list(ordered)


def assign_voices(
    turns: Sequence[Turn],
    voices: dict[str, str] | None,
    catalog: Sequence[Voice],
    genders: dict[str, Gender] | None = None,
) -> dict[str, str]:
    """Complete the speaker -> voice map, casting each speaker by gender.

    Explicit ``voices`` always win. Every other speaker, in first-appearance order,
    gets a gender (explicit ``genders`` > guessed from the name > whichever gender is
    less represented so far) and then the first *unused* catalog voice of that
    gender, so two women and a man get three distinct, fitting voices. Catalogs are
    expected to list their best voices first. When a gender runs out, any unused
    voice is taken; only once every voice is taken are voices reused, least-used
    first. Each speaker keeps one voice for the whole episode.
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

    gender_of = {v.id: v.gender for v in catalog}
    usage = {v.id: 0 for v in catalog}
    counts: dict[Gender, int] = {"female": 0, "male": 0}
    last: Gender | None = None
    for voice_id in assigned.values():
        if voice_id in usage:
            usage[voice_id] += 1
        if (g := gender_of.get(voice_id)) is not None:
            counts[g] += 1
            last = g

    for name in missing:
        gender = (genders or {}).get(name) or guess_gender(name)
        if gender is None:
            gender = _balancing_gender(counts, last)
        voice = _pick_voice(catalog, usage, gender)
        assigned[name] = voice.id
        usage[voice.id] += 1
        cast_as = voice.gender or gender
        counts[cast_as] += 1
        last = cast_as
    return assigned


def _balancing_gender(counts: dict[Gender, int], last: Gender | None) -> Gender:
    """For a speaker of unknown gender: the rarer gender, alternating on ties."""
    if counts["female"] != counts["male"]:
        return "female" if counts["female"] < counts["male"] else "male"
    return "male" if last == "female" else "female"


def _pick_voice(catalog: Sequence[Voice], usage: dict[str, int], gender: Gender) -> Voice:
    matching = [v for v in catalog if v.gender == gender]
    for pool in (matching, catalog):
        for voice in pool:
            if usage[voice.id] == 0:
                return voice
    # Everything is taken: reuse the least-used voice, preferring the right gender.
    pool = matching or list(catalog)
    return min(pool, key=lambda v: usage[v.id])


async def render(
    spec: PodcastSpec,
    synthesize: SynthesizeFn,
    on_progress: ProgressCallback | None = None,
) -> RenderResult:
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
            reported = on_progress(index, total)
            if inspect.isawaitable(reported):
                await reported

    mixed = audio.concat(segments, pause_ms=spec.pause_ms)
    loudness: dict[str, Any] | None = None
    if spec.loudness == "lufs":
        target = audio.LoudnessTarget(integrated=spec.target_lufs)
        mixed, loudness = await audio.loudness_normalize(mixed, target)
    elif spec.loudness == "peak":
        mixed = audio.normalize(mixed)
    encoded = await audio.encode(
        mixed,
        spec.format,
        sample_rate=spec.sample_rate,
        bitrate=spec.bitrate,
        metadata={"title": spec.title} if spec.title else None,
    )
    return RenderResult(
        audio=encoded, duration_s=audio.duration_ms(mixed) / 1000, loudness=loudness
    )


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
