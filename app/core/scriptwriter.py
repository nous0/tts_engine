"""Podcast scripts written by an LLM from a topic (step 1 of topic -> episode).

The flow has two steps on purpose: a writer turns a ``TopicBrief`` into a ``Script``,
the user reads and edits it, and only then is it rendered (``POST /v1/podcast``). A
script is returned in the exact shape ``POST /v1/podcast`` accepts, so a client can
send it back as is.

A writer is any ``async (TopicBrief) -> Script`` callable, injected like
``podcast.SynthesizeFn``:

- ``OpenAIScriptWriter`` asks a chat model for strict JSON (title, speakers with a
  gender, turns). Each speaker's gender goes straight into voice casting, so no name
  table is needed.
- ``FakeScriptWriter`` writes a fixed, deterministic dialogue, for tests and the smoke
  test (which never calls OpenAI).

Both run the output through ``finalize``: every turn must belong to a listed speaker,
text is made speakable (``normalize_for_speech``), and long turns are split at
sentence boundaries into consecutive turns of the same speaker, so no single TTS call
gets too long (OpenAI TTS takes at most 4096 characters).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
import openai
from openai import AsyncOpenAI

from app.config import Settings

from .chunker import SentenceChunker
from .podcast import Turn
from .providers.base import Gender, ProviderError
from .providers.openai import translate_openai_error
from .textnorm import normalize_for_speech

log = logging.getLogger("app.scriptwriter")

# Characters of dialogue per minute of speech. Measured on Kokoro at speed 1.0:
# 184 characters took 13.7 s, about 800 per minute.
CHARS_PER_MINUTE = 800

DEFAULT_MAX_TURN_CHARS = 1500
DEFAULT_HOSTS = (("Alice", "female"), ("Bob", "male"))


@dataclass(slots=True)
class Host:
    name: str
    gender: Gender | None = None


@dataclass(slots=True)
class TopicBrief:
    """What the episode should be about."""

    topic: str
    duration_min: float = 3.0
    hosts: list[Host] = field(default_factory=list)
    style: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "duration_min": self.duration_min,
            "hosts": [{"name": h.name, "gender": h.gender} for h in self.hosts],
            "style": self.style,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TopicBrief:
        return cls(
            topic=data["topic"],
            duration_min=data.get("duration_min", 3.0),
            hosts=[Host(h["name"], h.get("gender")) for h in data.get("hosts") or []],
            style=data.get("style"),
        )


@dataclass(slots=True)
class Script:
    title: str
    turns: list[Turn]
    genders: dict[str, Gender]

    @property
    def estimated_minutes(self) -> float:
        return round(sum(len(t.text) for t in self.turns) / CHARS_PER_MINUTE, 1)

    def to_request(self) -> dict[str, Any]:
        """The body for ``POST /v1/podcast`` (plus ``estimated_minutes``, ignored there)."""
        return {
            "title": self.title,
            "turns": [{"speaker": t.speaker, "text": t.text} for t in self.turns],
            "genders": self.genders,
            "normalize_text": True,
            "estimated_minutes": self.estimated_minutes,
        }


ScriptWriter = Callable[[TopicBrief], Awaitable[Script]]


def finalize(
    name: str,
    title: str,
    speakers: Sequence[dict[str, Any]],
    turns: Sequence[dict[str, Any]],
    max_turn_chars: int = DEFAULT_MAX_TURN_CHARS,
) -> Script:
    """Check and clean a writer's raw output; ``name`` is the writer, for errors.

    Raises ``ProviderError`` (kind ``upstream``) when the output can't be used.
    """
    genders: dict[str, Gender] = {}
    for speaker in speakers:
        speaker_name = str(speaker.get("name") or "").strip()
        if speaker_name and speaker.get("gender") in ("female", "male"):
            genders[speaker_name] = speaker["gender"]

    result: list[Turn] = []
    for raw in turns:
        speaker = str(raw.get("speaker") or "").strip()
        if speaker not in genders:
            raise ProviderError(
                name, f"the script uses a speaker that wasn't listed: {speaker!r}"
            )
        text = normalize_for_speech(str(raw.get("text") or ""))
        result.extend(Turn(speaker=speaker, text=part) for part in _split(text, max_turn_chars))
    if not result:
        raise ProviderError(name, "the script has no lines")

    used = {t.speaker for t in result}
    return Script(
        title=" ".join(str(title).split())[:200] or "Untitled episode",
        turns=result,
        genders={k: v for k, v in genders.items() if k in used},
    )


def _split(text: str, max_chars: int) -> list[str]:
    """Whole sentences grouped into pieces of at most ``max_chars``."""
    if len(text) <= max_chars:
        return [text] if text else []
    chunker = SentenceChunker(max_length=max_chars)
    chunker.append(text)
    pieces = []
    while (chunk := chunker.take(None, final=True)) is not None:
        pieces.append(chunk.text)
    return pieces


def _target_chars(brief: TopicBrief) -> int:
    return round(brief.duration_min * CHARS_PER_MINUTE)


# --- OpenAI ---

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "speakers", "turns"],
    "properties": {
        "title": {"type": "string"},
        "speakers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "gender"],
                "properties": {
                    "name": {"type": "string"},
                    "gender": {"type": "string", "enum": ["female", "male"]},
                },
            },
        },
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["speaker", "text"],
                "properties": {"speaker": {"type": "string"}, "text": {"type": "string"}},
            },
        },
    },
}

_SYSTEM = """You write scripts for an audio podcast that a text-to-speech engine reads aloud.

Rules:
- English only.
- Write for the ear: short, natural spoken sentences and contractions.
- No markdown, lists, headings, emoji, URLs, stage directions or sound effects.
- Spell out symbols and abbreviations that would be read wrongly ("percent", "versus").
- Each turn is one speaker talking without interruption, one to four sentences, and at most \
{max_turn} characters.
- The hosts talk to each other: questions, reactions, concrete examples, a little humor.
- Open with a short welcome that names the topic; close with a brief sign-off.
- Stay factual. If something is uncertain, say so instead of inventing figures.
- List every speaker with a first name and a gender, and use exactly those names in turns."""


def _messages(brief: TopicBrief, max_turn_chars: int) -> list[dict[str, str]]:
    target = _target_chars(brief)
    if brief.hosts:
        hosts = ", ".join(f"{h.name} ({h.gender})" if h.gender else h.name for h in brief.hosts)
        cast = f"Hosts: {hosts}. Use exactly these speakers."
    else:
        cast = "Hosts: choose two hosts with distinct first names, one female and one male."
    lines = [
        f"Topic: {brief.topic}",
        f"Length: about {target} characters of dialogue in total "
        f"(about {brief.duration_min:g} minutes spoken).",
        cast,
    ]
    if brief.style:
        lines.append(f"Style: {brief.style}")
    return [
        {"role": "system", "content": _SYSTEM.format(max_turn=max_turn_chars)},
        {"role": "user", "content": "\n".join(lines)},
    ]


class OpenAIScriptWriter:
    """Writes scripts with an OpenAI chat model and strict JSON output."""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: float = 120.0,
        max_retries: int = 2,
        max_turn_chars: int = DEFAULT_MAX_TURN_CHARS,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=httpx.Timeout(timeout, connect=min(5.0, timeout)),
            max_retries=max_retries,
            http_client=http_client,
        )
        self._model = model
        self._max_turn_chars = max_turn_chars

    @property
    def label(self) -> str:
        return f"openai:{self._model}"

    async def __call__(self, brief: TopicBrief) -> Script:
        # About 4 characters per token, plus room for the JSON around the dialogue.
        max_tokens = min(16_000, _target_chars(brief) // 3 + 800)
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=_messages(brief, self._max_turn_chars),
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "podcast_script", "strict": True, "schema": _SCHEMA},
                },
                max_tokens=max_tokens,
            )
        except (openai.APIError, httpx.HTTPError) as exc:
            raise translate_openai_error(exc, self.name) from exc

        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise ProviderError(
                self.name, f"the model declined: {choice.message.refusal}", kind="bad_request"
            )
        if choice.finish_reason == "length":
            raise ProviderError(
                self.name, "the script was cut off at the length limit; ask for fewer minutes"
            )
        try:
            data = json.loads(choice.message.content or "")
        except json.JSONDecodeError as exc:
            raise ProviderError(self.name, "the model returned invalid JSON") from exc
        log.info(
            "script written: model=%s turns=%d chars=%d tokens=%s",
            self._model,
            len(data.get("turns", [])),
            sum(len(str(t.get("text", ""))) for t in data.get("turns", [])),
            getattr(response.usage, "total_tokens", None),
        )
        script = finalize(
            self.name,
            data.get("title", ""),
            data.get("speakers", []),
            data.get("turns", []),
            self._max_turn_chars,
        )
        return _apply_host_genders(script, brief)


def _apply_host_genders(script: Script, brief: TopicBrief) -> Script:
    """Genders the user gave for their hosts win over the model's."""
    for host in brief.hosts:
        if host.gender and host.name in script.genders:
            script.genders[host.name] = host.gender
    return script


# --- Fake ---


class FakeScriptWriter:
    """A fixed, deterministic dialogue about the topic; never calls a model."""

    name = "fake"
    label = "fake"

    def __init__(self, max_turn_chars: int = DEFAULT_MAX_TURN_CHARS) -> None:
        self._max_turn_chars = max_turn_chars

    async def __call__(self, brief: TopicBrief) -> Script:
        hosts = [(h.name, h.gender or "female") for h in brief.hosts] or list(DEFAULT_HOSTS)
        first, second = hosts[0][0], hosts[1 % len(hosts)][0]
        lines = [
            (first, f"Welcome to the show. Today we're talking about {brief.topic}."),
            (second, f"Thanks, {first}. It's a topic a lot of people ask about."),
        ]
        index = 0
        while sum(len(t) for _, t in lines) < _target_chars(brief) - 80:
            speaker = hosts[index % len(hosts)][0]
            index += 1
            text = f"Here's point number {index} about {brief.topic}. It's worth a closer look."
            lines.append((speaker, text))
        lines.append((first, "That's all for today. Thanks for listening."))
        return finalize(
            self.name,
            f"A short episode about {brief.topic}",
            [{"name": n, "gender": g} for n, g in hosts],
            [{"speaker": s, "text": t} for s, t in lines],
            self._max_turn_chars,
        )


def build_writer(settings: Settings) -> ScriptWriter | None:
    """The configured writer, or None when script writing is off or impossible."""
    mode = settings.script_writer
    if mode == "off":
        return None
    if mode == "fake":
        return FakeScriptWriter(settings.script_max_turn_chars)
    if not settings.openai_api_key:
        if mode == "openai":
            log.warning("SCRIPT_WRITER=openai but OPENAI_API_KEY is not set; writing is off")
        return None
    return OpenAIScriptWriter(
        api_key=settings.openai_api_key,
        model=settings.script_model,
        timeout=settings.script_timeout,
        max_retries=settings.openai_max_retries,
        max_turn_chars=settings.script_max_turn_chars,
    )
