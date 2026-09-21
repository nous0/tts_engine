"""Tests for script parsing, voice assignment, and multi-speaker assembly."""

from __future__ import annotations

import json

import pytest

from app.core import audio
from app.core.podcast import (
    PodcastSpec,
    ScriptError,
    Turn,
    assign_voices,
    parse_script,
    render,
    speakers,
)
from app.core.providers.base import SynthOpts
from tests.test_audio import tone


def test_parse_json_script():
    raw = json.dumps(
        [
            {"speaker": "Alice", "text": "Hello there."},
            {"speaker": "Bob", "text": "Hi Alice.", "voice": "bm_george"},
        ]
    )
    turns = parse_script(raw)
    assert [t.speaker for t in turns] == ["Alice", "Bob"]
    assert turns[1].voice == "bm_george"


def test_parse_json_object_with_turns_key():
    raw = json.dumps({"turns": [{"speaker": "A", "text": "One."}]})
    assert parse_script(raw)[0].text == "One."


def test_parse_text_script():
    turns = parse_script("Alice: Hello there.\nBob: Hi Alice.")
    assert [(t.speaker, t.text) for t in turns] == [
        ("Alice", "Hello there."),
        ("Bob", "Hi Alice."),
    ]


def test_parse_text_script_continues_unlabelled_lines():
    turns = parse_script("Alice: First part.\nSecond part.\nBob: Reply.")
    assert turns[0].text == "First part. Second part."
    assert turns[1].text == "Reply."


def test_parse_text_script_without_labels_uses_default_speaker():
    turns = parse_script("Just a narration line.")
    assert turns[0].speaker == "Speaker"


def test_parse_script_blank_lines_are_ignored():
    turns = parse_script("Alice: One.\n\n\nBob: Two.\n")
    assert len(turns) == 2


def test_parse_script_rejects_empty_input():
    with pytest.raises(ScriptError):
        parse_script("   ")
    with pytest.raises(ScriptError):
        parse_script([])


def test_parse_script_rejects_invalid_json():
    with pytest.raises(ScriptError, match="valid JSON"):
        parse_script('[{"speaker": "A", ')


def test_parse_script_rejects_turn_without_text():
    with pytest.raises(ScriptError, match="no text"):
        parse_script([{"speaker": "A", "text": "  "}])


def test_parse_script_accepts_turn_objects():
    turns = parse_script([Turn(speaker="A", text="Hi")])
    assert turns[0].text == "Hi"


def test_speakers_preserves_first_appearance_order():
    turns = parse_script("Bob: One.\nAlice: Two.\nBob: Three.")
    assert speakers(turns) == ["Bob", "Alice"]


def test_assign_voices_gives_distinct_voices():
    turns = parse_script("Alice: One.\nBob: Two.")
    mapping = assign_voices(turns, None, ["v1", "v2", "v3"])
    assert mapping["Alice"] != mapping["Bob"]


def test_assign_voices_respects_explicit_map():
    turns = parse_script("Alice: One.\nBob: Two.")
    mapping = assign_voices(turns, {"Alice": "pinned"}, ["pinned", "v2"])
    assert mapping["Alice"] == "pinned"
    assert mapping["Bob"] == "v2"


def test_assign_voices_wraps_when_pool_is_small():
    turns = parse_script("A: 1.\nB: 2.\nC: 3.")
    mapping = assign_voices(turns, None, ["only"])
    assert set(mapping.values()) == {"only"}


def test_assign_voices_without_catalog_raises():
    turns = parse_script("Alice: One.")
    with pytest.raises(ScriptError, match="No voices available"):
        assign_voices(turns, None, [])


def _recording_synth(calls: list[tuple], clip_ms: int = 100):
    async def synthesize(text: str, voice: str, provider: str | None, opts: SynthOpts):
        calls.append((text, voice, provider, opts.format))
        return tone(clip_ms)

    return synthesize


async def test_render_stitches_turns_with_pauses():
    turns = parse_script("Alice: One.\nBob: Two.\nAlice: Three.")
    spec = PodcastSpec(
        turns=turns,
        voices={"Alice": "v1", "Bob": "v2"},
        format="pcm",
        pause_ms=500,
        normalize=False,
    )
    calls: list[tuple] = []
    data = await render(spec, _recording_synth(calls, clip_ms=100))

    # 3 clips of 100 ms + 2 gaps of 500 ms.
    assert audio.duration_ms(data) == pytest.approx(3 * 100 + 2 * 500, abs=2.0)
    assert [c[1] for c in calls] == ["v1", "v2", "v1"]


async def test_render_always_requests_pcm_from_the_provider():
    spec = PodcastSpec(
        turns=parse_script("A: One."), voices={"A": "v1"}, format="wav", normalize=False
    )
    calls: list[tuple] = []
    data = await render(spec, _recording_synth(calls))
    assert calls[0][3] == "pcm"
    assert data[:4] == b"RIFF"


async def test_render_reports_progress_per_turn():
    spec = PodcastSpec(
        turns=parse_script("A: One.\nB: Two."), voices={"A": "v1", "B": "v2"}, format="pcm"
    )
    seen: list[tuple[int, int]] = []
    await render(spec, _recording_synth([]), lambda done, total: seen.append((done, total)))
    assert seen == [(1, 2), (2, 2)]


async def test_render_normalizes_when_enabled():
    spec = PodcastSpec(
        turns=parse_script("A: One."), voices={"A": "v1"}, format="pcm", normalize=True
    )

    async def quiet(text: str, voice: str, provider: str | None, opts: SynthOpts):
        return tone(50, amplitude=500)

    data = await render(spec, quiet)
    assert audio.peak(data) > 20_000


async def test_render_unwraps_wav_returned_by_a_provider():
    spec = PodcastSpec(
        turns=parse_script("A: One."), voices={"A": "v1"}, format="pcm", normalize=False
    )

    async def wav_provider(text: str, voice: str, provider: str | None, opts: SynthOpts):
        return audio.pcm_to_wav(tone(80))

    data = await render(spec, wav_provider)
    assert data[:4] != b"RIFF"
    assert audio.duration_ms(data) == pytest.approx(80, abs=2.0)


async def test_render_uses_per_turn_overrides():
    turns = [Turn(speaker="A", text="One", voice="pinned", provider="other")]
    spec = PodcastSpec(turns=turns, voices={"A": "ignored"}, provider="base", format="pcm")
    calls: list[tuple] = []
    await render(spec, _recording_synth(calls))
    assert calls[0][1] == "pinned"
    assert calls[0][2] == "other"


async def test_render_without_a_voice_raises():
    spec = PodcastSpec(turns=parse_script("A: One."), voices={}, format="pcm")
    with pytest.raises(ScriptError, match="No voice assigned"):
        await render(spec, _recording_synth([]))
