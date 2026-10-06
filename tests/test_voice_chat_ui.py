"""Tests for the voice chat demo's non-UI logic (skipped without the demo extra)."""

from __future__ import annotations

import importlib.util
import io
import sys
import wave
from pathlib import Path

import pytest

pytest.importorskip("gradio")

_PATH = Path(__file__).resolve().parents[1] / "examples" / "voice_chat_ui.py"
_spec = importlib.util.spec_from_file_location("voice_chat_ui", _PATH)
demo = importlib.util.module_from_spec(_spec)
sys.modules["voice_chat_ui"] = demo
_spec.loader.exec_module(demo)


def test_history_becomes_openai_messages_with_the_voice_prompt():
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "assistant", "content": ""},  # the answer being written: skipped
        {"role": "user", "content": None},
    ]
    messages = demo.to_openai_messages(history)
    assert messages[0] == {"role": "system", "content": demo.SYSTEM_PROMPT}
    assert messages[1:] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]


async def test_fake_llm_streams_a_canned_answer_word_by_word():
    pieces = [piece async for piece in demo.fake_llm("hello", wps=1000)]
    assert len(pieces) > 5
    assert "".join(pieces) == demo.pick_fake_answer("hello")


def test_fake_answer_choice_is_stable():
    assert demo.pick_fake_answer("same question") == demo.pick_fake_answer("same question")


def test_wav_chunks_are_16_bit_mono_at_the_given_rate():
    pcm = b"\x01\x02" * 2400 + b"\x03"  # an odd trailing byte is dropped
    with wave.open(io.BytesIO(demo.wav_bytes(pcm, 24_000))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24_000)
        assert w.getnframes() == 2400


def test_the_app_builds_without_a_tts_server():
    app = demo.build_app("ws://127.0.0.1:9/v1/speech/ws", "http://127.0.0.1:9", False)
    assert app is not None
