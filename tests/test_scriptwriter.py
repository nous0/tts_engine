"""Tests for podcast script writing: output checks, the fake writer, and the OpenAI writer
against a mocked HTTP transport (no network, no key)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.core.providers.base import ProviderError
from app.core.scriptwriter import (
    CHARS_PER_MINUTE,
    FakeScriptWriter,
    Host,
    OpenAIScriptWriter,
    TopicBrief,
    build_writer,
    finalize,
)

SPEAKERS = [{"name": "Alice", "gender": "female"}, {"name": "Bob", "gender": "male"}]


# --- finalize ---


def test_finalize_builds_turns_and_genders():
    script = finalize(
        "w",
        "  Local   TTS ",
        SPEAKERS,
        [{"speaker": "Alice", "text": "Hi Bob."}, {"speaker": "Bob", "text": "Hi Alice."}],
    )
    assert script.title == "Local TTS"
    turns = [(t.speaker, t.text) for t in script.turns]
    assert turns == [("Alice", "Hi Bob."), ("Bob", "Hi Alice.")]
    assert script.genders == {"Alice": "female", "Bob": "male"}


def test_finalize_rejects_a_speaker_that_was_not_listed():
    with pytest.raises(ProviderError, match="wasn't listed: 'Carol'") as info:
        finalize("w", "T", SPEAKERS, [{"speaker": "Carol", "text": "Hello."}])
    assert info.value.kind == "upstream"


def test_finalize_makes_text_speakable_and_drops_empty_turns():
    script = finalize(
        "w",
        "T",
        SPEAKERS,
        [
            {"speaker": "Alice", "text": "**Welcome!** See https://example.com 🎉"},
            {"speaker": "Bob", "text": "🎉🎉"},
        ],
    )
    assert len(script.turns) == 1
    assert "*" not in script.turns[0].text and "https" not in script.turns[0].text
    assert script.genders == {"Alice": "female"}  # Bob ended up saying nothing


def test_finalize_splits_long_turns_at_sentences():
    long = " ".join(f"This is sentence {i} of a very long turn." for i in range(40))
    script = finalize("w", "T", SPEAKERS, [{"speaker": "Alice", "text": long}], max_turn_chars=300)
    assert len(script.turns) > 1
    assert all(t.speaker == "Alice" for t in script.turns)
    assert all(len(t.text) <= 300 for t in script.turns)
    assert all(t.text.endswith(".") for t in script.turns)  # whole sentences
    assert " ".join(t.text for t in script.turns) == long


def test_finalize_without_lines_raises():
    with pytest.raises(ProviderError, match="no lines"):
        finalize("w", "T", SPEAKERS, [])


def test_script_to_request_is_a_podcast_request_body():
    from app.models.schemas import PodcastRequest

    script = finalize("w", "Ep", SPEAKERS, [{"speaker": "Bob", "text": "x" * 400}])
    body = script.to_request()
    assert body["estimated_minutes"] == round(400 / CHARS_PER_MINUTE, 1)
    req = PodcastRequest(**body, format="mp3")  # extra keys are ignored
    assert req.title == "Ep"
    assert req.genders == {"Bob": "male"}
    assert req.normalize_text is True


def test_topic_brief_round_trips():
    brief = TopicBrief("AI", 5, [Host("Lan", "female"), Host("Minh")], "casual")
    assert TopicBrief.from_dict(json.loads(json.dumps(brief.to_dict()))) == brief


# --- fake writer ---


async def test_fake_writer_is_deterministic_and_about_the_right_length():
    writer = FakeScriptWriter()
    brief = TopicBrief("solar power", duration_min=2)
    first, second = await writer(brief), await writer(brief)
    assert first == second
    chars = sum(len(t.text) for t in first.turns)
    assert 2 * CHARS_PER_MINUTE - 200 <= chars <= 2 * CHARS_PER_MINUTE + 200
    assert "solar power" in first.turns[0].text
    assert first.genders == {"Alice": "female", "Bob": "male"}


async def test_fake_writer_uses_the_requested_hosts():
    script = await FakeScriptWriter()(
        TopicBrief("tea", duration_min=1, hosts=[Host("Lan", "female"), Host("Minh", "male")])
    )
    assert {t.speaker for t in script.turns} == {"Lan", "Minh"}
    assert script.genders == {"Lan": "female", "Minh": "male"}


# --- OpenAI writer ---

FAST_RETRY = {"retry-after-ms": "1"}


def _completion(content: str, finish_reason: str = "stop", refusal: str | None = None):
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "refusal": refusal},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        },
    )


def _error(status: int, message: str = "boom") -> httpx.Response:
    return httpx.Response(
        status, json={"error": {"message": message, "type": "test"}}, headers=FAST_RETRY
    )


def _writer(responses, **kwargs) -> tuple[OpenAIScriptWriter, list[httpx.Request]]:
    calls: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    kwargs.setdefault("max_retries", 0)
    return OpenAIScriptWriter(api_key="sk-test", http_client=client, **kwargs), calls


GOOD = json.dumps(
    {
        "title": "Tea, explained",
        "speakers": SPEAKERS,
        "turns": [
            {"speaker": "Alice", "text": "Welcome! Today it's all about tea."},
            {"speaker": "Bob", "text": "I drink way too much of it."},
        ],
    }
)


async def test_openai_writer_asks_for_strict_json_and_parses_it():
    writer, calls = _writer([_completion(GOOD)], model="gpt-4o-mini")
    script = await writer(TopicBrief("the history of tea", duration_min=2, style="playful"))

    assert script.title == "Tea, explained"
    assert [t.speaker for t in script.turns] == ["Alice", "Bob"]
    assert script.genders == {"Alice": "female", "Bob": "male"}

    body = json.loads(calls[0].content)
    assert body["model"] == "gpt-4o-mini"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["max_tokens"] == 2 * CHARS_PER_MINUTE // 3 + 800
    user = body["messages"][1]["content"]
    assert "Topic: the history of tea" in user
    assert f"about {2 * CHARS_PER_MINUTE} characters" in user
    assert "Style: playful" in user
    assert "one female and one male" in user
    assert "1500 characters" in body["messages"][0]["content"]


async def test_openai_writer_passes_the_hosts_and_their_genders_win():
    content = json.dumps(
        {
            "title": "T",
            "speakers": [{"name": "Sam", "gender": "male"}, {"name": "Lan", "gender": "female"}],
            "turns": [{"speaker": "Sam", "text": "Hi."}, {"speaker": "Lan", "text": "Hello."}],
        }
    )
    writer, calls = _writer([_completion(content)])
    script = await writer(TopicBrief("x y z", hosts=[Host("Sam", "female"), Host("Lan")]))
    assert "Hosts: Sam (female), Lan. Use exactly these speakers." in (
        json.loads(calls[0].content)["messages"][1]["content"]
    )
    assert script.genders == {"Sam": "female", "Lan": "female"}


@pytest.mark.parametrize(
    ("response", "kind", "message"),
    [
        (_completion("not json"), "upstream", "invalid JSON"),
        (_completion(GOOD, finish_reason="length"), "upstream", "cut off"),
        (_completion("", refusal="I can't help with that."), "bad_request", "declined"),
        (_error(401), "auth", "boom"),
        (_error(429), "rate_limited", "boom"),
        (_error(400), "bad_request", "boom"),
        (_error(500), "upstream", "boom"),
    ],
)
async def test_openai_writer_errors_become_provider_errors(response, kind, message):
    writer, _calls = _writer([response])
    with pytest.raises(ProviderError, match=message) as info:
        await writer(TopicBrief("tea time"))
    assert info.value.kind == kind


async def test_openai_writer_retries_server_errors():
    writer, calls = _writer([_error(500), _completion(GOOD)], max_retries=2)
    script = await writer(TopicBrief("tea time"))
    assert script.title == "Tea, explained"
    assert len(calls) == 2


# --- build_writer ---


@pytest.mark.parametrize(
    ("mode", "key", "expected"),
    [
        ("auto", "sk-x", OpenAIScriptWriter),
        ("auto", None, type(None)),
        ("openai", None, type(None)),
        ("fake", None, FakeScriptWriter),
        ("off", "sk-x", type(None)),
    ],
)
def test_build_writer(mode, key, expected):
    settings = Settings(_env_file=None, script_writer=mode, openai_api_key=key)
    assert isinstance(build_writer(settings), expected)


def test_openai_writer_label_names_the_model():
    settings = Settings(_env_file=None, openai_api_key="sk-x", script_model="gpt-test")
    assert build_writer(settings).label == "openai:gpt-test"
