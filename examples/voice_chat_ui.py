"""Voice chat demo: a chatbot that starts speaking while it is still writing (Gradio).

    uv sync --extra demo                              # gradio + websockets; needs ffmpeg
    uv run uvicorn app.main:app --port 8000           # terminal 1: the TTS server
    uv run python examples/voice_chat_ui.py           # terminal 2: http://127.0.0.1:7860

This app plays the chatbot backend. For each question it streams an answer from an
LLM (OpenAI, or a free **Fake LLM**), forwards every text delta to the TTS server's
WebSocket (/v1/speech/ws), and streams the PCM it gets back to the browser while the
answer is still being written. **Stop**, or asking something new, cancels the speech at
once (barge-in).

Gradio's streaming audio player adds about 0.7-1 s before playback (it re-encodes the
stream to AAC for the browser), so the audio starts later here than the TTS server's
own latency. ``scripts/bench_latency.py --mode ws`` measures the server alone.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import itertools
import json
import os
import time
import wave
from collections.abc import AsyncIterator

import gradio as gr
import httpx
from websockets.asyncio.client import connect

SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Answer in one to three short sentences of plain, "
    "spoken English. No markdown, lists, emoji or code."
)
DEFAULT_MODEL = os.environ.get("DEMO_CHAT_MODEL", "gpt-4o-mini")
FALLBACK_VOICES = ["af_heart", "af_bella", "am_michael", "am_fenrir", "bf_emma", "bm_george"]
FAKE_WORDS_PER_SECOND = 15.0

FAKE_ANSWERS = [
    "Sure! Text to speech has changed a lot over the last few years, mostly because "
    "models now generate audio directly. That makes voices sound far more natural.",
    "Great question. A streaming voice starts talking as soon as the first sentence is "
    "ready, so you hear the answer while the rest is still being written.",
    "**Here's the short version:** I cut the reply into chunks, synthesize each one, and "
    "send the audio right away. See https://example.com for the long version.",
    "Hi! The trick is to start with a short first chunk, then make the chunks longer as "
    "audio builds up, so playback never runs dry.",
]

# Sessions supersede their own earlier answers: a new question (or Stop) bumps this, and
# the running answer notices and cancels its speech.
_ACTIVE: dict[str, int] = {}
_TOKENS = itertools.count(1)


# --- pieces with no UI (unit-tested) ---------------------------------------------


def to_openai_messages(history: list[dict]) -> list[dict]:
    """Gradio chat history -> OpenAI chat messages, with the voice system prompt."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history:
        content = item.get("content")
        if item.get("role") in ("user", "assistant") and isinstance(content, str) and content:
            messages.append({"role": item["role"], "content": content})
    return messages


def pick_fake_answer(question: str) -> str:
    return FAKE_ANSWERS[sum(map(ord, question)) % len(FAKE_ANSWERS)]


async def fake_llm(question: str, wps: float = FAKE_WORDS_PER_SECOND) -> AsyncIterator[str]:
    """Stream a canned answer word by word, at LLM-like speed (no API key needed)."""
    words = pick_fake_answer(question).split(" ")
    for i, word in enumerate(words):
        await asyncio.sleep(1 / wps)
        yield word if i == len(words) - 1 else word + " "


async def openai_llm(messages: list[dict], model: str) -> AsyncIterator[str]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    stream = await client.chat.completions.create(model=model, messages=messages, stream=True)
    async for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content


def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """One streamed chunk for gr.Audio: 16-bit mono WAV (Gradio reads it natively)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm[: len(pcm) - len(pcm) % 2])
    return buf.getvalue()


# --- the app -----------------------------------------------------------------------


def build_app(tts_ws_url: str, tts_http_url: str, openai_ok: bool) -> gr.Blocks:
    voices = _fetch_voices(tts_http_url)

    async def respond(message, history, voice, chunking, use_fake, model,
                      request: gr.Request):
        if not message or not message.strip():
            yield gr.skip(), gr.skip(), gr.skip(), gr.skip()
            return
        session = request.session_hash or "default"
        token = next(_TOKENS)
        _ACTIVE[session] = token  # supersedes an answer still speaking in this tab
        history = [*(history or []), {"role": "user", "content": message},
                   {"role": "assistant", "content": ""}]
        yield history, None, "Thinking...", ""

        started = time.perf_counter()
        marks: dict[str, float] = {}

        def stamp(name: str) -> None:
            marks.setdefault(name, time.perf_counter() - started)

        def metrics(done: bool = False) -> str:
            parts = [f"{label} **{marks[key]:.2f} s**" for key, label in
                     (("llm", "LLM first token"), ("chunk", "first chunk to TTS"),
                      ("audio", "first audio from TTS")) if key in marks]
            tail = " (playback starts ~0.8 s later in Gradio's player)" if done and \
                "audio" in marks else ""
            return " · ".join(parts) + tail if parts else "..."

        try:
            ws = await connect(tts_ws_url, max_size=None)
        except OSError as exc:
            history[-1]["content"] = f"(Can't reach the TTS server at {tts_ws_url}: {exc})"
            yield history, gr.skip(), "", gr.skip()
            return

        events: asyncio.Queue = asyncio.Queue()
        finished = False
        tasks: list[asyncio.Task] = []
        try:
            await ws.send(json.dumps({"type": "start", "voice": voice, "chunking": chunking}))
            ready = json.loads(await ws.recv())
            if ready.get("type") != "ready":
                history[-1]["content"] = f"(TTS refused the session: {ready.get('detail')})"
                yield history, gr.skip(), "", gr.skip()
                return
            sample_rate = ready["sample_rate"]

            async def write() -> None:
                source = (fake_llm(message) if use_fake
                          else openai_llm(to_openai_messages(history[:-1]), model))
                try:
                    async for delta in source:
                        stamp("llm")
                        await events.put(("text", delta))
                        await ws.send(json.dumps({"type": "text", "text": delta}))
                except Exception as exc:  # noqa: BLE001 - shown in the chat
                    await events.put(("error", f"LLM error: {exc}"))
                await ws.send(json.dumps({"type": "flush"}))

            async def listen() -> None:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        stamp("audio")
                        await events.put(("audio", raw))
                        continue
                    event = json.loads(raw)
                    if event["type"] == "sentence":
                        stamp("chunk")
                    elif event["type"] in ("done", "cancelled"):
                        await events.put(("done", None))
                        return
                    elif event["type"] == "error":
                        await events.put(("error", f"TTS error: {event['detail']}"))
                        if "turn" in event:
                            await events.put(("done", None))
                            return

            tasks = [asyncio.create_task(write()), asyncio.create_task(listen())]
            while True:
                if _ACTIVE.get(session) != token:
                    break  # Stop, or a newer question: cancel this answer's speech
                try:
                    kind, value = await asyncio.wait_for(events.get(), timeout=0.1)
                except TimeoutError:
                    continue
                if kind == "text":
                    history[-1]["content"] += value
                    yield history, gr.skip(), metrics(), gr.skip()
                elif kind == "audio":
                    yield gr.skip(), wav_bytes(value, sample_rate), metrics(), gr.skip()
                elif kind == "error":
                    history[-1]["content"] += f"\n\n({value})"
                    yield history, gr.skip(), metrics(), gr.skip()
                elif kind == "done":
                    finished = True
                    break
            yield history, gr.skip(), metrics(done=True), gr.skip()
        finally:
            for task in tasks:
                task.cancel()
            if not finished:
                try:
                    await ws.send(json.dumps({"type": "cancel"}))  # barge-in
                except Exception:  # noqa: BLE001 - the socket may be gone already
                    pass
            await ws.close()

    def stop(request: gr.Request):
        _ACTIVE[request.session_hash or "default"] = next(_TOKENS)
        return None  # clears the player, so buffered audio stops too

    async def warm_voice(voice):
        """Load a newly picked voice in the background, so its first answer isn't slow."""
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                await client.post(f"{tts_http_url}/v1/speech",
                                  json={"text": "Hi.", "voice": voice, "format": "pcm"},
                                  headers={"Cache-Control": "no-cache"})
        except httpx.HTTPError:
            pass

    with gr.Blocks(title="Voice chat") as demo:
        gr.Markdown("## Voice chat\nAsk something: the answer is spoken while it is being "
                    "written. **Stop** (or a new question) interrupts it.")
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(height=420, label="Chat")
                with gr.Row():
                    box = gr.Textbox(placeholder="Ask something...", show_label=False,
                                     scale=5, autofocus=True)
                    send = gr.Button("Send", variant="primary", scale=1)
                    stop_btn = gr.Button("Stop", variant="stop", scale=1)
                speech = gr.Audio(streaming=True, autoplay=True, label="Voice",
                                  interactive=False)
                stats = gr.Markdown("")
            with gr.Column(scale=1):
                voice = gr.Dropdown(voices, value=voices[0], label="Voice")
                chunking = gr.Radio(["adaptive", "fast_start", "sentence"], value="adaptive",
                                    label="Chunking")
                use_fake = gr.Checkbox(value=not openai_ok, label="Fake LLM (no API key)")
                model = gr.Textbox(value=DEFAULT_MODEL, label="OpenAI chat model")
                if not openai_ok:
                    gr.Markdown("OpenAI isn't usable with the current key, so the Fake LLM "
                                "is on. Set a valid OPENAI_API_KEY to chat for real.")

        inputs = [box, chat, voice, chunking, use_fake, model]
        outputs = [chat, speech, stats, box]
        answers = [box.submit(respond, inputs, outputs, concurrency_limit=2),
                   send.click(respond, inputs, outputs, concurrency_limit=2)]
        stop_btn.click(stop, None, speech, cancels=answers)
        voice.change(warm_voice, voice, None)
    return demo


def _fetch_voices(tts_http_url: str) -> list[str]:
    try:
        r = httpx.get(f"{tts_http_url}/v1/voices", params={"provider": "kokoro"}, timeout=5)
        r.raise_for_status()
        voices = [v["id"] for v in r.json()["voices"]]
        return voices or FALLBACK_VOICES
    except (httpx.HTTPError, KeyError, ValueError):
        print(f"(couldn't load voices from {tts_http_url}; using a built-in list)")
        return FALLBACK_VOICES


def _openai_usable() -> bool:
    """Whether the configured key works (listing models is free)."""
    try:
        from openai import OpenAI

        OpenAI(timeout=10).models.list()
        return True
    except Exception:  # noqa: BLE001 - no key, bad key, offline: use the Fake LLM
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice chat demo (Gradio)")
    parser.add_argument("--tts-url", default="http://127.0.0.1:8000",
                        help="TTS server base URL (default http://127.0.0.1:8000)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    http_url = args.tts_url.rstrip("/")
    ws_url = http_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url += "/v1/speech/ws"

    from dotenv import load_dotenv  # the TTS server's .env also holds OPENAI_API_KEY

    load_dotenv()
    demo = build_app(ws_url, http_url, _openai_usable())
    # Local only: no public share link (it would let anyone spend your OpenAI quota).
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port, share=False
    )


if __name__ == "__main__":
    main()
