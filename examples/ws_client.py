"""Reference client for WS /v1/speech/ws: speak text while it is still being "generated".

It plays the role of a chatbot backend: it sends an answer to the server word by word,
at LLM-like speed, and receives audio for each sentence while later words are still
being sent. The audio is saved to a WAV file, and the timeline shows how soon speech
starts. It needs only the ``websockets`` package, not this project's code, so it doubles
as an integration example.

    uv run uvicorn app.main:app --port 8000          # terminal 1
    uv run python examples/ws_client.py              # terminal 2
    uv run python examples/ws_client.py --words-per-second 5 --text "Your own text."
    uv run python examples/ws_client.py --cancel-after 1.5    # barge-in test
    uv run python examples/ws_client.py --no-fast-start --no-cache

The key number is **first audio after the first sentence**: how long after the first
sentence was fully sent the first audio arrived. That is the delay a user would hear.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

from websockets.asyncio.client import connect

DEFAULT_TEXT = (
    "Text to speech has changed a lot over the last few years, mostly because models "
    "now generate audio directly. That makes voices sound far more natural. It also "
    "means a chatbot can start talking before it has finished thinking."
)


def words(text: str) -> list[str]:
    """Split into word tokens that keep their trailing space, like LLM deltas."""
    parts = text.split(" ")
    return [p + " " for p in parts[:-1]] + parts[-1:]


def first_sentence_end(text: str) -> int:
    ends = [text.find(c) for c in ".!?" if c in text]
    return min(ends) + 1 if ends else len(text)


async def run(args: argparse.Namespace) -> int:
    text = args.text or DEFAULT_TEXT
    start = {"type": "start", "fast_start": not args.no_fast_start, "cache": not args.no_cache}
    if args.voice:
        start["voice"] = args.voice
    if args.provider:
        start["provider"] = args.provider

    marks: dict[str, float] = {}
    pcm = bytearray()
    cached = 0
    sample_rate = 24_000

    async with connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps(start))
        ready = json.loads(await ws.recv())
        if ready["type"] != "ready":
            print(f"refused: {ready.get('detail')}")
            return 1
        sample_rate = ready["sample_rate"]
        print(f"ready: provider={ready['provider']} voice={ready['voice']} "
              f"{sample_rate} Hz pcm, fast_start={start['fast_start']}, cache={start['cache']}")
        t0 = time.perf_counter()

        def now() -> float:
            return time.perf_counter() - t0

        async def send_text() -> None:
            sent = 0
            boundary = first_sentence_end(text)
            for token in words(text):
                await ws.send(json.dumps({"type": "text", "text": token}))
                sent += len(token)
                if "first_sentence_sent" not in marks and sent >= boundary:
                    marks["first_sentence_sent"] = now()
                await asyncio.sleep(1 / args.words_per_second)
            await ws.send(json.dumps({"type": "flush"}))
            marks["all_text_sent"] = now()

        async def cancel_later() -> None:
            await asyncio.sleep(args.cancel_after)
            await ws.send(json.dumps({"type": "cancel"}))
            marks["cancel_sent"] = now()
            print(f"{now():6.2f}s  -> cancel sent")

        async def receive() -> None:
            nonlocal cached
            async for message in ws:
                if isinstance(message, bytes):
                    if "first_audio" not in marks:
                        marks["first_audio"] = now()
                        print(f"{now():6.2f}s  first audio")
                    pcm.extend(message)
                    continue
                event = json.loads(message)
                kind = event["type"]
                if kind == "sentence":
                    cached += event["cached"]
                    print(f"{now():6.2f}s  sentence #{event['index']} "
                          f"({event['chars']} chars, cached={event['cached']})")
                elif kind in ("done", "cancelled") or (kind == "error" and "turn" in event):
                    print(f"{now():6.2f}s  {kind}" + (f": {event['detail']}"
                                                      if kind == "error" else ""))
                    return
                elif kind == "error":
                    print(f"{now():6.2f}s  error: {event['detail']}")

        receiver = asyncio.create_task(receive())
        sender = asyncio.create_task(send_text())
        canceller = (asyncio.create_task(cancel_later())
                     if args.cancel_after is not None else None)
        await receiver
        for task in (sender, canceller):
            if task is not None:
                task.cancel()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(pcm))

    audio_s = len(pcm) / 2 / sample_rate
    print(f"\nsaved {out} ({audio_s:.1f} s of audio, {cached} cached sentence(s))")
    if "first_sentence_sent" in marks and "first_audio" in marks:
        delay = marks["first_audio"] - marks["first_sentence_sent"]
        print(f"first sentence fully sent at {marks['first_sentence_sent']:.2f}s, "
              f"first audio at {marks['first_audio']:.2f}s")
        if delay >= 0:
            print(f"-> first audio {delay:.2f}s after the first sentence")
        else:
            print(f"-> first audio {-delay:.2f}s BEFORE the first sentence was complete "
                  f"(fast_start and/or cache)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="WebSocket TTS reference client")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/speech/ws")
    parser.add_argument("--text", default=None, help="text to speak (default: a demo answer)")
    parser.add_argument("--words-per-second", type=float, default=15.0,
                        help="how fast the fake LLM sends words (default 15)")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--voice", default=None)
    parser.add_argument("--no-fast-start", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cancel-after", type=float, default=None, metavar="SECONDS",
                        help="send a cancel (barge-in) after this many seconds")
    parser.add_argument("--out", default="output/ws_client.wav")
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except OSError as exc:
        print(f"Can't connect to {args.url} ({exc}). Start the server first:")
        print("  uv run uvicorn app.main:app --port 8000")
        return 2


if __name__ == "__main__":
    sys.exit(main())
