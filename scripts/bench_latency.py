"""Measure streaming latency against a running server.

    uv run python scripts/bench_latency.py                     # http mode, kokoro
    uv run python scripts/bench_latency.py --runs 5 --fast-start
    uv run python scripts/bench_latency.py --mode ws --chunking all --wps 5,15,40
    uv run python scripts/bench_latency.py --base-url http://127.0.0.1:8765

**http mode** calls POST /v1/speech/stream (format pcm) for each text and reports:

- **TTFA**: time to first audio, from sending the request to the first audio byte.
- **RTF**: real-time factor = generation time / audio duration. Below 1.0 means audio is
  produced faster than it plays.
- **gaps**: playback underruns. Playback starts at the first byte and runs at 1x; a
  chunk that arrives after the audio before it has finished playing is a gap (a
  stutter), and the total silence added is reported.

**ws mode** plays a chatbot on WS /v1/speech/ws: each text is sent word by word at
``--wps`` words per second, like an LLM, and what a listener would hear is measured:

- **hear s**: from the moment the first sentence has been fully sent to the first
  *audible* sample (not the first byte, since chunks may start with silence).
- **gaps / stall s**: playback underruns, as above.
- **dead**: silences of 0.4 s or more inside the speech (unnatural pauses).
- **chunks / avg / word**: how many chunks, their average length, and how many ended
  mid-sentence at a word boundary (the cuts that can hurt prosody).
- A fit of generation time ``a + b * chars`` over all uncached chunks.

The cache is bypassed unless ``--use-cache`` is given. Medians over ``--runs`` are
printed (after one warm-up request), so a single slow run doesn't skew the table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from array import array

import httpx

from app.core.chunker import split_sentences

SAMPLE_RATE = 24_000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # 16-bit mono
AUDIBLE = 200  # sample amplitude counted as sound (matches app.core.audio.EDGE_THRESHOLD)
DEAD_AIR_S = 0.4

TEXTS = {
    "short": "Sure, I can help with that.",
    "medium": "The weather today is sunny with a light breeze, perfect for a walk in the park.",
    "long first sentence": (
        "When you think about how speech synthesis has changed over the last few years, "
        "the biggest shift is that models now generate audio directly. That makes them "
        "sound natural."
    ),
    "paragraph": (
        "Hello! Streaming lets a chatbot start talking early. Each sentence is spoken as "
        "soon as it is ready. Later sentences are synthesized while earlier ones play. "
        "That keeps the wait short."
    ),
}

# Chatbot-style answers for ws mode, chosen to stress chunking.
CHAT_TEXTS = {
    **TEXTS,
    "long, no comma": (
        "The weather today is sunny with a light breeze that makes it perfect for a long "
        "walk in the park near the river. Bring some water."
    ),
    "tiny first": (
        "Hi! I can definitely help you plan a relaxing trip to the coast this weekend. "
        "Let's start with the route and where to stay."
    ),
    "decimals and abbr.": (
        "It takes about 3.5 seconds, e.g. on Dr. Smith's laptop in the U.S. office. "
        "That is fast enough for most people."
    ),
    "markdown": (
        "**Sure!** Here are the steps: first install the package, then start the server. "
        "See https://example.com/docs for more details."
    ),
}

CHUNKING = {
    "sentence": {"fast_start": False},
    "fast_start": {"fast_start": True},
    "adaptive": {"chunking": "adaptive"},
}


# --- http mode ---------------------------------------------------------------------


def measure(client: httpx.Client, text: str, provider: str, voice: str | None,
            fast_start: bool, use_cache: bool = False) -> dict:
    body = {"text": text, "provider": provider, "format": "pcm", "fast_start": fast_start}
    if voice:
        body["voice"] = voice
    started = time.perf_counter()
    first: float | None = None
    received = 0
    gaps = 0
    silence = 0.0
    headers = {} if use_cache else {"Cache-Control": "no-cache"}
    with client.stream("POST", "/v1/speech/stream", json=body, headers=headers) as resp:
        if resp.status_code != 200:
            resp.read()
            raise SystemExit(f"HTTP {resp.status_code}: {resp.text}")
        for chunk in resp.iter_bytes():
            if not chunk:
                continue
            now = time.perf_counter()
            if first is None:
                first = now
            else:
                # When would the audio received so far finish playing (plus stalls)?
                playback_end = first + received / BYTES_PER_SECOND + silence
                if now > playback_end:
                    gaps += 1
                    silence += now - playback_end
            received += len(chunk)
    total = time.perf_counter() - started
    audio_s = received / BYTES_PER_SECOND
    return {
        "ttfa": (first - started) if first else float("nan"),
        "total": total,
        "audio": audio_s,
        "rtf": total / audio_s if audio_s else float("nan"),
        "gaps": gaps,
        "silence": silence,
    }


def run_http(args: argparse.Namespace) -> int:
    with httpx.Client(base_url=args.base_url, timeout=300) as client:
        print("warming up...")
        measure(client, "Warm up.", args.provider, args.voice, args.fast_start)

        print(f"\nprovider={args.provider} fast_start={args.fast_start} "
              f"use_cache={args.use_cache} runs={args.runs}\n")
        print(f"{'text':<20} {'chars':>5} {'TTFA s':>7} {'total s':>8} {'audio s':>8} "
              f"{'RTF':>5} {'gaps':>5} {'stall s':>8}")
        for name, text in TEXTS.items():
            runs = [measure(client, text, args.provider, args.voice, args.fast_start,
                            args.use_cache)
                    for _ in range(args.runs)]

            def med(key: str, runs: list[dict] = runs) -> float:
                return statistics.median(r[key] for r in runs)

            print(f"{name:<20} {len(text):>5} {med('ttfa'):>7.2f} {med('total'):>8.2f} "
                  f"{med('audio'):>8.2f} {med('rtf'):>5.2f} {med('gaps'):>5.0f} "
                  f"{med('silence'):>8.2f}")
    return 0


# --- ws mode -----------------------------------------------------------------------


def _words(text: str) -> list[str]:
    parts = text.split(" ")
    return [p + " " for p in parts[:-1]] + parts[-1:]


def _samples(pcm: bytes) -> array:
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    return samples


def _first_audible_s(pcm: bytes) -> float | None:
    for i, s in enumerate(_samples(pcm)):
        if abs(s) >= AUDIBLE:
            return i / SAMPLE_RATE
    return None


def _dead_air(pcm: bytes) -> list[float]:
    """Lengths of silences >= DEAD_AIR_S between the first and last sound."""
    samples = _samples(pcm)
    win = SAMPLE_RATE // 100  # 10 ms
    loud = [max((abs(s) for s in samples[i : i + win]), default=0) >= AUDIBLE
            for i in range(0, len(samples), win)]
    if True not in loud:
        return []
    first = loud.index(True)
    last = len(loud) - 1 - loud[::-1].index(True)
    runs, current = [], 0
    for flag in loud[first : last + 1]:
        if flag:
            if current * 0.01 >= DEAD_AIR_S:
                runs.append(current * 0.01)
            current = 0
        else:
            current += 1
    return runs


async def measure_ws(url: str, text: str, wps: float, chunking: str, provider: str,
                     voice: str | None, use_cache: bool) -> dict:
    from websockets.asyncio.client import connect

    start = {"type": "start", "cache": use_cache, **CHUNKING[chunking]}
    if provider:
        start["provider"] = provider
    if voice:
        start["voice"] = voice
    first_sentence = split_sentences(text)[0]
    boundary = text.index(first_sentence) + len(first_sentence)

    marks: dict[str, float] = {}
    arrivals: list[tuple[float, int]] = []  # (time, bytes) of each audio message
    sentences: list[tuple[float, dict]] = []
    first_audio_after: list[float | None] = []
    pcm = bytearray()

    async with connect(url, max_size=None) as ws:
        await ws.send(json.dumps(start))
        ready = json.loads(await ws.recv())
        if ready["type"] != "ready":
            raise SystemExit(f"refused: {ready}")
        t0 = time.perf_counter()

        async def send() -> None:
            sent = 0
            for word in _words(text):
                await ws.send(json.dumps({"type": "text", "text": word}))
                sent += len(word)
                if "first_sentence" not in marks and sent >= boundary:
                    marks["first_sentence"] = time.perf_counter() - t0
                await asyncio.sleep(1 / wps)
            await ws.send(json.dumps({"type": "flush"}))

        sender = asyncio.create_task(send())
        async for message in ws:
            now = time.perf_counter() - t0
            if isinstance(message, bytes):
                arrivals.append((now, len(message)))
                pcm.extend(message)
                if first_audio_after and first_audio_after[-1] is None:
                    first_audio_after[-1] = now
                continue
            event = json.loads(message)
            if event["type"] == "sentence":
                sentences.append((now, event))
                first_audio_after.append(None)
            elif event["type"] in ("done", "error", "cancelled"):
                if event["type"] == "error":
                    raise SystemExit(f"error: {event}")
                break
        await sender

    # Playback starts with the first byte at 1x; count underruns.
    gaps, stall = 0, 0.0
    if arrivals:
        play_start, received = arrivals[0]
        for when, size in arrivals[1:]:
            # When does the audio received so far (plus earlier stalls) finish playing?
            playback_end = play_start + received / BYTES_PER_SECOND + stall
            if when > playback_end:
                gaps += 1
                stall += when - playback_end
            received += size
    audible = _first_audible_s(bytes(pcm))
    hear = (arrivals[0][0] + audible - marks.get("first_sentence", 0.0)
            if arrivals and audible is not None else float("nan"))
    dead = _dead_air(bytes(pcm))
    gen = [
        (event["chars"], audio_at - at)
        for (at, event), audio_at in zip(sentences, first_audio_after, strict=False)
        if audio_at is not None and not event["cached"]
    ]
    chars = [event["chars"] for _, event in sentences]
    return {
        "hear": hear,
        "gaps": gaps,
        "stall": stall,
        "dead": len(dead),
        "dead_s": sum(dead),
        "chunks": len(sentences),
        "avg": statistics.mean(chars) if chars else 0.0,
        "word": sum(1 for _, e in sentences if e.get("boundary") == "word"),
        "gen": gen,
        "audio": len(pcm) / BYTES_PER_SECOND,
    }


def _fit(points: list[tuple[int, float]]) -> tuple[float, float] | None:
    if len(points) < 3:
        return None
    xs, ys = zip(*points, strict=True)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    var = sum((x - mx) ** 2 for x in xs)
    if var == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in points) / var
    return my - b * mx, b


async def run_ws(args: argparse.Namespace) -> int:
    url = args.base_url.replace("http://", "ws://").replace("https://", "wss://")
    url = url.rstrip("/") + "/v1/speech/ws"
    modes = list(CHUNKING) if args.chunking == "all" else [args.chunking]
    speeds = [float(x) for x in args.wps.split(",")]
    print("warming up...")
    await measure_ws(url, "Warm up.", 50, "sentence", args.provider, args.voice, False)

    for wps in speeds:
        print(f"\n=== LLM at {wps:g} words/s, provider={args.provider}, "
              f"use_cache={args.use_cache}, runs={args.runs} ===")
        print(f"{'mode':<11} {'text':<20} {'hear s':>6} {'gaps':>4} {'stall s':>7} "
              f"{'dead':>4} {'chunks':>6} {'avg':>5} {'word':>4}")
        for mode in modes:
            rows, gen_points = [], []
            for name, text in CHAT_TEXTS.items():
                runs = []
                for _ in range(args.runs):
                    try:
                        runs.append(await measure_ws(url, text, wps, mode, args.provider,
                                                     args.voice, args.use_cache))
                    except SystemExit as exc:
                        print(f"{mode:<11} {name:<20} skipped: {exc}")
                        break
                if not runs:
                    continue
                gen_points += [p for r in runs for p in r["gen"]]

                def med(key: str, runs: list[dict] = runs) -> float:
                    return statistics.median(r[key] for r in runs)

                row = {k: med(k) for k in ("hear", "gaps", "stall", "dead", "chunks", "avg",
                                           "word")}
                rows.append(row)
                if args.verbose:
                    print(f"{mode:<11} {name:<20} {row['hear']:>6.2f} {row['gaps']:>4.0f} "
                          f"{row['stall']:>7.2f} {row['dead']:>4.0f} {row['chunks']:>6.0f} "
                          f"{row['avg']:>5.0f} {row['word']:>4.0f}")
            if not rows:
                continue
            hears = [r["hear"] for r in rows]
            fit = _fit(gen_points)
            fit_text = f"gen ~ {fit[0]:.2f} s + {fit[1] * 1000:.0f} ms/char" if fit else ""
            print(f"{mode:<11} {'ALL (median/max)':<20} "
                  f"{statistics.median(hears):>4.2f}/{max(hears):.2f} "
                  f"{sum(r['gaps'] for r in rows):>3.0f} "
                  f"{sum(r['stall'] for r in rows):>7.2f} "
                  f"{sum(r['dead'] for r in rows):>4.0f} "
                  f"{sum(r['chunks'] for r in rows):>6.0f} "
                  f"{statistics.mean(r['avg'] for r in rows):>5.0f} "
                  f"{sum(r['word'] for r in rows):>4.0f}   {fit_text}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Streaming latency benchmark")
    parser.add_argument("--mode", choices=["http", "ws"], default="http")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--provider", default="kokoro")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--runs", type=int, default=None,
                        help="runs per text (default: 3 for http, 1 for ws)")
    parser.add_argument("--fast-start", action="store_true",
                        help="http: let the first chunk break at a comma")
    parser.add_argument("--use-cache", action="store_true",
                        help="don't bypass the cache (runs after the first become cache hits)")
    parser.add_argument("--chunking", default="all", choices=[*CHUNKING, "all"],
                        help="ws: chunking mode(s) to compare (default: all)")
    parser.add_argument("--wps", default="15", help="ws: LLM speed(s) in words/s, e.g. 5,15,40")
    parser.add_argument("--verbose", action="store_true", help="ws: one row per text")
    args = parser.parse_args()
    if args.runs is None:
        args.runs = 3 if args.mode == "http" else 1

    try:
        httpx.get(args.base_url.rstrip("/") + "/health", timeout=5)
    except httpx.HTTPError:
        print(f"Can't reach {args.base_url}; start the server first.")
        return 2
    if args.mode == "ws":
        return asyncio.run(run_ws(args))
    return run_http(args)


if __name__ == "__main__":
    sys.exit(main())
