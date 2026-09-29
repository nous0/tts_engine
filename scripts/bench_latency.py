"""Measure streaming latency against a running server.

    uv run python scripts/bench_latency.py                    # http mode, kokoro
    uv run python scripts/bench_latency.py --runs 5 --fast-start
    uv run python scripts/bench_latency.py --base-url http://127.0.0.1:8765

For each text it calls POST /v1/speech/stream (format pcm, cache bypassed) and reports:

- **TTFA**: time to first audio, from sending the request to the first audio byte.
- **RTF**: real-time factor = generation time / audio duration. Below 1.0 means audio is
  produced faster than it plays.
- **gaps**: playback underruns. Playback starts at the first byte and runs at 1x; a
  chunk that arrives after the audio before it has finished playing is a gap (a
  stutter), and the total silence added is reported.

Medians over ``--runs`` (after one warm-up request) are printed, so a single slow run
doesn't skew the table.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import httpx

SAMPLE_RATE = 24_000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # 16-bit mono

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


def measure(client: httpx.Client, text: str, provider: str, voice: str | None,
            fast_start: bool) -> dict:
    body = {"text": text, "provider": provider, "format": "pcm", "fast_start": fast_start}
    if voice:
        body["voice"] = voice
    started = time.perf_counter()
    first: float | None = None
    received = 0
    gaps = 0
    silence = 0.0
    with client.stream("POST", "/v1/speech/stream", json=body,
                       headers={"Cache-Control": "no-cache"}) as resp:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Streaming latency benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--provider", default="kokoro")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--fast-start", action="store_true",
                        help="let the first chunk break at a comma (if the server supports it)")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=300) as client:
        try:
            client.get("/health")
        except httpx.HTTPError:
            print(f"Can't reach {args.base_url}; start the server first.")
            return 2
        print("warming up...")
        measure(client, "Warm up.", args.provider, args.voice, args.fast_start)

        print(f"\nprovider={args.provider} fast_start={args.fast_start} runs={args.runs}\n")
        print(f"{'text':<20} {'chars':>5} {'TTFA s':>7} {'total s':>8} {'audio s':>8} "
              f"{'RTF':>5} {'gaps':>5} {'stall s':>8}")
        for name, text in TEXTS.items():
            runs = [measure(client, text, args.provider, args.voice, args.fast_start)
                    for _ in range(args.runs)]

            def med(key: str, runs: list[dict] = runs) -> float:
                return statistics.median(r[key] for r in runs)

            print(f"{name:<20} {len(text):>5} {med('ttfa'):>7.2f} {med('total'):>8.2f} "
                  f"{med('audio'):>8.2f} {med('rtf'):>5.2f} {med('gaps'):>5.0f} "
                  f"{med('silence'):>8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
