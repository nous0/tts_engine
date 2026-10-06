# CONTEXT — current working state

Read this before picking up work in this repo. It says where things actually
stand. Update it at the end of each work session; this file goes stale fast if
nobody touches it.

Last checked: 2026-09-29.

## Where things actually stand

- **Phase 5 (local hardening) finished on 2026-09-28**, in 5 commits:
  - logging;
  - streaming fixes and Kokoro concurrency;
  - timeouts, error classes and voice validation;
  - job queue, cancel and recovery;
  - disk cache.

  **208 tests pass**, `ruff check .` is clean.
- **Verified end-to-end against a live server with Kokoro (2026-09-28):**
  - A 3-sentence WAV stream has exactly 1 `RIFF` header. Before, there was one per
    sentence, heard as a click.
  - Cache: `/v1/speech` miss 4.5 s → hit 0.03 s. Re-rendering `script_trio.txt` via the
    CLI went from 45.7 s to 0.3 s.
  - Unknown voice / `.pt` voice / `flac` stream / unknown provider → 400.
  - OpenAI with `OPENAI_TIMEOUT=0.001` → 504 (SDK retry visible in the log). A bad
    OpenAI key → 502 with a clear message.
  - 3 parallel Kokoro requests run one at a time (they finished about 6 s apart), and
    `GET /v1/jobs` answered in 9 ms meanwhile.
  - A second podcast stays `queued`. Cancel works: queued is immediate, running takes
    about 10 ms, and cancelling again gives 409.
  - Hard-killing the server mid-render → after restart the job is `error` "Interrupted
    by a server restart".
  - A client disconnecting mid-stream → log "producer stopped: consumer went away".
- **Run a single uvicorn process.** Jobs live in process memory; `--workers N` is
  unsupported (documented in README and CLAUDE.md).
- **`.env` is on `DEFAULT_PROVIDER=kokoro` / `DEFAULT_VOICE=af_heart`.** Gemini was
  removed on 2026-09-22; ElevenLabs is out of scope.
- **ffmpeg isn't installed** on this machine, so only `wav` and `pcm` output work here.
- **The first Kokoro request after a server start takes about 40 s** (lazy model load).
  Preloading is a known to-do.
- **`kokoro-tts/`** (vendored upstream checkout, its own git repo) is gitignored; the app
  only uses the `kokoro` PyPI package.
- **Fixed a crash found while testing:** occasional "access violation" (segfault, exit
  139) in the test suite. Cause: `JobStore.close()` closed SQLite while a cancelled task's
  query was still running in its thread (the startup cleanup added in Phase 5). The lock
  is now held until the thread finishes. There is a regression test, and 15/15 full runs
  were clean afterwards.

## What's done

- **Phase 1–2:** provider abstraction, OpenAI provider, `/v1/speech`,
  `/v1/speech/stream`, sentence chunker, `/health`.
- **Phase 3:** `audio.py` (concat/normalize/encode), `podcast.py` (script parsing and
  rendering), `jobs.py` (SQLite job store with an async worker), `POST /v1/podcast`,
  `GET /v1/jobs[/{id}[/audio]]`, and `examples/generate_podcast.py`.
- **Phase 4:** Kokoro local provider (the default), `GET /v1/voices` with per-voice gender,
  podcast casting by name gender (`app/core/names.py`) with a `genders` override, and a
  Streamlit voice picker fed by `/v1/voices`. Gemini removed.
- **Phase 5:**
  - Request ids and logging.
  - One WAV header per stream; the first chunk is primed so early errors get a real
    status; no error text inside audio; `flac` stream → 400.
  - Kokoro: its own executor, one synthesis at a time, a model-load lock, and no thread
    leak on disconnect.
  - `ProviderError` kinds → 400/502/503/504 via central handlers; OpenAI timeout 30 s
    (was the SDK's 600 s) with SDK retries.
  - Voice validation, and per-provider default voices (`OPENAI_VOICE`).
  - Real job queue (`MAX_CONCURRENT_JOBS`), `POST /v1/jobs/{id}/cancel`, restart
    recovery, hourly cleanup (`JOBS_RETENTION_DAYS`).
  - Disk audio cache (`CACHE_*`, `X-Cache` header), also used by the podcast CLI.
- A static route for `./output/` is **not needed**: `GET /v1/jobs/{id}/audio` already
  serves results (409 before done, 410 if the file is gone).

- **`scripts/smoke_test.py` (2026-09-29):** end-to-end checks against a running
  server, with real Kokoro and no OpenAI calls. 22/22 pass, and the two-step restart
  check passes. Run it after any change to streaming, jobs, the cache or error handling.

## Decisions (2026-09-29)

- **Language: English only.** Kokoro covers it; no Vietnamese provider is planned. Kokoro
  supports en-US/en-GB, es, fr, hi, it, pt-br, ja and zh, but not Vietnamese. Revisit if
  the audience changes.
- **Priority: chatbot first**, then podcast, then packaging.
- **Chatbot integration: none exists yet**, so we build a complete demo (OpenAI chat
  streaming → our TTS → speakers).

## Measured baseline (Kokoro on this CPU, 2026-09-29)

- Real-time factor about **0.73** (1 s of audio takes about 0.73 s to make); short
  sentence about 1.0–1.5 s to audio; model load 9.7 s when already downloaded.
- **Streaming baseline** (`scripts/bench_latency.py`, HTTP stream, pcm, no cache, median
  of 3, before Phase 6 optimizations):

  | text | chars | TTFA s | total s | audio s | RTF | gaps | stall s |
  |---|---|---|---|---|---|---|---|
  | short | 27 | 1.40 | 1.40 | 2.05 | 0.68 | 0 | 0.00 |
  | medium | 79 | 3.07 | 3.07 | 4.90 | 0.63 | 0 | 0.00 |
  | long first sentence | 171 | 4.78 | 6.26 | 10.68 | 0.59 | 0 | 0.00 |
  | paragraph | 184 | 0.97 | 9.03 | 13.68 | 0.66 | 1 | 0.61 |

  TTFA is the time to synthesize the whole first sentence, so it grows with its length.
  The paragraph stalls because "Hello!" plays for less time than the next sentence takes
  to make. Splitting chunks smaller lowers TTFA but can add stalls like this, unless the
  next chunk is at most about 1.5× the previous one's length (since RTF ≈ 0.65), or the
  client buffers a little before playing.
- **Phase 6 commit 4b (2026-10-06): smooth joins + a WebSocket benchmark.** Kokoro pads
  every chunk with about 0.3 s of silence in front and 0.5 s behind. That padding is now
  trimmed, and boundary pauses are added instead. On a 4-chunk answer the audio went from
  15.2 → 12.8 s; joins went 770 → 200 ms at a comma and 840 → 350 ms between sentences;
  the 0.3 s of leading silence is gone.

  `bench_latency.py --mode ws` (LLM at 15 words/s, 8 texts, 1 run each, noisy):

  | mode | shaping | hear s (median/max) | gaps | stall s | dead air ≥0.4 s |
  |---|---|---|---|---|---|
  | sentence | off | 2.48 / 5.19 | 3 | 5.48 | 13 |
  | fast_start | off | 2.17 / 5.51 | 4 | 6.33 | 14 |
  | sentence | on | 3.42 / 7.02 | 5 | 9.19 | 2 |
  | fast_start | on | 2.03 / 5.55 | 6 | 10.20 | 0 |

  Dead air is gone, but **stalls got worse**: the padding had been an accidental buffer.
  Short first sentences ("Hi!", "**Sure!**") stall for 2–3 s before a long second one.
  That is exactly what commit 4c's adaptive scheduling targets. Generation fits about
  0.9–1.1 s + 30–43 ms/char over WebSocket (CPU load varies between runs).
- **Phase 6 commit 4a (2026-10-06): sentence splitting fixed.** It used to split
  `3.5`, `$4.99`, `example.com`, `Dr.`, `U.S.`, `e.g.` and `...` (read aloud as
  "three. five"). It also merged sentences before a newline, cut long sentences
  differently streamed vs whole, and with `fast_start` cut inside `https://`. A
  randomized token-split invariant test now guarantees the same chunks either way.
  Added `textnorm` (markdown, links, emoji), on by default for WebSocket.
- **After Phase 6 commit 3: WebSocket `/v1/speech/ws` works end to end with Kokoro**
  (`examples/ws_client.py`, 15 words/s fake LLM). Measured from the first sentence
  being fully sent to its first audio:
  - no `fast_start`: 4.99 s;
  - `fast_start`: 2.29 s (57-char first clause) or 1.44 s (43-char first clause);
  - cached: audio started 0.5 s *before* the first sentence was even complete;
  - barge-in: `cancelled` came back immediately, and the next sentence was never
    spoken.

  **Target ≤ 1.5 s is met only when the first clause is short (≲ 45 chars)**, because
  Kokoro produces a sentence's audio in one piece (about 2.8 s for 57 chars on this CPU).
  Options for later: a lower `clause_min_chars` (more stall risk), OpenAI for chat, or a
  GPU. **Waiting for the user's test of commits 1–3 before commit 4 (Gradio demo) and
  commit 5.**
- **After Phase 6 commit 2** (same bench, median of 3):
  - `fast_start` on: TTFA medium 3.50 → **1.98 s**, long first sentence 5.23 → **2.97 s**,
    short and paragraph unchanged (no comma to split), and no new stalls.
  - Cache hits (`--use-cache`): TTFA about **0.01 s** for every text. Streams are cached
    per sentence now.
  - The paragraph's 0.6 s stall ("Hello!" is too short to cover the next sentence)
    remains; it needs a small client-side buffer or a WebSocket-side strategy (commit 3+).
- **Kokoro preload** (Phase 6, commit 1): the model loads in about 13 s in the background
  after startup; `/health` shows `kokoro: loading → ready`.
- So one machine sustains about **one real-time listener**. More concurrent listeners
  need OpenAI or a GPU. `KOKORO_MAX_CONCURRENCY=1` is right for this CPU.

## Roadmap to the final goal

**Final goal:** one engine that (1) lets a chatbot speak while the LLM is still writing,
and (2) turns a topic or script into a finished multi-voice podcast.

### Phase 6: real-time chatbot (next)

1. Preload Kokoro at startup (background) and report it in `/health`.
2. `scripts/bench_latency.py`: measure time to first audio and RTF, to get a baseline
   before optimizing.
3. **WebSocket `/v1/speech/ws`**: the client sends LLM token deltas, the server chunks
   sentences (`SentenceChunker`) and streams PCM back while text keeps arriving. It
   supports `flush` (end of turn) and `cancel` (barge-in: stop speaking now), and
   synthesizes the next sentence while the current one plays.
4. Faster first audio: allow the first chunk to break at a comma.
5. Per-sentence cache for streaming (HTTP stream and WebSocket).
6. **Demo `examples/voice_chat.py`**: type a question, OpenAI chat streams the answer,
   the WebSocket turns it into speech, and it plays on the speakers. Press Enter to
   interrupt. It prints time to first audio.
7. Smoke-test and docs updates.

**Done when:** first audio within about 1.5 s of the LLM finishing its first sentence,
no stutter for one listener, and barge-in works.

### Phase 7: complete podcast

- ffmpeg for mp3 output.
- Loudness normalization to about −16 LUFS (today it is peak only).
- Topic → LLM-written dialogue → rendered episode.
- Retry endpoint for failed jobs (cached turns make it cheap).

**Done when:** a topic goes in and a normalized mp3 comes out.

### Phase 9: packaging and operations

- Dockerfile with ffmpeg.
- API key and CORS.
- CI (pytest and ruff).
- Detailed `/health` and metrics.

(Phase 8, a Vietnamese provider, was dropped by the language decision.)

### Smaller follow-ups

- Listen to `output/smoke_stream.wav` and confirm there are no clicks between sentences.
- Optionally, verify a podcast render with OpenAI (this uses API quota).
- Grow the name table in `app/core/names.py` when real scripts show misses.
- Single-flight for identical concurrent requests.
- Multi-worker/multi-host only if ever needed: Celery/RQ, object storage, a shared
  cache.
