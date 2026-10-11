# CONTEXT — current working state

Read this before picking up work in this repo. It says where things actually
stand. Update it at the end of each work session; this file goes stale fast if
nobody touches it.

Last checked: 2026-10-11.

## Where things actually stand

- **Phase 7 (complete podcast) finished on 2026-10-11**, in 5 commits:
  - live job progress;
  - mp3 via `libmp3lame` and −19 LUFS loudness;
  - stored job specs and `POST /v1/jobs/{id}/retry`;
  - LLM script writing from a topic (two steps, with review);
  - CLI and smoke test.

  **467 tests pass**, `ruff check .` is clean, and the smoke test passes 50/50. All of it
  was checked live with Kokoro, ffmpeg and `gpt-4o-mini`; details are under Phase 7 in the
  roadmap.
- **Next:** Phase 9 (packaging: Dockerfile with ffmpeg, API key and CORS, CI) or the
  per-sentence podcast rendering follow-up (it stops a Kokoro podcast from delaying the
  chatbot).

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
- **ffmpeg is installed** (2026-10-11, `C:\ffmpeg-master-latest-win64-gpl\bin`), so mp3, opus,
  aac and flac work. Restart a running server after installing it, because `HAS_FFMPEG` is
  read at import.
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
- **Phase 6 commit 5 (2026-10-07):** `scripts/smoke_test.py` adds a WebSocket group:
  ready, audio before flush, chunk text and boundary, done, cancel and resume,
  decimals/abbreviations kept whole, cache hits, bad start → 1008, foreign Origin
  refused. **32/32 pass.**

  **Phase 6 is functionally done. Still open:**
  1. The user switches Windows' power mode to **Best performance**.
  2. Rerun the final benchmark (`bench_latency.py --mode ws --chunking all --wps 5,15,40`)
     and record it here.
  3. ~~A valid `OPENAI_API_KEY`~~ (done 2026-10-11); the user tries the demo with a real LLM.
  4. The user's listening check: joins, pauses, prosody at cut points.

- **Phase 6 commit 4d (2026-10-07): Gradio voice chat demo** (`examples/voice_chat_ui.py`,
  extra `demo`, ffmpeg needed).
  - **Spike** (headless Chromium via `uv run --with playwright`): Gradio's streaming
    player added 0.76–1.0 s from yield to playback when warm, and 2.6 s on the first
    stream, but **no gaps** (one AAC encoder per stream). The earlier "end-of-stream
    stall" came only from the spike sleeping after its last yield. It misses the
    ≤ 0.4 s target, and **the user chose to keep Gradio** over a custom Web Audio page.
  - **Driven end to end in headless Chromium:** the text streams, speech plays, and
    **Stop** stops playback within 0.6 s, with the TTS log showing "speech cancelled".
    Enter → playback was ~2.0 s on the first question and ~1.3 s warm (TTS first audio
    0.45 s; answers repeated, so cache hits).
  - ~~The `OPENAI_API_KEY` in `.env` was rejected (401).~~ **Fixed 2026-10-11:** the new
    key works for chat and TTS, so the demo can now run with a real LLM.
- **Phase 6 commit 4c (2026-10-07): adaptive chunking** (WS default
  `chunking: "adaptive"`).
  - **How it works:** the speaker pulls each chunk when it is ready (`take()`), sized
    by a learned latency model `a + b·chars` against a playback clock.
  - **Rules:** the first chunk of a turn is sized for latency; later chunks stay above
    the break-even length `a / (d − b)` (capped at 120); timing is shared per provider
    across sessions.
  - **Torch threads:** `KOKORO_TORCH_THREADS=4` is ~20% faster than torch's default 12.
    The CPU has 4 P-cores.
  - **Interleaved benchmark, 15 words/s, CPU at its normal speed (~27 ms/char):**

    | mode | hear median / max | stall total | dead air | chunks (avg chars) | word cuts |
    |---|---|---|---|---|---|
    | sentence | 2.47 / 7.93 s | 5.82 s | 2 | 19 (53) | 0 |
    | fast_start | 1.08 / 3.62 s | 4.78 s | 0 | 22 (42) | 0 |
    | adaptive | 1.17 / 1.95 s | 0.13 s | 0 | 32 (29) | 16 |

  - **The CPU's speed is the dominant factor.** Across runs Kokoro measured from ~27 up
    to ~62 ms/char. The cause is Windows' power mode: **"Best power efficiency"**
    (overlay `961cc777...`) is active even on AC, and the clock sits at 1711/2200 MHz.
    At ~60 ms/char synthesis is no faster than speech, so every mode stalls (sentence
    15 s, adaptive 48 s total, the latter as one long pause after a fast first chunk).
    **To do (user): switch to Best performance, then rerun the final benchmark in
    commit 5.**
  - **Bugs found by tests while building it:** a narrow cut window that spoke "Hi!"
    alone; splitting a word (URL) at the budget; a small-chunk death spiral on a slow
    CPU; per-session learning that never carried over.
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

### Phase 7: complete podcast (done 2026-10-11; plan revised the same day)

**Decisions (2026-10-11):**
- Loudness target **−19 LUFS**, true peak −1.5 dBTP, configurable. The output is mono, and −16
  is the stereo norm. Measured on Kokoro speech: −19 fits in loudnorm's `linear` mode, while
  −16 would exceed the true-peak limit and fall back to dynamic compression.
- Topic flow in **two steps with review**: write a script, let the user read and edit it, then
  render.
- Per-sentence podcast rendering and Kokoro edge trimming are **deferred** (see follow-ups).

**Environment checked (2026-10-11):**
- ffmpeg `N-127222` has `libmp3lame`, `loudnorm` and `ebur128`.
- The OpenAI key works for chat (`gpt-4o-mini`, strict `json_schema`) and TTS.
- Today `ffmpeg_encode` writes mp3 at 24 kHz / 32 kbps, which is too low.

**Commits:**
1. Job/podcast fixes:
   - progress is written while rendering (it was only written at the end);
   - atomic, threaded result write;
   - `normalize_text` option;
   - drop the dead `PODCAST_PAUSE_MS`.
2. **Done (2026-10-11).** mp3 via `libmp3lame` (64k; 44.1 kHz for podcasts) with an ID3
   title, plus LUFS loudness. Request fields: `loudness: lufs|peak|off`, `target_lufs`,
   `title`; stats go in the job meta.
   - **Changed from the plan:** two-pass `loudnorm` with `linear=true` fell back to dynamic
     mode on `script_trio.txt`. The raw episode measured −25.8 LUFS with a true peak of
     −5.2 dBTP, so +6.8 dB would have clipped. Only 62 samples (0.009%) were above the
     limit.
   - **Now:** one gain, plus a 4x-oversampled `alimiter` only when the peaks need it.
   - **Compared with loudnorm dynamic on the same episode:**

     | method | LUFS | true peak (dBTP) | LRA | median gain wobble (dB) |
     |---|---|---|---|---|
     | gain + limiter | −19.0 | −2.0 | 3.8 | 0.05 |
     | loudnorm dynamic | −19.3 | −1.5 | 3.5 | 0.24 |

   - **Final `trio.mp3`:** 44.1 kHz mono 64 kbps, titled; `ebur128` measures −19.3 LUFS,
     −2.2 dBTP.
3. **Done (2026-10-11).** Each job stores its spec in a new `spec` column (lightweight
   migration).
   - `POST /v1/jobs/{id}/retry` creates a new job (`retry_of`) from an `error` or
     `cancelled` job (409 / 404 / 410 otherwise).
   - Per-kind job limits, so script jobs don't wait behind renders.
   - **Checked live** with Kokoro and mp3: the server was hard-killed at 3/8 turns, and after
     the restart the job was `error` ("Interrupted by a server restart"). The retry finished
     with **3 cache hits and 5 syntheses**, at −19.05 LUFS and −1.97 dBTP.
4. **Done (2026-10-11).** `app/core/scriptwriter.py` (OpenAI structured output, plus a
   fake writer).
   - `POST /v1/podcast/script` creates a `script` job.
   - `GET /v1/jobs/{id}/script` returns a body that can be posted to `/v1/podcast` as is.
   - Turns are capped at 1500 chars, split at whole sentences.
   - **Checked live** (`gpt-4o-mini`, Kokoro, mp3, 2 minutes):
     - writing the script took 7.6 s (12 turns, 1408 chars, Emily/Jake with genders);
     - after one line was edited, the render took 67 s for 107 s of audio, with progress
       0→12;
     - the mp3 measured −19.5 LUFS, −2.0 dBTP.
   - The 800 chars/min estimate held (1.8 min estimated, 1.8 min actual).
5. **Done (2026-10-11).** CLI `--topic/--minutes/--host/--script-only/--writer`; a script
   file keeps its genders and title when rendered.
   - `smoke_test.py` adds Retry, mp3 + loudness, and Topic → script → podcast groups. The
     topic group needs `SCRIPT_WRITER=fake` and still never calls OpenAI.
   - `--check-interrupted` now also retries the interrupted job.
   - **Smoke result: 50/50 pass**, and the restart + retry check passes. 467 unit tests pass.

**Still open for the user:**
- Listen to a full episode, e.g. `examples/script_trio.txt --format mp3`: check the level,
  the gaps between turns, and the limiter on plosives.
- Optionally render a topic script with the OpenAI TTS provider (uses quota).

**Done when:** topic → LLM script → review/edit → mp3 at 44.1 kHz and −19 ±1 LUFS (checked with
`ebur128`). Progress rises during the render. A failed or restart-interrupted job retries
without re-synthesizing turns that are already cached.

### Phase 9: packaging and operations

- Dockerfile with ffmpeg.
- API key and CORS.
- CI (pytest and ruff).
- Detailed `/health` and metrics.

(Phase 8, a Vietnamese provider, was dropped by the language decision.)

### Smaller follow-ups

- **Podcast rendering per sentence (deferred from Phase 7).** Today a turn is one provider call,
  so:
  - a Kokoro podcast holds the single Kokoro worker for a whole turn and blocks chatbot
    sentences;
  - cancel only lands after that turn;
  - turns over 4096 chars fail on OpenAI;
  - Kokoro's ~0.3 s + 0.5 s edge padding adds to `pause_ms`, giving gaps of about 1.4 s.

  The fix is to split turns with `split_chunks` and trim with `EdgeTrimmer`, as streaming does.

- Listen to `output/smoke_stream.wav` and confirm there are no clicks between sentences.
- Optionally, verify a podcast render with OpenAI (this uses API quota).
- Grow the name table in `app/core/names.py` when real scripts show misses.
- Single-flight for identical concurrent requests.
- Multi-worker/multi-host only if ever needed: Celery/RQ, object storage, a shared
  cache.
