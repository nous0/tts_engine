# CONTEXT — current working state

Read this before picking up work in this repo. It says where things actually
stand. Update it at the end of each work session; this file goes stale fast if
nobody touches it.

Last checked: 2026-09-28.

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

## Immediate to-dos (in rough priority order)

1. **Listen** to a streamed WAV (e.g. via Streamlit) and confirm the sentence-boundary
   clicks are gone. Also listen to `output/script_trio.wav`. Tests can't judge audio.
2. **Preload Kokoro at startup** (background task), so the first request doesn't wait
   about 40 s. This was P2 in the Phase 5 review.
3. **Cache streaming per sentence.** Repeated chatbot phrases are the biggest cache win,
   and streaming is all PCM now, so this is straightforward. This was P2 as well.
4. **Detailed `/health`**: providers, ffmpeg, Kokoro loaded, cache size, running jobs. P2.
5. Optionally, verify a podcast render with OpenAI (this uses API quota).
6. Grow the name table in `app/core/names.py` when real scripts show misses.

## Longer-term

- `Dockerfile` with ffmpeg (needed for mp3/opus/aac/flac); expect a 3–4 GB image with
  Kokoro/torch.
- API key auth and CORS if this is ever exposed beyond localhost.
- Multi-worker or multi-host: move jobs to Celery/RQ plus object storage, and the cache
  to shared storage.
- Single-flight for identical concurrent requests (today both synthesize; harmless but
  wasteful).
- Revisit SSE/WebSocket if chunked HTTP isn't enough for the chatbot client.
- Optional: an LLM script generator, so "make a podcast about X" is one call.
