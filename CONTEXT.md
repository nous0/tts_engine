# CONTEXT — current working state

Read this before picking up work in this repo. It says where things actually
stand, not where the original roadmap in `CLAUDE.md` assumed they'd be.
Update it at the end of each work session — this file goes stale fast if
nobody touches it.

Last checked: 2026-09-21.

## Where things actually stand

- **Step 0 cleanup done (2026-09-21):** all extras installed (`uv sync --extra dev
  --extra local --extra ui`), **94/94 tests pass**, `ruff check .` is clean, and all
  the previously uncommitted work (podcast pipeline, jobs, Gemini, Kokoro, tests) is
  committed.
- **Podcast pipeline verified end-to-end with Kokoro**: `examples/script.json` rendered
  to `output/script.wav` (45 s, 24 kHz mono, peak about -1 dBFS). The first run took about
  4 minutes because the model downloaded. After that, each turn takes about 6 s on CPU.
- **Gemini was removed on 2026-09-22** (decision: Kokoro + OpenAI are enough; no
  ElevenLabs). **Your `.env` must say `DEFAULT_PROVIDER=kokoro` / `DEFAULT_VOICE=af_heart`.**
  If it still says gemini, requests that don't name a provider fail.
- **Auto voice assignment picked two female voices** (Alice → `af_heart`, Bob → `af_alloy`),
  because it takes voices in catalog order. Pass `voices` explicitly, or improve
  `assign_voices` to alternate voices.
- **ffmpeg isn't installed** on this machine, so only `wav` and `pcm` output work here.
- **`kokoro-tts/`** (the vendored upstream checkout, which is its own git repo) is now
  gitignored. The app only uses the `kokoro` PyPI package.
- **Streamlit UI now offers Kokoro** (it previously listed only gemini/openai), with
  Kokoro's 28 voices. Verified on 2026-09-22 against a live server: Kokoro one-shot and
  streaming both return 200. The first request after the server starts takes about 40 s
  because the model loads lazily on first use.
- README.md was rewritten to match the current state. `.env.example` now documents Kokoro.

## What's done

- Phase 1–2: provider abstraction, Gemini + OpenAI providers, `/v1/speech`,
  `/v1/speech/stream`, sentence chunker, `/health`. Committed.
- Phase 3: `audio.py` (concat/normalize/encode), `podcast.py` (script
  parsing + rendering), `jobs.py` (SQLite job store + async worker),
  `POST /v1/podcast`, `GET /v1/jobs/{id}`, `GET /v1/jobs`,
  `GET /v1/jobs/{id}/audio`, `examples/generate_podcast.py`. Committed and
  verified end-to-end with Kokoro.
- Phase 4 (partial): Kokoro local provider, fully wired into the registry
  and set as default. Per-speaker provider/voice override already works in
  the podcast request schema.

## What's not done

- `GET /v1/voices` endpoint (`app/api/routes/voices.py` doesn't exist yet;
  `TTSEngine.list_voices()` exists internally but nothing exposes it over
  HTTP).
- ElevenLabs provider — skipped in favor of Kokoro; open decision whether to
  still add it or consider Phase 4 satisfied by Kokoro + Gemini + OpenAI.
- Phase 5 polish: no retries/timeouts around provider calls, no request-level
  caching, no `Dockerfile`, no static route serving `./output/`.
- Optional LLM podcast-script generator (topic → dialogue → audio) — not
  started, still just an idea in CLAUDE.md.

## Immediate to-dos (in rough priority order)

1. **Build `GET /v1/voices`** (`app/api/routes/voices.py`, backed by the existing
   `TTSEngine.list_voices()`), optionally filtered by provider, to close out Phase 4.
2. ~~Decide on ElevenLabs~~: decided. Phase 4 is Kokoro + OpenAI only.
3. Improve `assign_voices` so auto-assigned speakers sound clearly different.
4. Optionally, verify a podcast render with OpenAI as well (this uses API quota).

## Longer-term (Phase 5 and beyond)

- Error handling & retries/timeouts on provider calls (OpenAI network
  failures currently propagate raw).
- Request validation hardening + optional response caching for repeated
  `/v1/speech` calls.
- `Dockerfile` with ffmpeg baked in (needed for mp3/opus/aac/flac output).
- Serve `./output/` via a static route, or move to S3/GCS if this goes
  beyond local/dev use.
- Revisit whether SSE/WebSocket streaming is actually needed, or whether
  chunked HTTP (current approach) is sufficient long-term.
- Optional: LLM script generator so "make a podcast about X" is one call.
