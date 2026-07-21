# CLAUDE.md — TTS Engine

Guidance for Claude Code (and contributors) working in this repo.

## Project

A **TTS engine as a Python/FastAPI backend service** that wraps **cloud TTS providers** (OpenAI, ElevenLabs, later Azure/Google) behind one API. It serves two consumers:

1. **Chatbot integration** — real-time, **low-latency streaming** speech. Text may arrive incrementally (LLM tokens); synthesize sentence-by-sentence and stream audio chunks back so the bot can start speaking before the full reply is ready. Typically a single voice.
2. **Podcast generation** — **long-form, multi-speaker** dialogue. Takes a script (or a topic), assigns a voice per speaker, synthesizes each turn, stitches segments with pauses/normalization, and returns a downloadable audio file. Quality over speed; runs as a background job.

Both modes share one **provider abstraction** and one **audio pipeline** — a single pluggable engine, not two scripts. Cloud APIs chosen for quality, built-in streaming, and fast time-to-ship; the abstraction keeps a local backend (Piper/XTTS) as a future drop-in.

## Architecture

**Layers:**
- **API (FastAPI)** — HTTP endpoints + streaming (SSE/WebSocket) + OpenAPI docs.
- **Engine (orchestrator)** — routes requests to a provider, applies chunking, drives audio assembly and jobs.
- **Providers** — one class per cloud service behind a common `TTSProvider` interface; selected via config/registry.
- **Audio pipeline** — concatenation, silence padding, loudness normalization, format encoding (mp3/wav/opus) via `pydub` + `ffmpeg`.
- **Jobs** — async store for long-running podcast renders (SQLite-backed, in-proc worker for MVP).

**Provider interface** (`app/core/providers/base.py`):
```python
class TTSProvider(Protocol):
    async def synthesize(self, text: str, voice: str, opts: SynthOpts) -> bytes: ...
    async def synthesize_stream(self, text: str, voice: str, opts: SynthOpts) -> AsyncIterator[bytes]: ...
    async def list_voices(self) -> list[Voice]: ...
```
A `registry.py` maps provider name → instance, configured from env. Start with **OpenAI** (simplest streaming) and **ElevenLabs** (best quality + voice variety + native dialogue). Azure/Google are later additions requiring no interface change.

**Streaming for chatbot** — a `SentenceChunker` buffers incoming text and flushes on sentence boundaries (`.` `?` `!`, newline, or max-length), so each complete sentence is synthesized and streamed while later text is still arriving. Endpoint yields audio chunks via chunked HTTP / SSE (or WebSocket for bidirectional text-in/audio-out).

**Podcast assembly** (`app/core/podcast.py`) — input is a structured script (JSON list of `{speaker, text}` turns) plus a speaker→voice map; optional plain-text parser for `Alice: ...` line format. For each turn: synthesize → decode → append with a configurable inter-turn pause → normalize loudness → export single file. **Optional stretch:** an LLM script generator that turns a topic/article into a two-host dialogue, then feeds it into this pipeline so "make a podcast" is one call.

## Project structure

```
tts/
  pyproject.toml            # deps + tooling (ruff, pytest)
  .env.example              # API keys, default provider/voices
  README.md                 # setup, run, API usage, examples
  Dockerfile                # python + ffmpeg
  app/
    main.py                 # FastAPI app, router wiring, lifespan
    config.py               # pydantic-settings (keys, defaults)
    models/schemas.py       # request/response pydantic models
    api/routes/
      speech.py             # POST /v1/speech, POST /v1/speech/stream
      podcast.py            # POST /v1/podcast, GET /v1/jobs/{id}
      voices.py             # GET /v1/voices
    core/
      engine.py             # orchestrator
      chunker.py            # SentenceChunker for streaming
      audio.py              # concat / pad / normalize / encode (pydub+ffmpeg)
      podcast.py            # script parse + multi-speaker assembly
      jobs.py               # SQLite job store + background worker
      providers/
        base.py             # TTSProvider protocol + SynthOpts/Voice
        openai.py
        elevenlabs.py
        registry.py
  examples/
    chatbot_stream.py       # LLM tokens -> engine stream -> play audio
    generate_podcast.py     # script.json -> mp3
  tests/
    test_chunker.py  test_audio.py  test_providers.py  test_api.py
```

## API surface

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/speech` | One-shot synth; returns audio file (mp3/wav/opus). Chatbot non-streaming + general use. |
| POST | `/v1/speech/stream` | Low-latency streaming synth (chunked/SSE); optional WebSocket for token-in/audio-out. Chatbot voice. |
| POST | `/v1/podcast` | Submit multi-speaker script (or topic); returns a `job_id`. |
| GET | `/v1/jobs/{id}` | Poll podcast job status; returns audio URL when done. |
| GET | `/v1/voices` | List voices, filterable by provider. |
| GET | `/health` | Liveness. |

## Key dependencies

`fastapi`, `uvicorn[standard]`, `pydantic`, `pydantic-settings`, `httpx` (async provider calls), `sse-starlette` (streaming), `pydub` + system **ffmpeg** (audio), `openai` + `elevenlabs` SDKs, `aiosqlite` (jobs); dev: `pytest`, `pytest-asyncio`, `ruff`. Python 3.11+.

## Build phases (incremental, each independently runnable)

1. **Scaffold + first provider** — `pyproject.toml`, `config.py`, provider `base.py` + `openai.py` + `registry.py`, `POST /v1/speech`, `/health`. Verify: synthesize "hello world" to mp3.
2. **Chatbot streaming** — `chunker.py`, `POST /v1/speech/stream`, `examples/chatbot_stream.py`. Verify: stream a paragraph and confirm audio chunks arrive before input completes.
3. **Podcast pipeline** — `audio.py`, `podcast.py`, `jobs.py`, `POST /v1/podcast` + `GET /v1/jobs/{id}`, `examples/generate_podcast.py`. Verify: 2-speaker script → single stitched mp3 with pauses.
4. **Second provider + voices** — `elevenlabs.py`, `GET /v1/voices`, per-speaker provider/voice selection. Verify: same podcast rendered with ElevenLabs voices.
5. **Polish** — error handling & retries/timeouts on provider calls, request validation, optional response caching, loudness normalization pass, `Dockerfile` (with ffmpeg), `README.md`, tests. *Optional:* LLM podcast-script generator (`topic → dialogue → audio`).

## Design defaults (sensible choices; revisit as needed)

- **Streaming transport:** chunked HTTP + SSE for MVP (simplest for chatbot HTTP clients); add WebSocket only if bidirectional text-in is needed.
- **Podcast jobs:** in-process background worker + SQLite for MVP; Celery/RQ + object storage is the scale-out path.
- **Audio storage:** local `./output/` dir served via static route for MVP; S3/GCS for production.
- **Default formats:** mp3 for downloads, opus/mp3 chunks for streaming, wav internally before encode.

## Verification

- **Unit:** `pytest` — `test_chunker` (sentence boundaries, max-length flush), `test_audio` (concat length = sum + pauses, format round-trip), `test_providers` (mocked httpx), `test_api` (FastAPI `TestClient` on each route).
- **End-to-end (real keys in `.env`):**
  - Speech: `curl -X POST /v1/speech -d '{"text":"Hello","voice":"..."}' --output out.mp3` and play it.
  - Streaming: run `examples/chatbot_stream.py`, confirm first audio plays before the full text is sent.
  - Podcast: run `examples/generate_podcast.py examples/script.json`, confirm a single mp3 with distinct voices per speaker and pauses between turns.
- **Docs:** open `/docs` (Swagger) and exercise each endpoint interactively.
