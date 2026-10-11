"""Podcast routes: write a script from a topic, render a script, poll and fetch jobs.

Both steps take a while, so they run as background jobs and return a ``job_id`` at
once:

1. ``POST /v1/podcast/script`` asks the LLM for a script about a topic; it is read
   from ``GET /v1/jobs/{id}/script``, reviewed and edited by the user.
2. ``POST /v1/podcast`` validates a script, then renders it on an in-process worker
   into a file under the configured output directory.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse

from app.config import get_settings
from app.core import audio as audio_ops
from app.core import podcast as podcast_core
from app.core.audio import AudioError
from app.core.engine import InvalidRequest, ProviderNotConfigured, TTSEngine
from app.core.jobs import Job, JobNotActive, JobNotFound, JobStatus, JobStore
from app.core.podcast import PodcastSpec, ScriptError, Turn
from app.core.providers.base import ProviderError, SynthOpts, UnsupportedFormat
from app.core.scriptwriter import Host, ScriptWriter, TopicBrief
from app.models.schemas import (
    JobResponse,
    PodcastJobResponse,
    PodcastRequest,
    ScriptRequest,
    ScriptResponse,
)

router = APIRouter(prefix="/v1", tags=["podcast"])

# Progress is written to the job store at most this often (and always for the last turn).
PROGRESS_INTERVAL_S = 1.0

_MEDIA_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/L16",
}


def _engine(request: Request) -> TTSEngine:
    return request.app.state.engine


def _jobs(request: Request) -> JobStore:
    return request.app.state.jobs


def _output_dir(request: Request) -> Path:
    return Path(request.app.state.output_dir)


def _writer(request: Request) -> ScriptWriter:
    writer = getattr(request.app.state, "writer", None)
    if writer is None:
        raise HTTPException(
            status_code=400,
            detail="No script writer is configured: set OPENAI_API_KEY "
            "(or SCRIPT_WRITER=fake for testing).",
        )
    return writer


def _to_turns(req: PodcastRequest) -> list[Turn]:
    if req.turns:
        turns = podcast_core.parse_script([t.model_dump() for t in req.turns])
    else:
        turns = podcast_core.parse_script(req.script or "")
    return podcast_core.normalize_turns(turns) if req.normalize_text else turns


async def _resolve_voices(
    engine: TTSEngine, turns: list[Turn], req: PodcastRequest
) -> dict[str, str]:
    """Fill in a voice for every speaker, pulling the provider's catalog if needed."""
    if all(t.voice or req.voices.get(t.speaker) for t in turns):
        return dict(req.voices)
    catalog = await engine.list_voices(req.provider)
    return podcast_core.assign_voices(turns, req.voices, catalog, req.genders)


async def _check_spec(engine: TTSEngine, spec: PodcastSpec) -> None:
    """Reject now (400) what would otherwise fail minutes into the render.

    Every voice that will actually be used is checked against its turn's provider, so
    a typo is a 400 at submit time rather than an errored job; the output format and
    loudness mode must be possible on this server.
    """
    try:
        provider = engine.get_provider(spec.provider)
        for turn in spec.turns:
            voice = turn.voice or spec.voices.get(turn.speaker)
            await engine.resolve_voice(engine.get_provider(turn.provider or spec.provider), voice)
    except (ProviderNotConfigured, InvalidRequest) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    can_produce = getattr(provider, "can_produce", None)
    if can_produce is not None and not can_produce(spec.format):
        raise HTTPException(
            status_code=400, detail=str(UnsupportedFormat(spec.format, provider.name))
        )
    if spec.format not in audio_ops.NATIVE_FORMATS and not audio_ops.HAS_FFMPEG:
        raise HTTPException(
            status_code=400,
            detail=f"Format '{spec.format}' requires ffmpeg on the server; use wav or pcm.",
        )
    if spec.loudness == "lufs" and not audio_ops.HAS_FFMPEG:
        raise HTTPException(
            status_code=400,
            detail="Loudness 'lufs' requires ffmpeg on the server; use 'peak' or 'off'.",
        )


def _job_response(job: Job) -> JobResponse:
    done = job.status is JobStatus.DONE
    return JobResponse(
        job_id=job.id,
        kind=job.kind,
        status=job.status.value,
        created_at=job.created_at,
        updated_at=job.updated_at,
        progress=job.progress,
        total=job.total,
        error=job.error,
        audio_url=(
            f"/v1/jobs/{job.id}/audio" if done and job.kind == "podcast" else None
        ),
        script_url=f"/v1/jobs/{job.id}/script" if done and job.kind == "script" else None,
        voices=job.meta.get("voices", {}),
        title=job.meta.get("title"),
        duration_s=job.meta.get("duration_s"),
        loudness=job.meta.get("loudness"),
        retry_of=job.meta.get("retry_of"),
    )


@router.post("/podcast", status_code=202)
async def create_podcast(req: PodcastRequest, request: Request) -> PodcastJobResponse:
    engine = _engine(request)

    # Validate everything that can fail fast, so the client gets a 400 now
    # rather than an errored job later.
    try:
        turns = _to_turns(req)
        voices = await _resolve_voices(engine, turns, req)
    except ScriptError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ProviderNotConfigured, InvalidRequest) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    settings = get_settings()
    spec = PodcastSpec(
        turns=turns,
        voices=voices,
        provider=req.provider,
        format=req.format,
        pause_ms=req.pause_ms,
        loudness=podcast_core.pick_loudness(req.loudness, req.normalize, audio_ops.HAS_FFMPEG),
        target_lufs=(
            req.target_lufs if req.target_lufs is not None else settings.podcast_target_lufs
        ),
        speed=req.speed,
        instructions=req.instructions,
        title=req.title,
        sample_rate=settings.podcast_mp3_sample_rate if req.format == "mp3" else None,
        bitrate=settings.podcast_mp3_bitrate if req.format == "mp3" else None,
    )
    await _check_spec(engine, spec)
    job = await _start_podcast(request, spec)
    return PodcastJobResponse(
        job_id=job.id, status=JobStatus.QUEUED.value, status_url=f"/v1/jobs/{job.id}"
    )


async def _start_podcast(request: Request, spec: PodcastSpec, retry_of: str | None = None) -> Job:
    """Record a podcast job (with its spec, for retries) and queue the render."""
    jobs = _jobs(request)
    meta = {
        "voices": spec.voices,
        "format": spec.format,
        "provider": spec.provider,
        "title": spec.title,
        "loudness_mode": spec.loudness,
    }
    if retry_of:
        meta["retry_of"] = retry_of
    job = await jobs.create(
        kind="podcast", total=len(spec.turns), meta=meta, spec=spec.to_dict()
    )
    out_path = _output_dir(request) / f"podcast_{job.id}.{spec.format}"
    jobs.spawn(job.id, _renderer(jobs, _engine(request), job.id, spec, out_path, meta))
    return job


@router.post("/podcast/script", status_code=202)
async def create_script(req: ScriptRequest, request: Request) -> PodcastJobResponse:
    """Step 1: have the LLM write a script about a topic, as a background job."""
    brief = TopicBrief(
        topic=req.topic.strip(),
        duration_min=req.duration_min,
        hosts=[Host(h.name.strip(), h.gender) for h in req.hosts],
        style=req.style,
    )
    job = await _start_script(request, brief)
    return PodcastJobResponse(
        job_id=job.id, status=JobStatus.QUEUED.value, status_url=f"/v1/jobs/{job.id}"
    )


async def _start_script(request: Request, brief: TopicBrief, retry_of: str | None = None) -> Job:
    writer = _writer(request)
    jobs = _jobs(request)
    meta: dict = {"topic": brief.topic[:200], "duration_min": brief.duration_min}
    if retry_of:
        meta["retry_of"] = retry_of
    job = await jobs.create(kind="script", total=1, meta=meta, spec=brief.to_dict())
    out_path = _output_dir(request) / f"script_{job.id}.json"

    async def work() -> None:
        try:
            script = await writer(brief)
        except ProviderError as exc:
            await jobs.update(job.id, status=JobStatus.ERROR, error=str(exc))
            return
        body = script.to_request()
        data = json.dumps(body, ensure_ascii=False, indent=2).encode("utf-8")
        await asyncio.to_thread(_write_atomic, out_path, data)
        await jobs.update(
            job.id,
            status=JobStatus.DONE,
            progress=1,
            result_path=str(out_path),
            meta={
                **meta,
                "title": script.title,
                "turns": len(script.turns),
                "estimated_minutes": script.estimated_minutes,
            },
        )

    jobs.spawn(job.id, work, kind="script")
    return job


@router.get("/jobs/{job_id}/script")
async def get_job_script(job_id: str, request: Request) -> ScriptResponse:
    """A written script, ready to edit and send to ``POST /v1/podcast``."""
    job = await _jobs(request).get(job_id)
    if job is None or job.kind != "script":
        raise HTTPException(status_code=404, detail=f"Script job '{job_id}' not found.")
    if job.status is not JobStatus.DONE or not job.result_path:
        raise HTTPException(
            status_code=409, detail=f"Job '{job_id}' is {job.status.value}, not done."
        )
    try:
        body = json.loads(await asyncio.to_thread(Path(job.result_path).read_bytes))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=410, detail="The script is no longer available.") from exc
    return ScriptResponse(**body)


def _renderer(
    jobs: JobStore,
    engine: TTSEngine,
    job_id: str,
    spec: PodcastSpec,
    out_path: Path,
    meta: dict,
):
    """Build the coroutine factory that renders ``spec`` and records the result."""

    async def synthesize(
        text: str, voice: str, provider: str | None, opts: SynthOpts
    ) -> bytes:
        return await engine.synthesize(text, voice=voice, provider=provider, opts=opts)

    async def work() -> None:
        last_write = 0.0

        async def on_progress(done: int, total: int) -> None:
            # Clients poll the job, so progress has to reach the store while rendering.
            nonlocal last_write
            now = time.monotonic()
            if done < total and now - last_write < PROGRESS_INTERVAL_S:
                return
            last_write = now
            await jobs.update(job_id, progress=done)

        try:
            result = await podcast_core.render(spec, synthesize, on_progress)
        except (
            AudioError,
            UnsupportedFormat,
            ScriptError,
            ProviderNotConfigured,
            ProviderError,
            InvalidRequest,
        ) as exc:
            await jobs.update(job_id, status=JobStatus.ERROR, error=str(exc))
            return

        await asyncio.to_thread(_write_atomic, out_path, result.audio)
        await jobs.update(
            job_id,
            status=JobStatus.DONE,
            progress=len(spec.turns),
            result_path=str(out_path),
            meta={
                **meta,
                "duration_s": round(result.duration_s, 2),
                "loudness": result.loudness,
            },
        )

    return work


def _write_atomic(path: Path, data: bytes) -> None:
    """Write via a temp file and rename, so a crash never leaves a truncated result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> JobResponse:
    job = await _jobs(request).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _job_response(job)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request) -> JobResponse:
    """Cancel a queued or running job. A running render stops once the turn it is
    synthesizing finishes (a provider call can't be interrupted midway)."""
    try:
        job = await _jobs(request).cancel(job_id)
    except JobNotFound as exc:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.") from exc
    except JobNotActive as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _job_response(job)


@router.post("/jobs/{job_id}/retry", status_code=202)
async def retry_job(job_id: str, request: Request) -> PodcastJobResponse:
    """Start a new job from a failed or cancelled one's input.

    The new job has its own id and ``retry_of`` pointing back; the old record is left
    as it was. Turns rendered before the failure come from the audio cache, so only
    the rest is synthesized again.
    """
    jobs = _jobs(request)
    job = await jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status not in (JobStatus.ERROR, JobStatus.CANCELLED):
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is {job.status.value}; only failed or cancelled jobs "
            "can be retried.",
        )
    data = await jobs.get_spec(job_id)
    if data is None:
        raise HTTPException(
            status_code=410,
            detail=f"Job '{job_id}' was created before inputs were stored; submit it again.",
        )
    if job.kind == "script":
        new_job = await _start_script(request, TopicBrief.from_dict(data), retry_of=job_id)
        return PodcastJobResponse(
            job_id=new_job.id, status=JobStatus.QUEUED.value, status_url=f"/v1/jobs/{new_job.id}"
        )
    if job.kind != "podcast":
        raise HTTPException(status_code=400, detail=f"Jobs of kind '{job.kind}' can't be retried.")

    spec = PodcastSpec.from_dict(data)
    await _check_spec(_engine(request), spec)  # voices or ffmpeg may have changed since
    new_job = await _start_podcast(request, spec, retry_of=job_id)
    return PodcastJobResponse(
        job_id=new_job.id, status=JobStatus.QUEUED.value, status_url=f"/v1/jobs/{new_job.id}"
    )


@router.get("/jobs", response_model=list[JobResponse])
async def list_jobs(request: Request, limit: int = 50) -> list[JobResponse]:
    jobs = await _jobs(request).list(limit=max(1, min(limit, 200)))
    return [_job_response(j) for j in jobs]


@router.get(
    "/jobs/{job_id}/audio",
    responses={200: {"content": {"audio/wav": {}}, "description": "Rendered audio."}},
)
async def get_job_audio(job_id: str, request: Request) -> Response:
    job = await _jobs(request).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.kind != "podcast":
        raise HTTPException(
            status_code=404, detail=f"Job '{job_id}' is a {job.kind} job and has no audio."
        )
    if job.status is not JobStatus.DONE or not job.result_path:
        raise HTTPException(
            status_code=409, detail=f"Job '{job_id}' is {job.status.value}, not done."
        )
    path = Path(job.result_path)
    if not path.exists():
        raise HTTPException(status_code=410, detail="Rendered audio is no longer available.")

    fmt = job.meta.get("format", "wav")
    return FileResponse(
        path,
        media_type=_MEDIA_TYPES.get(fmt, "application/octet-stream"),
        filename=path.name,
    )
