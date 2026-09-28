"""Podcast routes: submit a multi-speaker script, poll the job, fetch the audio.

Rendering a long script takes minutes, so ``POST /v1/podcast`` validates the
request, registers a job and returns immediately with a ``job_id``. The render
runs on an in-process worker and writes the finished file under the configured
output directory.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse

from app.core import podcast as podcast_core
from app.core.audio import AudioError
from app.core.engine import InvalidRequest, ProviderNotConfigured, TTSEngine
from app.core.jobs import Job, JobNotActive, JobNotFound, JobStatus, JobStore
from app.core.podcast import PodcastSpec, ScriptError, Turn
from app.core.providers.base import ProviderError, SynthOpts, UnsupportedFormat
from app.models.schemas import (
    JobResponse,
    PodcastJobResponse,
    PodcastRequest,
)

router = APIRouter(prefix="/v1", tags=["podcast"])

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


def _to_turns(req: PodcastRequest) -> list[Turn]:
    if req.turns:
        return podcast_core.parse_script([t.model_dump() for t in req.turns])
    return podcast_core.parse_script(req.script or "")


async def _resolve_voices(
    engine: TTSEngine, turns: list[Turn], req: PodcastRequest
) -> dict[str, str]:
    """Fill in a voice for every speaker, pulling the provider's catalog if needed."""
    if all(t.voice or req.voices.get(t.speaker) for t in turns):
        voices = dict(req.voices)
    else:
        catalog = await engine.list_voices(req.provider)
        voices = podcast_core.assign_voices(turns, req.voices, catalog, req.genders)
    # Check every voice that will actually be used against its turn's provider now,
    # so a typo is a 400 at submit time rather than an errored job minutes later.
    for turn in turns:
        voice = turn.voice or voices.get(turn.speaker)
        provider = engine.get_provider(turn.provider or req.provider)
        await engine.resolve_voice(provider, voice)
    return voices


def _job_response(job: Job) -> JobResponse:
    return JobResponse(
        job_id=job.id,
        kind=job.kind,
        status=job.status.value,
        created_at=job.created_at,
        updated_at=job.updated_at,
        progress=job.progress,
        total=job.total,
        error=job.error,
        audio_url=f"/v1/jobs/{job.id}/audio" if job.status is JobStatus.DONE else None,
        voices=job.meta.get("voices", {}),
    )


@router.post("/podcast", status_code=202)
async def create_podcast(req: PodcastRequest, request: Request) -> PodcastJobResponse:
    engine = _engine(request)

    # Validate everything that can fail fast, so the client gets a 400 now
    # rather than an errored job later.
    try:
        turns = _to_turns(req)
        provider = engine.get_provider(req.provider)
        voices = await _resolve_voices(engine, turns, req)
    except ScriptError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ProviderNotConfigured, InvalidRequest) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    can_produce = getattr(provider, "can_produce", None)
    if can_produce is not None and not can_produce(req.format):
        raise HTTPException(
            status_code=400, detail=str(UnsupportedFormat(req.format, provider.name))
        )

    spec = PodcastSpec(
        turns=turns,
        voices=voices,
        provider=req.provider,
        format=req.format,
        pause_ms=req.pause_ms,
        normalize=req.normalize,
        speed=req.speed,
        instructions=req.instructions,
    )

    jobs = _jobs(request)
    job = await jobs.create(
        kind="podcast",
        total=len(turns),
        meta={"voices": voices, "format": req.format, "provider": req.provider},
    )
    out_path = _output_dir(request) / f"podcast_{job.id}.{req.format}"
    jobs.spawn(job.id, _renderer(jobs, engine, job.id, spec, out_path))

    return PodcastJobResponse(
        job_id=job.id, status=JobStatus.QUEUED.value, status_url=f"/v1/jobs/{job.id}"
    )


def _renderer(
    jobs: JobStore, engine: TTSEngine, job_id: str, spec: PodcastSpec, out_path: Path
):
    """Build the coroutine factory that renders ``spec`` and records the result."""

    async def synthesize(
        text: str, voice: str, provider: str | None, opts: SynthOpts
    ) -> bytes:
        return await engine.synthesize(text, voice=voice, provider=provider, opts=opts)

    async def work() -> None:
        pending: list[int] = []

        def on_progress(done: int, _total: int) -> None:
            pending.append(done)

        try:
            data = await podcast_core.render(spec, synthesize, on_progress)
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

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        await jobs.update(
            job_id,
            status=JobStatus.DONE,
            progress=pending[-1] if pending else len(spec.turns),
            result_path=str(out_path),
        )

    return work


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> JobResponse:
    job = await _jobs(request).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _job_response(job)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request) -> JobResponse:
    """Cancel a queued or running job. A running render stops after the sentence
    it is currently synthesizing."""
    try:
        job = await _jobs(request).cancel(job_id)
    except JobNotFound as exc:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.") from exc
    except JobNotActive as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _job_response(job)


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
