"""Request/response models for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

AudioFormat = Literal["mp3", "wav", "opus", "aac", "flac", "pcm"]
JobState = Literal["queued", "running", "done", "error", "cancelled"]
Gender = Literal["female", "male"]


class SpeechRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    voice: str | None = Field(None, description="Provider voice id; falls back to server default.")
    provider: str | None = Field(None, description="Provider name; falls back to server default.")
    format: AudioFormat = "wav"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    instructions: str | None = Field(
        None, description="Optional style/tone hint (supported by some providers)."
    )


class SpeechStreamRequest(BaseModel):
    """Request body for the streaming speech endpoint.

    The full text is sent up-front; the server splits it into sentences and
    streams audio chunks back so the client can start playing before the whole
    clip is synthesized.
    """

    text: str = Field(..., min_length=1, max_length=50000)
    voice: str | None = Field(None, description="Provider voice id; falls back to server default.")
    provider: str | None = Field(None, description="Provider name; falls back to server default.")
    format: AudioFormat = "wav"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    instructions: str | None = Field(
        None, description="Optional style/tone hint (supported by some providers)."
    )
    max_sentence_length: int = Field(
        200, ge=20, le=2000, description="Max chars per synthesized sentence chunk."
    )


class PodcastTurn(BaseModel):
    """One turn of dialogue in a podcast script."""

    speaker: str = Field(..., min_length=1, max_length=60)
    text: str = Field(..., min_length=1, max_length=8000)
    voice: str | None = Field(None, description="Overrides the speaker->voice map.")
    provider: str | None = Field(None, description="Overrides the request provider.")
    instructions: str | None = Field(None, description="Per-turn style/tone hint.")


class PodcastRequest(BaseModel):
    """Submit a multi-speaker script for background rendering.

    Provide either structured ``turns`` or a plain-text ``script`` using
    ``Alice: line`` lines. Speakers without an entry in ``voices`` are cast
    automatically: each gets a distinct voice matching their gender, taken from
    ``genders`` or guessed from the name (Alice -> female, Bob -> male).
    """

    turns: list[PodcastTurn] | None = Field(None, max_length=500)
    script: str | None = Field(
        None, max_length=200_000, description="Plain-text or JSON script."
    )
    voices: dict[str, str] = Field(
        default_factory=dict, description="Speaker name -> provider voice id."
    )
    genders: dict[str, Gender] = Field(
        default_factory=dict,
        description="Speaker name -> 'female'/'male', for names that can't be guessed.",
    )
    provider: str | None = Field(None, description="Provider name; falls back to default.")
    format: AudioFormat = "wav"
    pause_ms: int = Field(600, ge=0, le=5000, description="Silence between turns.")
    normalize: bool = Field(True, description="Peak-normalize the stitched audio.")
    speed: float = Field(1.0, ge=0.25, le=4.0)
    instructions: str | None = Field(None, description="Default style/tone hint.")

    @model_validator(mode="after")
    def _require_script_or_turns(self) -> PodcastRequest:
        if not self.turns and not (self.script and self.script.strip()):
            raise ValueError("Provide either 'turns' or a non-empty 'script'.")
        return self


class PodcastJobResponse(BaseModel):
    """Returned when a podcast render is accepted."""

    job_id: str
    status: JobState
    status_url: str


class JobResponse(BaseModel):
    """Current state of a background job."""

    job_id: str
    kind: str
    status: JobState
    created_at: float
    updated_at: float
    progress: int = Field(0, description="Turns rendered so far.")
    total: int = Field(0, description="Total turns to render.")
    error: str | None = None
    audio_url: str | None = Field(None, description="Set once the render succeeds.")
    voices: dict[str, str] = Field(
        default_factory=dict, description="Resolved speaker -> voice map."
    )


class VoiceInfo(BaseModel):
    """One voice offered by a provider."""

    id: str
    name: str
    provider: str
    language: str | None = None
    gender: Gender | None = Field(None, description="Perceived gender; null if neutral.")


class VoicesResponse(BaseModel):
    """Voice catalog across the registered providers."""

    default_provider: str
    providers: list[str] = Field(description="Registered (usable) providers.")
    voices: list[VoiceInfo]
