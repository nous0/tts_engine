"""Voice catalog route: GET /v1/voices."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.core.engine import ProviderNotConfigured, TTSEngine
from app.models.schemas import Gender, VoiceInfo, VoicesResponse

router = APIRouter(prefix="/v1", tags=["voices"])


def _engine(request: Request) -> TTSEngine:
    return request.app.state.engine


@router.get("/voices")
async def list_voices(
    request: Request, provider: str | None = None, gender: Gender | None = None
) -> VoicesResponse:
    """List voices of one provider, or of every registered provider when omitted."""
    engine = _engine(request)
    names = [provider] if provider else engine.provider_names()

    voices: list[VoiceInfo] = []
    for name in names:
        try:
            catalog = await engine.list_voices(name)
        except ProviderNotConfigured as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        voices.extend(
            VoiceInfo(
                id=v.id, name=v.name, provider=v.provider, language=v.language, gender=v.gender
            )
            for v in catalog
            if gender is None or v.gender == gender
        )

    return VoicesResponse(
        default_provider=engine.default_provider,
        providers=engine.provider_names(),
        voices=voices,
    )
