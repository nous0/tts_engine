"""How engine/provider errors map to HTTP statuses, for both HTTP and WebSocket."""

from __future__ import annotations

from app.core.engine import InvalidRequest, ProviderNotConfigured
from app.core.providers.base import ProviderError, UnsupportedFormat

# How each ProviderError kind reaches the client.
PROVIDER_ERROR_STATUS = {
    "bad_request": 400,
    "auth": 502,
    "upstream": 502,
    "rate_limited": 503,
    "timeout": 504,
}

CLIENT_ERRORS = (ProviderNotConfigured, UnsupportedFormat, InvalidRequest)


def error_payload(exc: Exception) -> dict:
    """``{"detail", "status", "kind"?}`` for an error; unexpected ones become a 500."""
    if isinstance(exc, ProviderError):
        return {
            "detail": str(exc),
            "status": PROVIDER_ERROR_STATUS.get(exc.kind, 502),
            "kind": exc.kind,
        }
    if isinstance(exc, CLIENT_ERRORS):
        return {"detail": str(exc), "status": 400}
    return {"detail": "Internal error.", "status": 500}
