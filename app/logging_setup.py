"""Logging for the app: one handler, request ids on every line.

Every request gets an id (the client's ``X-Request-ID`` if sent, otherwise a new one)
that is stored in a context variable, so log lines from the route, the engine, the
providers and the job worker can be tied back to the request that caused them.

Only the ``app`` and ``openai`` loggers get a handler; they still propagate, so test
tools that listen on the root logger (pytest's ``caplog``) keep working. Uvicorn
configures its own loggers and is left alone. Never log synthesized text: log its
length instead.
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_FORMAT = "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"
_LOGGERS = ("app", "openai")
_HANDLER_NAME = "tts-engine"


class RequestIdFilter(logging.Filter):
    """Stamp each record with the id of the request being served (or ``-``)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    """Attach the app's handler once; calling again only updates the level."""
    lvl = logging.getLevelName(level.upper())
    if not isinstance(lvl, int):
        lvl = logging.INFO
    for name in _LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(lvl)
        if any(h.get_name() == _HANDLER_NAME for h in logger.handlers):
            continue
        handler = logging.StreamHandler()
        handler.set_name(_HANDLER_NAME)
        handler.addFilter(RequestIdFilter())
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)


def new_request_id(incoming: str | None = None) -> str:
    """Use the client's id when it looks sane, otherwise make a short random one."""
    if incoming and len(incoming) <= 128 and incoming.isprintable():
        return incoming
    return uuid.uuid4().hex[:12]
