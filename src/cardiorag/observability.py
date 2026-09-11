"""Structured logging (Phase 17): one JSON object per log line, with a
per-request correlation id automatically attached to every log emitted
while that request is being handled - not just the ones written inside
api/main.py itself.

The mechanism: a `ContextVar` holds the current request id; a logging
`Filter` reads it and stamps it onto every `LogRecord` before formatting.
Since Python's logging propagates a module's log calls up to the root
logger's handlers, existing logger.info/warning/exception calls in
ingestion, embeddings, retrieval, and generation automatically pick up
`request_id` and JSON formatting for free once this is configured at the
API's startup - no need to touch every call site individually.
"""

import contextvars
import json
import logging
import sys
import time

_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

# Fields that are part of every LogRecord regardless of what was logged;
# anything else on the record came from a caller's `extra={...}` and should
# be included in the JSON output as its own field.
_STANDARD_LOG_RECORD_FIELDS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


def set_request_id(request_id: str) -> contextvars.Token:
    return _request_id_var.set(request_id)


def reset_request_id(token: contextvars.Token) -> None:
    _request_id_var.reset(token)


def get_request_id() -> str | None:
    return _request_id_var.get()


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
        }
        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_RECORD_FIELDS and key != "request_id"
        }
        payload.update(extra_fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Idempotent: safe to call more than once (e.g. once at API startup,
    once in a test) without accumulating duplicate handlers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(_RequestIdFilter())
    root_logger.addHandler(handler)
