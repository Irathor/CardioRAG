"""API-key authentication (Phase 20 fix #6).

A single shared secret via the `X-API-Key` header, not per-user accounts -
appropriate for a small research/demo deployment with one or a handful of
trusted clients (the Streamlit UI, evaluation scripts), not a multi-tenant
product needing real user management.

When `settings.api_key` is unset, authentication is disabled entirely: the
pre-fix behavior (fully open access) stays the default for local/dev use
and existing tests, and an operator opts in by setting API_KEY in `.env`.
"""

from fastapi import Header, HTTPException

from cardiorag.config import settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_key is None:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
