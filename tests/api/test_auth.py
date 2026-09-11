import pytest
from fastapi import HTTPException

from cardiorag.api.auth import require_api_key
from cardiorag.config import settings


def test_require_api_key_is_a_noop_when_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "api_key", None)

    require_api_key(x_api_key=None)  # must not raise
    require_api_key(x_api_key="anything")  # must not raise


def test_require_api_key_accepts_the_correct_key(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret123")

    require_api_key(x_api_key="secret123")  # must not raise


def test_require_api_key_rejects_a_missing_key(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret123")

    with pytest.raises(HTTPException) as exc_info:
        require_api_key(x_api_key=None)

    assert exc_info.value.status_code == 401


def test_require_api_key_rejects_a_wrong_key(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret123")

    with pytest.raises(HTTPException) as exc_info:
        require_api_key(x_api_key="wrong")

    assert exc_info.value.status_code == 401
