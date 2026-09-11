import io
import json
import logging

from cardiorag.observability import (
    JsonFormatter,
    _RequestIdFilter,
    configure_logging,
    get_request_id,
    reset_request_id,
    set_request_id,
)


def _make_logger(name: str) -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(_RequestIdFilter())

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = [handler]
    logger.propagate = False
    return logger, stream


def test_json_formatter_produces_valid_json_with_expected_fields():
    logger, stream = _make_logger("test.json_fields")

    logger.info("something happened")

    record = json.loads(stream.getvalue())
    assert record["message"] == "something happened"
    assert record["level"] == "INFO"
    assert record["logger"] == "test.json_fields"
    assert "timestamp" in record


def test_json_formatter_includes_extra_fields():
    logger, stream = _make_logger("test.extra_fields")

    logger.info("query completed", extra={"latency_ms": 842, "num_sources": 5})

    record = json.loads(stream.getvalue())
    assert record["latency_ms"] == 842
    assert record["num_sources"] == 5


def test_json_formatter_includes_exception_traceback():
    logger, stream = _make_logger("test.exceptions")

    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("something failed")

    record = json.loads(stream.getvalue())
    assert "ValueError: boom" in record["exception"]


def test_request_id_is_none_by_default():
    logger, stream = _make_logger("test.no_request_id")

    logger.info("no request in flight")

    record = json.loads(stream.getvalue())
    assert record["request_id"] is None


def test_request_id_is_attached_while_set():
    logger, stream = _make_logger("test.with_request_id")

    token = set_request_id("req-123")
    try:
        logger.info("inside a request")
    finally:
        reset_request_id(token)

    record = json.loads(stream.getvalue())
    assert record["request_id"] == "req-123"


def test_request_id_reverts_to_none_after_reset():
    token = set_request_id("req-456")
    reset_request_id(token)

    assert get_request_id() is None


def test_configure_logging_is_idempotent():
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    try:
        configure_logging("INFO")
        configure_logging("INFO")

        assert len(root_logger.handlers) == 1
    finally:
        # configure_logging replaces root handlers globally - restore them so
        # this test doesn't affect pytest's own log capture for other tests.
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)
