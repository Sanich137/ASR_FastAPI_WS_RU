import json
import logging
import pytest
from core.logging_config import JsonFormatter, setup_logging, request_id_var


def test_json_formatter_outputs_valid_json():
    """JsonFormatter должен возвращать валидную JSON-строку с обязательными полями."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="test.py", lineno=1,
        msg="hello world", args=(), exc_info=None, func="<module>"
    )
    output = formatter.format(record)
    parsed = json.loads(output)

    assert parsed["message"] == "hello world"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test"
    assert parsed["module"] == "test"
    assert parsed["function"] == "<module>"
    assert parsed["line"] == 1
    assert "timestamp" in parsed
    assert "request_id" in parsed


def test_json_formatter_includes_request_id_from_contextvar():
    """При установленном ContextVar request_id должен попадать в лог."""
    formatter = JsonFormatter()
    token = request_id_var.set("req-abc-123")
    try:
        record = logging.LogRecord(
            name="test", level=logging.DEBUG, pathname="test.py", lineno=2,
            msg="with request id", args=(), exc_info=None, func="test_func"
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["request_id"] == "req-abc-123"
    finally:
        request_id_var.reset(token)


def test_setup_logging_configures_root_logger():
    """setup_logging() должен создать хотя бы один handler у root-логгера."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    setup_logging()

    assert len(logging.root.handlers) > 0
    handler = logging.root.handlers[0]
    assert isinstance(handler.formatter, JsonFormatter)


def test_do_logging_import_has_no_side_effects():
    """Импорт utils.do_logging не должен модифицировать root handlers."""
    before = logging.root.handlers.copy()
    import importlib
    import utils.do_logging
    importlib.reload(utils.do_logging)
    after = logging.root.handlers.copy()
    assert before == after
