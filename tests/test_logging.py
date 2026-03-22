"""Tests for alphaforge.logging module."""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import structlog
from structlog.testing import capture_logs

from alphaforge.logging import configure_logging, get_logger


class TestConfigureConsole:
    def test_configure_console(self) -> None:
        configure_logging(level="DEBUG", format="console")
        log = get_logger("test_console")
        log.info("console test")  # should not crash


class TestConfigureJson:
    def test_configure_json(self, capsys) -> None:
        configure_logging(level="DEBUG", format="json")
        log = get_logger("test_json")
        log.info("json test", key="value")


class TestGetLogger:
    def test_get_logger(self) -> None:
        log = get_logger("my_module")
        assert log is not None


class TestBoundContext:
    def test_bound_context(self) -> None:
        configure_logging(level="DEBUG", format="console")
        with capture_logs() as cap:
            log = get_logger("test_bound")
            log.info("hello", source="test")
        events = [e for e in cap if e.get("event") == "hello"]
        assert len(events) == 1
        assert events[0]["source"] == "test"


class TestFileLogging:
    def test_file_logging(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        configure_logging(level="DEBUG", format="console", log_file=str(log_file))
        log = get_logger("test_file")
        log.info("file test", data=42)
        # File should have been written
        assert log_file.exists()
        content = log_file.read_text()
        assert "file test" in content


class TestCaptureLogs:
    def test_capture_logs(self) -> None:
        configure_logging(level="DEBUG", format="console")
        with capture_logs() as cap:
            log = get_logger("test_capture")
            log.info("captured", rows=10)
        events = [e for e in cap if e.get("event") == "captured"]
        assert len(events) == 1
        assert events[0]["rows"] == 10


class TestStdlibIntegration:
    def test_stdlib_integration(self, capsys) -> None:
        configure_logging(level="DEBUG", format="console")
        stdlib_logger = logging.getLogger("stdlib_test")
        stdlib_logger.info("stdlib message")
        # stdlib messages go through structlog's formatter to stderr
        captured = capsys.readouterr()
        assert "stdlib message" in captured.err
