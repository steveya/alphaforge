"""Structured logging for alphaforge and downstream packages.

Usage::

    from alphaforge.logging import configure_logging, get_logger

    # Call once at startup
    configure_logging(level="INFO", format="json")  # or "console"

    # In any module
    log = get_logger(__name__)
    log.info("ingesting data", source="cftc_cot", rows=48)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog

# ---------------------------------------------------------------------------
# Standard context keys
# ---------------------------------------------------------------------------

CTX_SOURCE = "source"
CTX_PIPELINE = "pipeline"
CTX_INSTRUMENT = "instrument"
CTX_ASOF = "asof"
CTX_TABLE = "table"
CTX_PHASE = "phase"
CTX_ROWS = "rows"
CTX_DURATION_MS = "duration_ms"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(
    level: str = "INFO",
    format: str = "console",
    log_file: str | Path | None = None,
) -> None:
    """Configure structured logging for the entire process.

    Parameters
    ----------
    level : minimum log level
    format : ``"console"`` for human-readable or ``"json"`` for JSON lines
    log_file : optional path for JSON-line file output (append mode)
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        json_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
        file_handler = logging.FileHandler(str(log_file), mode="a")
        file_handler.setFormatter(json_formatter)
        root.addHandler(file_handler)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for *name*."""
    return structlog.get_logger(name)
