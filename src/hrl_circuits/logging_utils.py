"""
hrl_circuits.logging_utils
==========================

Structured logging setup for the HRL Circuit Localization project.

Features
--------
* Configurable log level via :func:`setup_logging`.
* Colourised console handler (falls back gracefully if ``colorama`` is absent).
* Optional rotating file handler that writes JSON-structured log records.
* A ``get_logger`` convenience wrapper that prefixes the package namespace.
* A ``JsonFormatter`` that serialises log records as newline-delimited JSON
  for consumption by log aggregation tools (e.g., Loki, CloudWatch).

Usage
-----
::

    from hrl_circuits.logging_utils import setup_logging, get_logger

    setup_logging(level="INFO", log_dir="logs", json_file=True)

    log = get_logger("my_module")
    log.info("Training started", extra={"seed": 42, "task": "ioi"})
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── optional colour support ───────────────────────────────────────────────────
try:
    import colorama  # type: ignore[import]
    colorama.init(autoreset=True)
    _HAS_COLORAMA = True
except ImportError:
    _HAS_COLORAMA = False

# ── package root logger ───────────────────────────────────────────────────────
_PKG_LOGGER_NAME = "hrl_circuits"

# ── ANSI colour codes ─────────────────────────────────────────────────────────
_LEVEL_COLOURS: dict[int, str] = {
    logging.DEBUG: "\033[36m",     # cyan
    logging.INFO: "\033[32m",      # green
    logging.WARNING: "\033[33m",   # yellow
    logging.ERROR: "\033[31m",     # red
    logging.CRITICAL: "\033[35m",  # magenta
}
_RESET = "\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────


class ColourFormatter(logging.Formatter):
    """A formatter that prepends a coloured level name to each log record.

    Falls back to plain text if the terminal does not support ANSI or if
    ``colorama`` is unavailable.

    Attributes
    ----------
    use_colour : bool
        Whether ANSI colour codes are applied.
    """

    _FMT = "{colour}[{levelname:<8}]{reset} {asctime}  {name}  {message}"
    _DATE_FMT = "%Y-%m-%dT%H:%M:%S"

    def __init__(self, use_colour: bool = True) -> None:
        super().__init__(style="{")
        self.use_colour = use_colour and _HAS_COLORAMA

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = _LEVEL_COLOURS.get(record.levelno, "") if self.use_colour else ""
        reset = _RESET if self.use_colour else ""
        fmt = self._FMT.format(
            colour=colour,
            levelname=record.levelname,
            reset=reset,
            asctime="{asctime}",
            name="{name}",
            message="{message}",
        )
        formatter = logging.Formatter(fmt=fmt, datefmt=self._DATE_FMT, style="{")
        return formatter.format(record)


class JsonFormatter(logging.Formatter):
    """A formatter that emits each log record as a JSON object on one line.

    Designed for structured log ingestion (e.g., Loki, Fluentd, CloudWatch).
    All ``extra`` fields passed to the logger are included in the output.

    The JSON schema is::

        {
            "ts": "<ISO-8601 UTC timestamp>",
            "level": "<LEVELNAME>",
            "logger": "<logger name>",
            "msg": "<formatted message>",
            "module": "<module>",
            "lineno": <int>,
            "exc": "<traceback string or null>",
            ... (extra fields)
        }
    """

    # Fields that are part of the standard LogRecord and should not be
    # duplicated under "extra".
    _STANDARD_ATTRS: frozenset[str] = frozenset(
        {
            "args", "asctime", "created", "exc_info", "exc_text",
            "filename", "funcName", "id", "levelname", "levelno",
            "lineno", "message", "module", "msecs", "msg", "name",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "taskName", "thread", "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        record.message = record.getMessage()
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()

        payload: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.message,
            "module": record.module,
            "lineno": record.lineno,
        }

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        else:
            payload["exc"] = None

        # Include any extra fields added via logger.info(…, extra={…})
        for key, value in record.__dict__.items():
            if key not in self._STANDARD_ATTRS:
                try:
                    json.dumps(value)  # only include JSON-serialisable extras
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)

        return json.dumps(payload, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# setup_logging
# ─────────────────────────────────────────────────────────────────────────────


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path | None = None,
    json_file: bool = True,
    console: bool = True,
    use_colour: bool = True,
    max_bytes: int = 50 * 1024 * 1024,  # 50 MB
    backup_count: int = 5,
    propagate: bool = False,
) -> logging.Logger:
    """Initialise the package-level logger with console and/or file handlers.

    Call this once at the start of a training run (typically from the CLI
    entry-point or the ``train`` command).

    Parameters
    ----------
    level : str, optional
        Logging level as a string: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``.  Default is ``"INFO"``.
    log_dir : str | Path | None, optional
        If provided, write logs to ``<log_dir>/hrl_circuits.log`` (plain text)
        and optionally ``<log_dir>/hrl_circuits.jsonl`` (JSON lines).
    json_file : bool, optional
        If ``True`` and ``log_dir`` is set, also write a JSON-lines log file.
    console : bool, optional
        If ``True`` (default), add a ``StreamHandler`` writing to ``stderr``.
    use_colour : bool, optional
        If ``True`` (default), use coloured output for the console handler.
    max_bytes : int, optional
        Maximum bytes per log file before rotation.  Default 50 MB.
    backup_count : int, optional
        Number of rotated log file backups to keep.  Default 5.
    propagate : bool, optional
        Whether the package logger propagates to the root logger.  Default
        is ``False`` to avoid duplicate records.

    Returns
    -------
    logging.Logger
        The configured ``hrl_circuits`` package logger.

    Examples
    --------
    >>> log = setup_logging(level="DEBUG", log_dir="logs")
    >>> log.info("Ready.")
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    pkg_logger = logging.getLogger(_PKG_LOGGER_NAME)
    pkg_logger.setLevel(numeric_level)
    pkg_logger.propagate = propagate

    # Remove any pre-existing handlers (idempotent re-initialisation)
    pkg_logger.handlers.clear()

    # ── console handler ───────────────────────────────────────────────────────
    if console:
        ch = logging.StreamHandler(stream=sys.stderr)
        ch.setLevel(numeric_level)
        ch.setFormatter(ColourFormatter(use_colour=use_colour))
        pkg_logger.addHandler(ch)

    # ── file handlers ─────────────────────────────────────────────────────────
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Plain-text rotating handler
        plain_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / "hrl_circuits.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        plain_handler.setLevel(numeric_level)
        plain_formatter = logging.Formatter(
            fmt="[{levelname:<8}] {asctime}  {name}  {message}",
            datefmt="%Y-%m-%dT%H:%M:%S",
            style="{",
        )
        plain_handler.setFormatter(plain_formatter)
        pkg_logger.addHandler(plain_handler)

        # JSON-lines rotating handler
        if json_file:
            json_handler = logging.handlers.RotatingFileHandler(
                filename=log_path / "hrl_circuits.jsonl",
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            json_handler.setLevel(numeric_level)
            json_handler.setFormatter(JsonFormatter())
            pkg_logger.addHandler(json_handler)

    pkg_logger.debug(
        "Logging initialised: level=%s  log_dir=%s  json=%s",
        level,
        log_dir,
        json_file,
    )
    return pkg_logger


# ─────────────────────────────────────────────────────────────────────────────
# get_logger
# ─────────────────────────────────────────────────────────────────────────────


def get_logger(name: str) -> logging.Logger:
    """Return a child logger namespaced under the ``hrl_circuits`` package.

    Parameters
    ----------
    name : str
        The sub-module or component name.  Automatically prefixed with
        ``"hrl_circuits."`` if not already.

    Returns
    -------
    logging.Logger
        A child logger inheriting handlers and level from the package logger.

    Examples
    --------
    >>> log = get_logger("env.graph_env")
    >>> log.name
    'hrl_circuits.env.graph_env'
    """
    if not name.startswith(_PKG_LOGGER_NAME):
        name = f"{_PKG_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

__all__: list[str] = [
    "setup_logging",
    "get_logger",
    "ColourFormatter",
    "JsonFormatter",
]
